
"""intent_classifier.py

The LLM's ONLY job here is to output structured JSON.
It does NOT call tools. It does NOT generate responses.
It just classifies what the user wants.
"""

import json
import os
import re
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
logger = logging.getLogger("wpa.intent")

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME = os.getenv("OLLAMA_MODEL")   # change to your model

_llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0,          # zero temp = deterministic, no creativity needed
    base_url="http://localhost:11434",
    format="json",            # force JSON output mode
)

# ── The complete tool catalogue given to the classifier ───────────────────────
# The LLM only needs to know tool names and what triggers them.
# It does NOT need implementation details.
TOOL_CATALOGUE = """
AVAILABLE TOOLS:
- open_app(app_name: str)          → open an application
- close_app(app_name: str)         → close a running application  
- list_running_apps()              → list what apps are open
- search_files(filename, extension)→ find files on the PC
- read_file(file_path)             → read contents of a file
- write_file(file_path, content, mode) → create or append to a file
- delete_file(file_path)           → delete a file permanently
- list_directory(folder_path)      → list files in a folder
- get_system_info()                → CPU, RAM, battery, disk, time
- get_battery_status()             → battery percentage and charging state
- set_volume(level: int)           → set system volume 0-100
- get_clipboard()                  → read clipboard contents
- set_clipboard(text)              → copy text to clipboard
- get_current_time()               → current date and time
- open_url(url)                    → open URL in browser
- web_search(query)                → search the web
- search_and_open(query)           → search web and open top result
- take_screenshot(save_path)       → capture the screen
- read_screen_text()               → OCR all visible text on screen
"""

# ── The classifier prompt ─────────────────────────────────────────────────────
CLASSIFIER_PROMPT = f"""
You are an intent classifier for a Windows voice assistant.
Your ONLY job is to output valid JSON. Nothing else.

{TOOL_CATALOGUE}

INTENTS:
- "tool_use"  → user wants to do something on the PC (use a tool)
- "chat"      → general question or conversation (no tool needed)
- "clarify"   → request is too vague to determine which tool
- "confirm"   → user is saying yes/confirming a previous action
- "cancel"    → user is saying no/cancelling a previous action

OUTPUT FORMAT (strict JSON, nothing else):
{{
  "intent": "tool_use" | "chat" | "clarify" | "confirm" | "cancel",
  "tool_name": "exact_tool_name_or_empty_string",
  "tool_args": {{}},
  "confidence": 0.0-1.0,
  "raw_query": "the original user input"
}}

RULES:
- If intent is "chat", set tool_name to "" and tool_args to {{}}
- If intent is "tool_use", always set tool_name to one of the exact tool names above
- For set_volume, extract the number from the query for the level arg
- For open_app/close_app, extract the app name for app_name arg
- For write_file, extract file path and content
- Confidence should reflect how certain you are (0.9+ for clear requests)

EXAMPLES:
User: "open notepad"
{{"intent":"tool_use","tool_name":"open_app","tool_args":{{"app_name":"notepad"}},"confidence":0.99,"raw_query":"open notepad"}}

User: "what time is it"
{{"intent":"tool_use","tool_name":"get_current_time","tool_args":{{}},"confidence":0.98,"raw_query":"what time is it"}}

User: "hello how are you"
{{"intent":"chat","tool_name":"","tool_args":{{}},"confidence":0.97,"raw_query":"hello how are you"}}

User: "set volume to 40"
{{"intent":"tool_use","tool_name":"set_volume","tool_args":{{"level":40}},"confidence":0.99,"raw_query":"set volume to 40"}}

User: "yes go ahead"
{{"intent":"confirm","tool_name":"","tool_args":{{}},"confidence":0.95,"raw_query":"yes go ahead"}}

User: "no cancel that"
{{"intent":"cancel","tool_name":"","tool_args":{{}},"confidence":0.95,"raw_query":"no cancel that"}}

User: "do the thing"
{{"intent":"clarify","tool_name":"","tool_args":{{}},"confidence":0.3,"raw_query":"do the thing"}}

When extracting app names:
- Correct obvious spelling mistakes
- Normalize common aliases
- Use the closest valid application name when confidence is high

Examples:
"open chromee" -> open_app(app_name="chrome")
"launch vlcc" -> open_app(app_name="vlc")
"open calcultor" -> open_app(app_name="calculator")
"""
# def build_classifier_prompt_with_memory(
    
# ) -> str:

#     """Build the classifier prompt, optionally injecting memory context."""

    

#     return prompt

def classify_intent(user_input: str,memory_context: str = "") -> dict:
    """
    Classify user input into a structured intent dict.
    Returns a safe default if LLM fails or returns invalid JSON.
    This function NEVER raises — always returns a valid dict.
    """
    # logger.info(f"[classifier] input: '{user_input}'")
    prompt = CLASSIFIER_PROMPT.replace(
        "{tool_catalogue}",
        TOOL_CATALOGUE
    )

    if memory_context:

        prompt = (

            f"MEMORY CONTEXT "
            f"(use this to better understand the user):\\n"

            f"{memory_context}\\n\\n"

            + prompt
        )

    try:
        messages = [
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=f'Classify this: "{user_input}"'),
        ]

        response = _llm.invoke(messages)

        raw = response.content.strip()

        # logger.debug(f"[classifier] raw LLM output: {raw}")

        # Strip markdown fences if model wrapped JSON in ```json ... ```
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        raw = raw.strip()

        result = json.loads(raw)

        # Validate required fields exist
        required = [
            "intent",
            "tool_name",
            "tool_args",
            "confidence",
            "raw_query"
        ]

        for field in required:
            if field not in result:
                raise ValueError(f"Missing field: {field}")

        # Validate intent is a known value
        valid_intents = {
            "tool_use",
            "chat",
            "clarify",
            "confirm",
            "cancel"
        }

        if result["intent"] not in valid_intents:
            raise ValueError(f"Unknown intent: {result['intent']}")

        # logger.info(
        #     f"[classifier] intent={result['intent']} "
        #     f"tool={result['tool_name']} "
        #     f"args={result['tool_args']} "
        #     f"confidence={result['confidence']:.2f}"
        # )

        return result

    except json.JSONDecodeError as e:
        logger.error(
            f"[classifier] JSON parse failed: {e} | raw='{raw}'"
        )

    except Exception as e:
        logger.error(f"[classifier] failed: {e}")

    # ── Safe fallback — never crash the pipeline ──────────────────────────────
    # logger.warning("[classifier] using fallback: chat intent")

    return {
        "intent": "chat",
        "tool_name": "",
        "tool_args": {},
        "confidence": 0.0,
        "raw_query": user_input,
    }
