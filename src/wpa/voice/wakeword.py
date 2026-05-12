# wake word module 
import os
import time
import logging
import numpy as np
import sounddevice as sd
from openwakeword import get_pretrained_model_paths
from openwakeword.model import Model as WakeWordModel

logger = logging.getLogger(__name__)

BLOCK_SAMPLES = 1280
SAMPLE_RATE   = 16000

class WakeWordDetector:
    """
    Loads model once, reusable across multiple listen() calls.
    Best practice: instantiate once at app startup, call .listen() in loop.
    """
    
    def __init__(self, wake_word: str = "hey_jarvis", threshold: float = None):
        self.wake_word = wake_word
        self.threshold = threshold if threshold is not None else 0.01

        try:
            # Load model
            paths = get_pretrained_model_paths()
            model_path = next((p for p in paths if wake_word in p.lower()), None)
            if not model_path:
                available = [os.path.basename(p) for p in paths]
                raise FileNotFoundError(f"'{wake_word}' not found. Available: {available}")
            
            self._model = WakeWordModel(
                wakeword_model_paths=[model_path],
                enable_speex_noise_suppression=False
            )
            
            # Discover output key
            warmup = self._model.predict(np.zeros(BLOCK_SAMPLES, dtype=np.int16))
            self._key = next(
                (k for k in warmup if wake_word in k.lower()),
                list(warmup.keys())[0]
            )
            
            # logger.info(f"✅ WakeWordDetector initialized")
            logger.debug(f"   wake_word: {wake_word}, key: {self._key}, threshold: {self.threshold:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WakeWordDetector: {e}", exc_info=True)
            raise
    
    def listen(self, timeout: int = 30, verbose: bool = True) -> bool:
        """
        Listen for wake word. Returns True if detected, False if timeout.
        Automatically skips audio while TTS is speaking to avoid false detections.
        
        Args:
            timeout: Maximum seconds to listen
            verbose: Print debug output
        
        Returns:
            True if wake word detected, False if timeout
        """
        if verbose:
            # logger.info(f"👂 Listening for '{self.wake_word}'... (timeout={timeout}s)")
            pass
        
        window = []
        MAX_WIN = 3
        start = time.time()
        max_score = 0.0
        tts_was_speaking = False  # Track if we skipped audio due to TTS
        
        try:
            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                                blocksize=BLOCK_SAMPLES, dtype='int16') as stream:
                while time.time() - start < timeout:
                    try:
                        # Check if TTS is currently speaking
                        try:
                            from src.wpa.voice.tts import get_speaking_status
                            is_tts_speaking = get_speaking_status()
                        except:
                            is_tts_speaking = False
                        
                        # Skip audio chunks while TTS is playing
                        if is_tts_speaking:
                            if not tts_was_speaking:
                                # logger.debug("🔇 TTS detected - skipping audio to avoid false wake word detection")
                                tts_was_speaking = True
                            # Still read from stream to keep it flowing, but don't process
                            chunk, _ = stream.read(BLOCK_SAMPLES)
                            continue
                        
                        if tts_was_speaking:
                            # logger.debug("🔊 TTS finished - resuming wake word detection")
                            tts_was_speaking = False
                        
                        # Normal wake word detection
                        chunk, _ = stream.read(BLOCK_SAMPLES)
                        score = self._model.predict(chunk.flatten())[self._key]
                        
                        # Smoothing window
                        window.append(score)
                        if len(window) > MAX_WIN:
                            window.pop(0)
                        smoothed = sum(window) / len(window)
                        max_score = max(max_score, score)
                        
                        if verbose and score > self.threshold * 0.3:
                            # logger.debug(f"Score: {score:.4f} (smoothed: {smoothed:.4f})")
                            pass
                        
                        if smoothed >= self.threshold:
                            if verbose:
                                # logger.info(f"🟢 Wake word detected! (score: {score:.4f})")
                                pass
                            return True
                    
                    except Exception as e:
                        logger.warning(f"Error processing audio chunk: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Audio stream error: {e}", exc_info=True)
            return False
        
        if verbose:
            logger.info(f"⏰ Timeout. Max score: {max_score:.4f} (threshold: {self.threshold:.4f})")
        return False
    
    # def profile_threshold(self, silence_s: int = 5, speech_s: int = 8) -> float:
    #     """
    #     Auto-calibrate threshold for this mic + environment.
    #     Call once after init if you want automatic calibration.
    #     """
    #     print(f"\n🔧 Calibrating... Stay SILENT for {silence_s}s, then say '{self.wake_word}' for {speech_s}s")
        
    #     silence_scores, speech_scores = [], []
    #     start = time.time()
    #     phase = "silence"
        
    #     with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
    #                         blocksize=BLOCK_SAMPLES, dtype='int16') as stream:
    #         while time.time() - start < silence_s + speech_s:
    #             elapsed = time.time() - start
                
    #             if phase == "silence" and elapsed >= silence_s:
    #                 phase = "speech"
    #                 print(f"\n→ Now say '{self.wake_word}' clearly!")
                
    #             chunk, _ = stream.read(BLOCK_SAMPLES)
    #             score = self._model.predict(chunk.flatten())[self._key]
                
    #             if phase == "silence":
    #                 silence_scores.append(score)
    #             else:
    #                 speech_scores.append(score)
    #                 print(f"  {score:.4f}", end="\r")
        
    #     noise_floor = max(silence_scores) if silence_scores else 0.001
    #     speech_peak = max(speech_scores)  if speech_scores  else 0.01
        
    #     # Set threshold midway between noise floor and peak
    #     self.threshold = noise_floor + (speech_peak - noise_floor) * 0.4
        
    #     print(f"\n📊 Noise floor: {noise_floor:.4f}  Speech peak: {speech_peak:.4f}")
    #     print(f"✅ Threshold set to: {self.threshold:.4f}")
    #     return self.threshold
    
    def listen(self, timeout: int = 30, verbose: bool = True) -> bool:
        """Listen for wake word. Returns True if detected."""
        if verbose:
            print(f"👂 Listening for '{self.wake_word}'... (timeout={timeout}s)")
        
        window, MAX_WIN = [], 3
        start, max_score = time.time(), 0.0
        
        try:
            with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                                blocksize=BLOCK_SAMPLES, dtype='int16') as stream:
                while time.time() - start < timeout:
                    chunk, _ = stream.read(BLOCK_SAMPLES)
                    score = self._model.predict(chunk.flatten())[self._key]
                    
                    window.append(score)
                    if len(window) > MAX_WIN:
                        window.pop(0)
                    smoothed = sum(window) / len(window)
                    max_score = max(max_score, score)
                    
                    if verbose and score > self.threshold * 0.3:
                        print(f"  {score:.4f} smooth={smoothed:.4f}", end="\r")
                    
                    if smoothed >= self.threshold:
                        if verbose:
                            print(f"\n🟢 Detected! score={score:.4f}")
                        return True
        except Exception as e:
            print(f"\n❌ {e}")
            return False
        
        if verbose:
            print(f"\n⏰ Timeout. Max score: {max_score:.4f} (threshold: {self.threshold:.4f})")
        return False


# ── Usage ─────────────────────────────────────────────────────────────────────
# wwd = WakeWordDetector(wake_word="hey_jarvis", threshold=0.25)

# Optional: auto-calibrate for your mic
# wwd.profile_threshold()