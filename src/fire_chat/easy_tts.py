import threading
import io
import pygame
import time
from queue import Queue, Empty
from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

load_dotenv()

class easy_tts:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("No API key provided and no API key found in .env")
        self.client = OpenAI(api_key=api_key)
        self.mixer_initialized = False
        
        try:
            pygame.mixer.init()
            self.mixer_initialized = True
        except Exception as e:
            logging.error(f"Pygame mixer initialization failed: {e}")
        
        self.text_queue = Queue()
        self.audio_queue = Queue()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.play_thread = threading.Thread(target=self._play_audio, daemon=True)
        self.play_thread.start()
        self.write_thread = threading.Thread(target=self._generate_audio_chunks, daemon=True)
        self.write_thread.start()

    def _generate_audio_chunks(self):
        while not self.stop_event.is_set():
            try:
                text, voice = self.text_queue.get(timeout=1)
                audio_io = self._openai_tts(text, voice)
                with self.lock:
                    self.audio_queue.put((audio_io, text))
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error generating audio chunks: {e}")

    def _play_audio(self):
        while not self.stop_event.is_set():
            try:
                audio_io, text = self.audio_queue.get(timeout=1)
                with self.lock:
                    if self.mixer_initialized:
                        pygame.mixer.music.load(audio_io)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() and not self.stop_event.is_set():
                            time.sleep(0.1)
                    else:
                        logging.error("Mixer not initialized, cannot play audio.")
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error playing audio: {e}")
                self._reset_pygame_mixer()

    def _reset_pygame_mixer(self):
        with self.lock:
            try:
                if self.mixer_initialized:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    pygame.mixer.quit()
                    self.mixer_initialized = False
            except Exception as e:
                logging.error(f"Error resetting pygame mixer: {e}")

    def _openai_tts(self, text, voice):
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="opus"
        )
        audio_io = io.BytesIO(response.content)
        return audio_io

    def speak(self, text, voice='nova'):
        if self.mixer_initialized:
            self.text_queue.put((text, voice))
        else:
            logging.error("Mixer not initialized, cannot enqueue text.")

    def stop(self):
        self.stop_event.set()
        with self.lock:
            if self.mixer_initialized:
                pygame.mixer.music.stop()  # Force stop the audio playback
                pygame.mixer.quit()
                self.mixer_initialized = False
        self.play_thread.join(timeout=5)
        self.write_thread.join(timeout=5)

    def cleanup(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        print("Cleaned up easy_tts")

# Example usage
if __name__ == "__main__":
    with easy_tts() as speakers:
        speakers.speak("Hello, world!")
        time.sleep(5) 