import threading
import io
import pygame
import time
from queue import Queue, Empty
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv()

class easy_tts:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
        
        try:
            pygame.mixer.init()
        except Exception as e:
            logging.error(f"Pygame mixer initialization failed: {e}")
            raise
        
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
                    pygame.mixer.music.load(audio_io)
                    pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error playing audio: {e}")
                self._reset_pygame_mixer()

    def _reset_pygame_mixer(self):
        with self.lock:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            pygame.mixer.quit()
            pygame.mixer.init()

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
        self.text_queue.put((text, voice))

    def stop(self):
        self.stop_event.set()
        self.play_thread.join(timeout=5)
        self.write_thread.join(timeout=5)
        pygame.mixer.quit()

    def cleanup(self):
        self.stop()

    def __del__(self):
        try:
            self.cleanup()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")