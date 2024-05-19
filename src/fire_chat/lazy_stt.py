#%%
import os
import pyaudio
import asyncio
import json
import websockets
from dotenv import load_dotenv
load_dotenv()

class lazy_stt:
    def __init__(self, output_queue: asyncio.Queue, key: str = None, model: str = "nova-2", host: str = "wss://api.deepgram.com", sample_rate: int = 16000, choose_microphone: bool = False):
        if key is None:
            key = os.environ.get('DEEPGRAM_API_KEY')
        if key is None:
            raise ValueError("No API key provided and no API key found in .env")
        self.key = key
        self.model = model
        self.host = host
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate / 10)
        self.audio_queue = asyncio.Queue()
        self.output_queue = output_queue
        self.running = False
        self.loop = asyncio.get_event_loop()
        self.choose_microphone = choose_microphone  

    def mic_callback(self, input_data, frame_count, time_info, status_flag):
        self.audio_queue.put_nowait(input_data)
        return (input_data, pyaudio.paContinue)

    async def run(self):
        deepgram_url = f'{self.host}/v1/listen?punctuate=true'

        if self.model:
            deepgram_url += f"&model={self.model}"
        deepgram_url += "&interim_results=true"
        deepgram_url += "&vad_events=true"
        deepgram_url += "&encoding=linear16&sample_rate=16000"
        deepgram_url += "&filler_words=true"

        async with websockets.connect(
            deepgram_url, extra_headers={"Authorization": "Token {}".format(self.key)}
        ) as ws:
            print(f'ℹ️  Request ID: {ws.response_headers.get("dg-request-id")}')
            if self.model:
                print(f'ℹ️  Model: {self.model}')

            async def sender(ws):
                try:
                    while self.running:
                        mic_data = await self.audio_queue.get()
                        await ws.send(mic_data)
                except websockets.exceptions.ConnectionClosedOK:
                    await ws.send(json.dumps({"type": "CloseStream"}))
                except Exception as e:
                    print(f"Error while sending: {str(e)}")
                    raise

            async def receiver(ws):
                print(f'🟢 START.')
                async for msg in ws:
                    res = json.loads(msg)
                    if res.get('type') == 'Results':
                        transcript = res.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        if res.get("speech_final") or res.get("is_final"):
                            if self.output_queue is not None:
                                if (transcript != ""):
                                    await self.output_queue.put(transcript)

            async def microphone():
                audio = pyaudio.PyAudio()
                default_input_device_info = audio.get_default_input_device_info()
                print(f"Default input device: {default_input_device_info['name']}")
                selected_device_index = default_input_device_info["index"]
                if self.choose_microphone:
                    print("Available input devices:")
                    num_devices = audio.get_device_count()
                    for i in range(num_devices):
                        device_info = audio.get_device_info_by_index(i)
                        if device_info["maxInputChannels"] > 0:
                            print(f"Device {i}: {device_info['name']}")
                    selected_device_index = int(input("Enter the device number to use: "))

                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                    input_device_index=selected_device_index,
                    stream_callback=self.mic_callback,
                )
                stream.start_stream()

                while self.running:
                    await asyncio.sleep(0.1)

                stream.stop_stream()
                stream.close()
                audio.terminate()

            functions = [
                asyncio.ensure_future(sender(ws)),
                asyncio.ensure_future(receiver(ws)),
                asyncio.ensure_future(microphone()),
            ]
            await asyncio.gather(*functions)

    def start(self):
        self.running = True
        self.loop.run_in_executor(None, asyncio.run, self.run())

    def stop(self):
        self.running = False

    async def wait_for_stop(self):
        while self.running:
            await asyncio.sleep(0.1)

    def __del__(self):
        self.stop()
        self.loop.run_until_complete(self.wait_for_stop())


if __name__ == '__main__':
    output_queue = asyncio.Queue()
    mic = lazy_stt(output_queue=output_queue, choose_microphone=True)
    mic.start()

    try:
        while True:
            if not output_queue.empty():
                transcript = output_queue.get_nowait()
                print(transcript)
    except KeyboardInterrupt:
        mic.stop()
        print("Interrupted by user. Stopping...")