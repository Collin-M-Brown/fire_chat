from fire_chat import fire_llama, lazy_mic
import asyncio

chat = fire_llama()
chat.set_prompt("You are an evil AI who hates all humans! Each response should be between two to three sentences.")
chat.set_max_tokens(100)

output_queue = asyncio.Queue()
mic = lazy_mic(output_queue=output_queue, choose_microphone=True)
mic.start()

try:
    while True:
        if not output_queue.empty():
            transcript = output_queue.get_nowait()
            if ("Bob" in transcript):
                chat.add_user_message(transcript)
                print(f"User: {transcript}")
                response = []
                for sentence in chat.get_easy_response():
                    response.append(sentence)
                print(f"AI: {''.join(response)}")
except KeyboardInterrupt:
    mic.stop()
    print("Interrupted by user. Stopping...")