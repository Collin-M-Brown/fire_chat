from fire_chat import fire_llama, easy_stt, easy_tts
import asyncio

output_queue = asyncio.Queue()

speakers = easy_tts()
chat = fire_llama()
mic = easy_stt(output_queue=output_queue, choose_microphone=True)
mic.start()

chat.set_prompt("You are an evil AI who hates all humans! Each response should be between two to three sentences.")
chat.set_max_tokens(100)
mic.set_activation_words(["Bob"])
try:
    while True:
        if not output_queue.empty():
            transcript = output_queue.get_nowait()
            chat.add_user_message(transcript)
            print(f"User: {transcript}")
            response = []
            for sentence in chat.get_easy_response():
                response.append(sentence)
                speakers.speak(sentence)
            print(f"AI: {''.join(response)}")
except KeyboardInterrupt:
    mic.stop()
    print("Interrupted by user. Stopping...")