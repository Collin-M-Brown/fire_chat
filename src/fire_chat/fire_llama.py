import os
from fireworks.client import Fireworks
import re
from typing import Generator
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

class Parameters:
    model: str
    max_tokens: int
    messages: list[dict[str, str]]
    temperature: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    n: int
    stream: bool
    stop: set[str]

    def to_dict(self):
        param_dict = vars(self).copy()
        param_dict['stop'] = list(self.stop)
        return param_dict

class fire_llama:

    def __init__(self, 
                model: str = "accounts/fireworks/models/llama-v3-70b-instruct", 
                max_tokens: int = 300, 
                messages: list[dict[str, str]] = None,
                temperature: int = 0.6, 
                top_p: float = 1.0, 
                frequency_penalty: float = 1.0, 
                presence_penalty: float = 0.0, 
                n: int = 1, 
                stream: bool = True, 
                stop: set[str] = None,
                api_key: str = None,
                prompt: dict = None):
        
        if api_key is None:
            api_key = os.environ.get('FIREWORKS_API_KEY')
        if api_key is not None:
            self.client = Fireworks(api_key=api_key).chat.completions
        else:
            print('No API key provided and arguments and no API key found in .env. You can still set one with set_api_key()')
            
        if (messages is None):
            messages = []
        if stop is None:
            stop = {"<|eot_id|>"}

        self.Params = Parameters()
        self.set_model(model)
        self.set_max_tokens(max_tokens)
        self.set_messages(messages)
        self.set_temperature(temperature)
        self.set_top_p(top_p)
        self.set_frequency_penalty(frequency_penalty)
        self.set_presence_penalty(presence_penalty)
        self.set_n(n)
        self.set_stream(stream)
        self.set_stop_tokens(stop)
        if prompt is not None:
            self.set_prompt(prompt)

    def set_api_key(self, api_key: str):
        self.client = Fireworks(api_key=api_key).chat.completions

    def set_model(self, model: str):
        self.Params.model = model
    
    def set_prompt(self, prompt: str):
        if len(self.Params.messages) == 0:
            self.Params.messages.append({'role': 'system', 'content': prompt})
        else:
            #if user, push front, if system, replace
            if self.Params.messages[0]['role'] == 'user':
                self.Params.messages.insert(0, {'role': 'system', 'content': prompt})
            else:
                self.Params.messages[0] = {'role': 'system', 'content': prompt}
    
    def set_max_tokens(self, max_tokens: int):
        self.Params.max_tokens = max_tokens
    
    def set_messages(self, messages: list[dict[str, str]]):
        self.Params.messages = messages
    
    def set_temperature(self, temperature: int):
        self.Params.temperature = temperature
    
    def set_top_p(self, top_p: float):
        self.Params.top_p = top_p
    
    def set_frequency_penalty(self, frequency_penalty: float):
        self.Params.frequency_penalty = frequency_penalty

    def set_presence_penalty(self, presence_penalty: float):
        self.Params.presence_penalty = presence_penalty
    
    def set_n(self, n: int):
        self.Params.n = n

    def set_stream(self, stream: bool):
        self.Params.stream = stream
    
    def set_stop_tokens(self, stop_tokens):
        if isinstance(stop_tokens, list):
            stop_tokens = set(stop_tokens)
        elif not isinstance(stop_tokens, set):
            raise TypeError("stop_tokens must be a set or a list")

        self.Params.stop = stop_tokens
        self.Params.stop.add("<|eot_id|>")

    def _add_message(self, message: dict[str, str]):
        # validate message is a dict with role and content keys
        print(message)
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
        if 'role' not in message:
            raise ValueError("Message must have a role key")
        if 'content' not in message:
            raise ValueError("Message must have a content key")
        
        if len(self.Params.messages) == 0:
            if message['role'] == 'assistant':
                raise ValueError("First message must be from user")
            else:
                self.Params.messages.append(message)
                return
            
        if (len(self.Params.messages) == 1):
            if (self.Params.messages[0]['role'] == 'system' and message['role'] == 'assistant'):
                raise ValueError("First message must be from user")
        
        if (self.Params.messages[-1]['role'] == message['role']):
            self.Params.messages[-1]['content'] = self.Params.messages[-1]['content'].strip() + " " + message['content']
            return

        self.Params.messages.append(message)

    def add_user_message(self, message: str):
        self._add_message({"role": "user", "content": message})

    def add_assistant_message(self, message: str):
        self._add_message({"role": "assistant", "content": message})

    def clear_memory(self):
        self.Params.messages = []

    def clear_prompt(self):
        self.Params.prompt = {}
    
    def remove_last_message(self):
        self.Params.messages.pop()
    
    def get_context(self):
        return self.Params.messages
    
    def get_user_message_length(self):
        print(self.Params.messages)
        if len(self.Params.messages) == 0 or self.Params.messages[0]['role'] != 'user':
            return 0
        return len(self.Params.messages[0]['content'])
    
    # Cleanse content -- emojis, and multiple punctuation marks. Meant to clean the response for audio generation.
    def _process_content(self, content: str):
        # Replace multiple punctuation marks with a single one
        content = re.sub(r'\.{2,}', '.', content)   # Replace multiple periods with one
        content = re.sub(r',{2,}', ',', content)    # Replace multiple commas with one
        content = re.sub(r'\?{2,}', '?', content)   # Replace multiple question marks with one
        content = re.sub(r'\!{2,}', '!', content)   # Replace multiple exclamation marks with

        # Remove emojis
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F700-\U0001F77F"  # alchemical symbols
                                u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                u"\U00002702-\U000027B0"  # Dingbats
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)
        content = emoji_pattern.sub(r'', content)
        return content
    
    def _get_raw_response(self):
        if not getattr(self, 'client', False):
            raise AttributeError("client was not initialized. Make sure set_api_key() was called")

        args = self.Params.to_dict()
        for chunk in self.client.create(**args):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # Generator that yields each sentence in the response
    # Appends response into global context
    def get_easy_response(self) -> Generator[str, None, None]:
        sentences = []
        parts = []
        current_sentence = ""
        sentence_enders = '.?!:;,'
        for chunk in self._get_raw_response():
            parts.append(chunk)
            if (any(punct in chunk for punct in sentence_enders)):
                punct_index = next((i for i, x in enumerate(chunk) if x in sentence_enders), None) 
                leftover = ""
                if punct_index is not None:
                    parts[-1] = chunk[:punct_index + 1]
                    leftover = chunk[punct_index + 1:]
                current_sentence = ''.join(parts)
                current_sentence = self._process_content(current_sentence)
                sentences.append(current_sentence.strip())
                yield current_sentence
                parts = [leftover] if leftover else []
        if parts:
            current_sentence = ''.join(parts)
            current_sentence = self._process_content(current_sentence)
            sentences.append(current_sentence.strip())
            yield current_sentence
        
        self.add_assistant_message(' '.join(sentences))

    def _print_chat(self):
        for message in self.Params.messages:
            print(f"{message['role']}: {message['content']}")
