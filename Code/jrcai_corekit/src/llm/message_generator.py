from abc import ABC, abstractmethod
from typing import *
import time
from openai import OpenAI
from anthropic import Anthropic
from .llm_loader import LLMLoader
from .text_generator import TextGenerator
import os
import json
import requests


class MessageGenerator(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, conversations) -> List[str]:
        pass


class MessageGeneratorFromLocalLLM(MessageGenerator):

    def __init__(self, llm_loader: LLMLoader):
        self.llm_loader = llm_loader
        self.text_generator = None

    @property
    def name(self) -> str:
        return self.llm_loader.name

    def __call__(self, conversations) -> List[str]:
        if self.text_generator is None:
            self.text_generator =  TextGenerator.from_llm_loader(
                loader=self.llm_loader,
                #max_tokens_per_batch='auto',
                max_samples_per_batch='1',
            )
        return self.text_generator.generate_chat_message(
            conversations,
            max_new_tokens=512,
        )


class MessageGeneratorFromOpenAIAPI(MessageGenerator):

    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name

    @property
    def name(self) -> str:
        return self.model_name

    def __call__(self, conversations) -> List[str]:
        client = OpenAI()
        return [
            client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            ).choices[0].message.content
            for messages in conversations
        ]


class MessageGeneratorFromAnthropicAPI:

    def __init__(self, model_name="claude-3-5-sonnet-20240620"):
        self.model_name = model_name

    @property
    def name(self) -> str:
        return self.model_name

    def __call__(self, conversations) -> List[str]:
        client = Anthropic()
        return [
            self._call_anthropic_api(client, messages)
            for messages in conversations
        ]

    def _call_anthropic_api(self, client, messages):
        sleep_period = 60
        while True:
            try:
                return client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=messages,
                ).content[0].text
            except Exception as e:
                print(f'sleep for {sleep_period} seconds', e)
                time.sleep(sleep_period)
                sleep_period *= 2
                if sleep_period > 30 * 60:
                    sleep_period = 30 * 60


class MessageGeneratorFromAllam(MessageGenerator):

    def __init__(self, api_key=None):
        if api_key:
            self.api_key = api_key
        elif 'ALLAM_TOKEN' in os.environ:
            self.api_key = os.environ['ALLAM_TOKEN']
        else:
            self.api_key = None

    @property
    def name(self) -> str:
        return "allam"

    def __call__(self, conversations) -> List[str]:
        return [
            self._call_allam_and_retry_on_error(messages)
            for messages in conversations
        ]

    def _call_allam_and_retry_on_error(self, messages):
        sleep_period = 1
        while True:
            try:
                return self._call_allam(messages)
            except:
                print(f'sleep for {sleep_period} seconds')
                time.sleep(sleep_period)
                sleep_period *= 2
                if sleep_period > 5 * 60:
                    sleep_period = 5 * 60

    def _call_allam(self, messages):
        payload = {
            "messages": messages,
            "temperature": 0.6,
            "stream": False,
            "model": "allam",
            "top_p": 0.98,
            "n": 1,
            "add_generation_prompt": True,
            "echo": False,
            "stop": " </s>",
        }
        response = requests.post(
            'https://vllm-v19.allam.ai/v1/chat/completions',
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            data=json.dumps(payload),
            timeout=150,
            verify=False,
        )
        if response.status_code == 200:
            chat_response_data = response.json()
            return chat_response_data['choices'][0]['message']['content']
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)
            raise Exception()
