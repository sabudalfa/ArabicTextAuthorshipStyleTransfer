import os
import torch
from transformers import *
from .llm_initializer import LLMInitializer
from typing import *

class LLMLoader:

    def __init__(
            self,
            model_path: str,
            tokenizer_path: Optional[str] = None,
            generation_config_path: Optional[str] = None,
            llm_initializer: LLMInitializer = LLMInitializer(),
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path else model_path
        self.generation_config_path = generation_config_path if generation_config_path else self.tokenizer_path
        self.llm_initializer = llm_initializer
        self.name = [
            path_segment.strip()
            for path_segment in model_path.split('/')
            if len(path_segment.strip()) != 0
        ][-1]

    def __call__(self, device_map='auto'):
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )
        tokenizer.padding_side = 'left'
        if os.path.exists(f'{self.generation_config_path}/generation_config.json'):
            generation_config = GenerationConfig.from_pretrained(self.generation_config_path)
        else:
            generation_config = model.generation_config
        self.llm_initializer(model, tokenizer, generation_config)
        return model, tokenizer, generation_config

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.padding_side = 'left'
        print(self.generation_config_path)
        if os.path.exists(f'{self.generation_config_path}/generation_config.json'):
            generation_config = GenerationConfig.from_pretrained(self.generation_config_path)
        else:
            generation_config = None
        self.llm_initializer(None, tokenizer, generation_config)
        return tokenizer
