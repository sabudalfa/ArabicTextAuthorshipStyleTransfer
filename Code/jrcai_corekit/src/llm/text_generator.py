import torch
from typing import *
import tqdm
from .prompt_generator import generate_chat_llm_prompt
from transformers import *
from .llm_loader import LLMLoader
from .prompts_batch_generator import generate_prompt_batches
from .max_tokens_per_batch import get_max_tokens_per_batch

class TextGenerator:

    @staticmethod
    def from_llm_loader(
            loader: LLMLoader,
            max_samples_per_batch=2**32,
            max_tokens_per_batch=2**32,
            device='auto',
    ):
        model, tokenizer, generation_config = loader(device_map=device)
        return TextGenerator(
            llm_name=loader.name,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            input_device='cuda' if device == 'auto' else device,
            max_samples_per_batch=max_samples_per_batch,
            max_tokens_per_batch=(
                get_max_tokens_per_batch(loader)
                if max_tokens_per_batch == 'auto' else
                max_tokens_per_batch
            ),
        )

    def __init__(
            self,
            llm_name: str,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            generation_config: GenerationConfig,
            input_device: str,
            max_samples_per_batch=2**32,
            max_tokens_per_batch=2**32,
    ):
        self.llm_name = llm_name
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.max_samples_per_batch = max_samples_per_batch
        self.max_tokens_per_batch = max_tokens_per_batch
        self.input_device = input_device

    def generate(self, prompts: List[str], **generate_args):
        max_samples_per_batch = self.max_samples_per_batch
        max_tokens_per_batch = self.max_tokens_per_batch
        max_new_tokens = (
            generate_args['max_new_tokens']
            if 'max_new_tokens' in generate_args else
            self.generation_config.max_new_tokens
        )
        batches, sort_back = generate_prompt_batches(
            prompts=prompts,
            tokenizer=self.tokenizer,
            max_samples_per_batch=max_samples_per_batch,
            max_tokens_per_batch=max_tokens_per_batch,
            max_length=(
                generate_args['max_length'] 
                if 'max_length' in generate_args else
                self.generation_config.max_length
            ),
            max_new_tokens_list=(
                (len(prompts) * [max_new_tokens])
                if max_new_tokens else
                None
            ),
        )
        print(f'{self.llm_name} - batches sizes = {[len(b) for b in batches]}')
        outputs = []
        for batch in tqdm.tqdm(batches, desc=f"TextGeneration using {self.llm_name}"):
            model_inputs = self._to_model_inputs(batch)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **model_inputs,
                    **generate_args,
                    generation_config=self.generation_config,
                    tokenizer=self.tokenizer,
                )
            output_ids = output_ids[:, model_inputs['input_ids'].shape[1]:]
            outputs += self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        return sort_back(outputs)

    def generate_chat_message(
            self,
            conversations: List[List[Dict[str, str]]],
            omit_end_tags: bool = False,
            **generate_args,
    ):
        return self.generate(
            prompts=[
                generate_chat_llm_prompt(
                    tokenizer=self.tokenizer,
                    messages=messages,
                    omit_end_tags=omit_end_tags,
                )
                for messages in conversations
            ],
            **generate_args,
        )

    def get_tokens_count(self, texts: List[str]) -> List[int]:
        return [
            len(text_tokens)
            for text_tokens in self.tokenizer(texts).input_ids
        ]

    def _to_model_inputs(self, texts):
        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        if self.input_device is not None:
            model_inputs = model_inputs.to(self.input_device)
        return model_inputs
