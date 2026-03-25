
_JAIS_CHAT_TEMPLATE = "{{ bos_token }}{% for message in messages %}{{ '# ' + message['role'] + ':\n'  + message['content'] + '\n\n' }}{% endfor %}{% if add_generation_prompt %}# assistant:\n{% endif %}"


class LLMInitializer:

    def __init__(self, **generation_config):
        self.generation_config = generation_config


    def __call__(self, model, tokenizer, generation_config):
        if generation_config:
            generation_config.update(**self.generation_config)


class Llama3Initializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if generation_config:
            generation_config.pad_token_id = tokenizer.pad_token_id


class Llama3InstructInitializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if generation_config:
            generation_config.eos_token_id = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            generation_config.pad_token_id = tokenizer.pad_token_id


class Llama31Initializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if generation_config:
            generation_config.pad_token_id = tokenizer.pad_token_id


class JAISInitializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        if generation_config:
            generation_config.max_length = 2048
            generation_config.bos_token_id = tokenizer.bos_token_id
            generation_config.pad_token_id = tokenizer.pad_token_id
            generation_config.eos_token_id = tokenizer.eos_token_id


class JAISChatInitializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        if generation_config:
            generation_config.max_length = 2048
            generation_config.bos_token_id = tokenizer.bos_token_id
            generation_config.pad_token_id = tokenizer.pad_token_id
            generation_config.eos_token_id = tokenizer.eos_token_id
            generation_config.stop_strings = ["# user:"]
        tokenizer.chat_template = _JAIS_CHAT_TEMPLATE


class JAISAdapted70BChatInitializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if generation_config:
            generation_config.max_length = 2048


class JAISFamily30B16KChatInitializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        if generation_config:
            generation_config.max_length = 16_000


class JAISFamily13BChatInitializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        if generation_config:
            generation_config.max_length = 2048


class Llama2Initializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        tokenizer.pad_token_id = tokenizer.eos_token_id


class Gemma2Initializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        if generation_config:
            generation_config.max_length = 8096


class AceGPT2Initializer(LLMInitializer):

    def __call__(self, model, tokenizer, generation_config):
        super().__call__(model, tokenizer, generation_config)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if generation_config:
            generation_config.pad_token_id = tokenizer.pad_token_id
