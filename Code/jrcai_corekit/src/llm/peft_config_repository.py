from peft import *


class LoRAConfigRepository:

    @staticmethod
    def llama_3(r: int = 16, lora_alpha: int = 32, lora_dropout = 0.05):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['v_proj', 'q_proj'],
        )

    @staticmethod
    def jais_v1(r: int = 16, lora_alpha: int = 32, lora_dropout=0.05):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['c_attn'],
        )
