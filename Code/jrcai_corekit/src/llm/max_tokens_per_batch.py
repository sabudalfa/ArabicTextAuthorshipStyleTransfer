import torch
from llm import LLMLoader

# Results obtained from an experiment
_MAX_TOKENS_PER_BATCH_DICT = {
    'NVIDIA A100-SXM4-80GB': {
        1: {
            'AceGPT-7B': 74400,
            'AceGPT-13B': 38300,
            'c4ai-command-r-v01': 4700,
            'gemma-2-9b-it': 34200,
            'gemma-2-27b-it': 15200,
            'jais-13b': 26700,
            'jais-family-13b': 26700, # JUST ASSUMED THE VALUE, IT WAS NOT TESTED
            'jais-30b-v1': 6400,
            'AceGPT-v2-8B': 72700,
            'Meta-Llama-3-8B': 72700,
            'Llama-2-7b-chat-hf': 72700,
            'Meta-Llama-3.1-8B': 72700,
            'Mistral-Nemo-Instruct-2407': 58800,
            'Phi-3.5-mini-instruct': 50000, # JUST ASSUMED THE VALUE, IT WAS NOT TESTED
            'Phi-3-mini-4k-instruct': 59200,
            'Phi-3-small-8k-instruct': 71900,
            'Phi-3-medium-4k-instruct': 57400,
            'Qwen2-7B-Instruct': 66900,
        },
        2: {
            'AceGPT-7B': 100000,
            'AceGPT-13B': 93600,
            'Phi-3.5-MoE-instruct': 10000, # TODO: I just assumed this, FIND IT
            'c4ai-command-r-v01': 20300,
            'gemma-2-9b-it': 42900,
            'gemma-2-27b-it': 32100,
            'jais-13b': 64900,
            'jais-family-13b-chat': 64900, # JUST ASSUMED THE VALUE, IT WAS NOT TESTED
            'jais-30b-v1': 29100,
            'jais-family-30b-16k-chat': 29100,
            'jais-30b-chat-v1': 29100,
            'Meta-Llama-3-8B': 84900,
            'Llama-2-7b-chat-hf': 84900,
            'Meta-Llama-3-70B': 13700,
            'Meta-Llama-3.1-8B': 84900,
            'Meta-Llama-3.1-70B': 13700,
            'Mistral-Nemo-Instruct-2407': 75400,
            'Phi-3-mini-4k-instruct': 94800,
            'Phi-3-small-8k-instruct': 1200,
            'Phi-3-medium-4k-instruct': 96800,
            'Qwen2-7B-Instruct': 78100,
            'Qwen2-72B-Instruct': 13700, # JUST ASSUMED THE VALUE, IT WAS NOT TESTED
        },
        3: {
            'AceGPT-7B': 100000,
            'AceGPT-13B': 100000,
            'c4ai-command-r-v01': 29300,
            'gemma-2-9b-it': 47300,
            'gemma-2-27b-it': 39700,
            'jais-13b': 91200,
            'jais-30b-v1': 48700,
            'jais-adapted-70b-chat': 30600, # Copied from llama
            'Meta-Llama-3-8B': 92100,
            'Llama-2-7b-chat-hf': 92100,
            'Meta-Llama-3-70B': 40600,
            'Meta-Llama-3.1-8B': 92100,
            'Meta-Llama-3.1-70B': 40600,
            'Mistral-Nemo-Instruct-2407': 86000,
            'Phi-3-mini-4k-instruct': 100000,
            'Phi-3-small-8k-instruct': 1900,
            'Phi-3-medium-4k-instruct': 100000,
            'Qwen2-7B-Instruct': 81900,
            'Qwen2-72B-Instruct': 20000,  # JUST ASSUMED THE VALUE, IT WAS NOT TESTED, I tried 33K and it got stuck
        },
        4: {
            'AceGPT-7B': 100000,
            'AceGPT-13B': 100000,
            'c4ai-command-r-v01': 34700,
            'gemma-2-9b-it': 49300,
            'gemma-2-27b-it': 43400,
            'jais-13b': 100000,
            'jais-30b-v1': 68200,
            'Meta-Llama-3-8B': 97000,
            'Llama-2-7b-chat-hf': 97000,
            'Meta-Llama-3-70B': 54200,
            'Meta-Llama-3.1-8B': 97000,
            'Meta-Llama-3.1-70B': 54200,
            'Mistral-Nemo-Instruct-2407': 89700,
            'Phi-3-mini-4k-instruct': 100000,
            'Phi-3-small-8k-instruct': 0,
            'Phi-3-medium-4k-instruct': 100000,
            'Qwen2-7B-Instruct': 84100,
            'Qwen2-72B-Instruct': 45000
        }
    }
}


def get_max_tokens_per_batch(llm_loader: LLMLoader):
    gpu_name = torch.cuda.get_device_name(0)
    max_tokens_dict = _MAX_TOKENS_PER_BATCH_DICT
    if gpu_name not in max_tokens_dict:
        raise Exception('Unknown GPU')
    max_tokens_dict = max_tokens_dict[gpu_name]
    gpus_count = torch.cuda.device_count()
    max_tokens_dict = max_tokens_dict[gpus_count]
    for model_name, max_tokens_count in max_tokens_dict.items():
        if model_name in llm_loader.name:
            return int(0.3 * max_tokens_count)
    raise Exception(f'Unknown model {llm_loader.name}')
