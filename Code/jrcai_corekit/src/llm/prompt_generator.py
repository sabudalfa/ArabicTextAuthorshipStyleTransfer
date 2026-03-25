import uuid
from typing import *

def _validate_messages(messages):
    if not isinstance(messages, List):
        raise ValueError('messages should be a list of dicts')
    if len(messages) == 0:
        raise ValueError('empty messages list')
    for message in messages:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError(f'{message} must be a dict that contains "role" and "content"')
        if message['role'] not in ['system', 'user', 'assistant']:
            raise ValueError('message role should be one of {}')
        if not isinstance(message['content'], str):
            raise ValueError(f'message content must be a string')


def generate_chat_llm_prompt(
        tokenizer,
        messages: List[Dict[str, str]],
        omit_start_tags: bool = False,
        omit_end_tags: bool = False,
) -> str:
    _validate_messages(messages)
    start_marker = f'{uuid.uuid4()}'.replace('-', '')
    end_marker = f'{uuid.uuid4()}'.replace('-', '')
    if len(messages) == 1:
        messages = [
            {
                'role': messages[0]['role'],
                'content': f"{start_marker}{messages[0]['content']}{end_marker}",
            },
        ]
    else:
        messages = [
            {
                'role': messages[0]['role'],
                'content': f"{start_marker}{messages[0]['content']}",
            },
            *messages[1:-1],
            {
                'role': messages[-1]['role'],
                'content': f"{messages[-1]['content']}{end_marker}",
            },
        ]
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert start_marker in prompt and end_marker in prompt

    if omit_start_tags:
        prompt = prompt[prompt.index(start_marker)+len(start_marker):]
    else:
        prompt = prompt.replace(start_marker, '')

    if omit_end_tags:
        prompt = prompt[:prompt.index(end_marker)]
    else:
        prompt = prompt.replace(end_marker, '')

    return prompt
