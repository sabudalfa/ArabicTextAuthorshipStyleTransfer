
def generate_prompt_batches(
    prompts,
    tokenizer,
    max_samples_per_batch=2**32,
    max_tokens_per_batch=2**32,
    max_length=None,
    max_new_tokens_list=None,
):
    if max_new_tokens_list:
        if len(max_new_tokens_list) != len(prompts):
            raise ValueError()
        tokens_count = [
            max_new_tokens + len(tokens)
            for max_new_tokens, tokens in zip(max_new_tokens_list, tokenizer(prompts).input_ids)
        ]
    elif max_length:
        tokens_count = len(prompts) * [max_length]
    else:
        raise ValueError()
    
    samples = list(zip(range(len(prompts)), prompts, tokens_count))
    samples = list(sorted(
        samples,
        key=lambda x: x[2],
        reverse=True,
    ))

    batches = [[samples[0]]]
    for sample in samples[1:]:
        last_batch = batches[-1]
        if ((len(last_batch) + 1) * last_batch[0][2]) <= max_tokens_per_batch and (len(last_batch) < max_samples_per_batch):
            last_batch.append(sample)
        else:
            batches.append([sample])
    indices = [
        idx
        for batch in batches
        for idx, _, _ in batch
    ]
    prompts = [
        [
            prompt
            for _, prompt, _ in batch
        ]
        for batch in batches
    ]
    def sort_back(items):
        return [
            item
            for _, item in sorted(zip(indices, items), key=lambda x: x[0])
        ]
    return prompts, sort_back
