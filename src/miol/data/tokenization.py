"""
Tokenization utilities for token count prediction.
"""


def drop_duplicate_bos(text, tokenizer):
    """
    Remove a leading BOS from a chat-template string when the tokenizer adds one itself.

    Chat templates such as Llama-3.1's and Mistral-v0.3's start with the BOS
    token, and tokenizing that string with add_special_tokens=True adds a second
    one. Templates without a leading BOS are returned unchanged.
    """
    bos = tokenizer.bos_token
    if not bos or not text.startswith(bos):
        return text
    adds_bos = tokenizer("", add_special_tokens=True)["input_ids"][:1] == [tokenizer.bos_token_id]
    return text[len(bos):] if adds_bos else text


def tokenize_function(examples, tokenizer, max_length=512, truncation=True):
    """
    Tokenize only the prompt (no completion).
    The model will generate its own completion during training.

    Args:
        examples: Dict with 'prompt' and 'completion' keys
        tokenizer: The tokenizer
        max_length: Length prompts are left-padded to (and truncated at, if truncation)
        truncation: Truncate prompts longer than max_length. Hidden-state
            extraction passes False so prompts reach the model unmodified.

    Returns:
        Dict with tokenized prompts only
    """
    batch_size = len(examples["prompt"])

    input_ids_list = []
    attention_mask_list = []

    for i in range(batch_size):
        prompt = drop_duplicate_bos(examples["prompt"][i], tokenizer)

        # Tokenize only the prompt
        prompt_tokens = tokenizer(
            prompt,
            add_special_tokens=True,  # Add BOS if needed
            truncation=truncation,
            max_length=max_length if truncation else None,
            padding=False,
        )

        prompt_ids = prompt_tokens["input_ids"]
        prompt_length = len(prompt_ids)

        # Pad to max_length with LEFT padding (required for decoder-only models)
        # Left padding ensures all prompts "end" at the same position for batch generation
        padding_length = max_length - prompt_length
        if padding_length > 0:
            input_ids = [tokenizer.pad_token_id] * padding_length + prompt_ids
            attention_mask = [0] * padding_length + [1] * prompt_length
        else:
            input_ids = prompt_ids
            attention_mask = [1] * prompt_length

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
    }
