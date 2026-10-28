"""
Unit tests for prompt tokenization: one BOS per prompt, no truncation at extraction.
"""

from miol.data.tokenization import drop_duplicate_bos, tokenize_function


class StubTokenizer:
    """Character-level tokenizer with a configurable automatic BOS."""

    bos_token = "<s>"
    bos_token_id = 1
    pad_token_id = 0

    def __init__(self, adds_bos=True, bos_token="<s>"):
        self.adds_bos = adds_bos
        self.bos_token = bos_token

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=False):
        ids = []
        if self.bos_token and text.startswith(self.bos_token):
            ids.append(self.bos_token_id)
            text = text[len(self.bos_token):]
        ids += [ord(c) for c in text]
        if add_special_tokens and self.adds_bos:
            ids = [self.bos_token_id] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _prompt_ids(tokenized):
    ids, mask = tokenized["input_ids"][0], tokenized["attention_mask"][0]
    return [i for i, m in zip(ids, mask) if m]


class TestDropDuplicateBos:
    def test_template_bos_removed_when_tokenizer_adds_bos(self):
        # Llama-3.1 / Mistral-v0.3: template starts with BOS and the tokenizer adds one.
        assert drop_duplicate_bos("<s>hello", StubTokenizer(adds_bos=True)) == "hello"

    def test_template_bos_kept_when_tokenizer_does_not_add_bos(self):
        assert drop_duplicate_bos("<s>hello", StubTokenizer(adds_bos=False)) == "<s>hello"

    def test_no_bos_token(self):
        # Qwen3: no BOS token at all.
        assert drop_duplicate_bos("hello", StubTokenizer(bos_token=None)) == "hello"

    def test_tokenize_function_yields_single_bos(self):
        tokenized = tokenize_function({"prompt": ["<s>hi"]}, StubTokenizer(adds_bos=True))
        assert _prompt_ids(tokenized) == [1, ord("h"), ord("i")]


class TestTruncation:
    def test_long_prompt_truncated_by_default(self):
        tokenized = tokenize_function({"prompt": ["x" * 600]}, StubTokenizer(), max_length=512)
        assert len(_prompt_ids(tokenized)) == 512

    def test_long_prompt_kept_whole_without_truncation(self):
        tokenized = tokenize_function({"prompt": ["x" * 600]}, StubTokenizer(), max_length=512, truncation=False)
        assert len(_prompt_ids(tokenized)) == 601  # BOS + 600 characters
