from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import tiktoken


class ReportTokenizer:
    """GPT-2 BPE tokenizer with explicit report special tokens."""

    def __init__(self, encoding: str = "gpt2", max_len: int = 256):
        self.encoding_name = encoding
        self.enc = tiktoken.get_encoding(encoding)
        self.max_len = max_len
        self.pad_id = self.enc.n_vocab
        self.bos_id = self.enc.n_vocab + 1
        self.eos_id = self.enc.n_vocab + 2
        self.vocab_size = self.enc.n_vocab + 3

    def encode(self, text: object) -> torch.Tensor:
        text = self._clean_text(text)
        ids = [self.bos_id, *self.enc.encode(text.lower()), self.eos_id]
        if len(ids) < self.max_len:
            ids.extend([self.pad_id] * (self.max_len - len(ids)))
        else:
            ids = ids[: self.max_len - 1] + [self.eos_id]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        decoded_ids: list[int] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_id:
                break
            if token_id in {self.pad_id, self.bos_id}:
                if skip_special_tokens:
                    continue
            if token_id < self.enc.n_vocab:
                decoded_ids.append(token_id)
        return self.enc.decode(decoded_ids).strip()

    @staticmethod
    def _clean_text(text: object) -> str:
        if text is None:
            return "no findings"
        if isinstance(text, float) and math.isnan(text):
            return "no findings"
        text = str(text).strip()
        if not text or text.lower() == "nan":
            return "no findings"
        return text
