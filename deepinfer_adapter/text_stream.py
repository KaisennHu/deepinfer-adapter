# SPDX-License-Identifier: Apache-2.0
"""Incremental text decoding for DeepInfer's fast tokenizer."""
from tokenizers.decoders import DecodeStream


class TokenTextStream:
    """Decode each token once, buffering incomplete UTF-8 between frames.

    A final full decode flushes any incomplete byte sequence at EOS. This is
    one linear pass per request, instead of re-decoding the prefix every step.
    DeepInfer's tokenizer must have text cleanup disabled: cleanup can rewrite
    previously emitted text and cannot be applied to an append-only stream.
    """

    def __init__(self, tokenizer):
        if tokenizer.clean_up_tokenization_spaces:
            raise ValueError("incremental decoding requires clean_up_tokenization_spaces=False")
        self._tokenizer = tokenizer
        self._stream = DecodeStream(skip_special_tokens=False)
        self._ids = []
        self._emitted_length = 0

    def push(self, token_id: int, *, finished: bool = False) -> str:
        self._ids.append(token_id)
        delta = self._stream.step(self._tokenizer.backend_tokenizer, token_id) or ""
        self._emitted_length += len(delta)
        if finished:
            text = self._tokenizer.decode(self._ids)
            delta += text[self._emitted_length:]
            self._emitted_length = len(text)
        return delta
