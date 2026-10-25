# SPDX-License-Identifier: Apache-2.0
import unittest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from deepinfer_adapter.text_stream import TokenTextStream


class FastTokenizer:
    clean_up_tokenization_spaces = False

    def __init__(self):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(
            ["hello world", "你好🙂", "print('hello')\n", " café naïve"],
            trainers.BpeTrainer(vocab_size=256, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
        )
        self.backend_tokenizer = tokenizer
        self.decode_calls = 0

    def decode(self, ids):
        self.decode_calls += 1
        return self.backend_tokenizer.decode(ids, skip_special_tokens=False)


class TextStreamTests(unittest.TestCase):
    def test_unicode_spaces_code_and_incomplete_bytes(self):
        tokenizer = FastTokenizer()
        for text in ["hello world", "你好🙂", "print('hello')\n", " café naïve", "🙂🙂"]:
            ids = tokenizer.backend_tokenizer.encode(text).ids
            for length in range(1, len(ids) + 1):
                with self.subTest(text=text, length=length):
                    partial = ids[:length]
                    stream = TokenTextStream(tokenizer)
                    output = ""
                    for i, token_id in enumerate(partial):
                        output += stream.push(token_id, finished=i == len(partial) - 1)
                    self.assertEqual(output, tokenizer.decode(partial))

    def test_full_decode_occurs_only_once_at_finish(self):
        tokenizer = FastTokenizer()
        ids = tokenizer.backend_tokenizer.encode("hello world" * 100).ids
        stream = TokenTextStream(tokenizer)
        for token_id in ids[:-1]:
            stream.push(token_id)
        self.assertEqual(tokenizer.decode_calls, 0)
        stream.push(ids[-1], finished=True)
        self.assertEqual(tokenizer.decode_calls, 1)


if __name__ == "__main__":
    unittest.main()
