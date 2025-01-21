from concurrent.futures import ThreadPoolExecutor
import functools
from typing import List, Sequence, Set

from python import _smoltoken


class BytePairTokenizer:
    """A tokenizer that uses byte pair encoding algorithm to encode/decode text."""

    def __init__(self, name, *, pattern: str, special_tokens: Set[str], n_vocab: int):
        self.name = name
        self._pattern = pattern
        self._n_vocab = n_vocab
        self._special_tokens = special_tokens

    def train(self, data: str):
        """Trains a Byte Pair Encoding tokenizer on the given data with the specified vocabulary size."""
        self._core = _smoltoken.BytePairTokenizer(
            data, self._pattern, self._n_vocab, self._special_tokens
        )

    def encode_ordinary(self, text: str):
        """Encodes a given text into token ranks using ordinary (non-special) tokens."""
        return self._core.encode(text)

    def encode(self, text: str, allowed_special: Set[str]):
        """Encodes a given text into tokens, allowing specified special tokens."""
        return self._core.encode(text, allowed_special)

    def encode_ordinary_batch(self, text: List[str], num_threads: int = 8):
        """Encodes a lsit of text into token ranks using ordinary (non-special) tokens."""
        encoder = functools.partial(self.encode_ordinary)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def encode_batch(
        self, text: List[str], allowed_special: Set[str], num_threads: int = 8
    ):
        """Encodes a list of text into tokens, allowing specified special tokens."""
        encoder = functools.partial(self.encode, allowed_special=allowed_special)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def decode(self, tokens: Sequence[int], errors: str = "replace"):
        """Decodes a list of tokens into a string."""
        return self._core.decode(tokens).decode("utf-8", errors=errors)

    def decode_batch(
        self,
        batch: Sequence[Sequence[int]],
        errors: str = "replace",
        num_threads: int = 8,
    ):
        """Decodes a batch (list of lists of tokens) into a list of strings."""
        decoder = functools.partial(self.decode, errors=errors)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(decoder, batch))
