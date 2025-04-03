import base64
import collections
from concurrent.futures import ThreadPoolExecutor
import functools
from pathlib import Path
from typing import List, Sequence, Set, Union

from smoltoken import _smoltoken


class NotTrainedError(Exception):
    """Custom exception to indicate that the class instance must be trained before other operations."""

    def __init__(
        self, message="You must call the 'train' method before calling this method."
    ):
        super().__init__(message)


class BytePairTokenizer:
    """A tokenizer that uses byte pair encoding algorithm to encode/decode text."""

    def __init__(self, name, *, pattern: str, special_tokens: Set[str], n_vocab: int):
        self.name = name
        self._pattern = pattern
        self._n_vocab = n_vocab
        self._special_tokens = special_tokens

    def train(self, *, data: Union[str, Sequence[str]] = None, path: str = None):
        """
        Trains a Byte Pair Encoding tokenizer on the given data with the specified vocabulary size.

        Args:
            data:
                The training data to use. Can be a single string, a sequence of strings,
                or an iterator yielding strings. Either this or `path` must be provided.
            path:
                Path to a directory containing the files

        Raises:
            AssertionError: If neither `data` nor `path` is provided.
            ValueError: If any error arises when compiling the regex pattern.
        """
        assert data or path, "Either data or path must be provided."
        if data:
            if isinstance(data, str):
                self._core = _smoltoken.BytePairTokenizer.from_text(
                    data, self._pattern, self._n_vocab, self._special_tokens
                )
            elif isinstance(data, collections.abc.Sequence):
                self._core = _smoltoken.BytePairTokenizer.from_seq(
                    data, self._pattern, self._n_vocab, self._special_tokens
                )
        elif path:
            self._core = _smoltoken.BytePairTokenizer.from_dir(
                path, self._pattern, self._n_vocab, self._special_tokens
            )

    def encode_ordinary(self, text: str):
        """Encodes a given text into token ranks using ordinary (non-special) tokens."""
        if not hasattr(self, "_core"):
            raise NotTrainedError()
        return self._core.encode_ordinary(text)

    def encode(self, text: str, allowed_special: Set[str]):
        """Encodes a given text into tokens, allowing specified special tokens."""
        if not hasattr(self, "_core"):
            raise NotTrainedError()
        return self._core.encode(text, allowed_special)

    def encode_ordinary_batch(self, text: List[str], num_threads: int = 8):
        """Encodes a lsit of text into token ranks using ordinary (non-special) tokens."""
        if not hasattr(self, "_core"):
            raise NotTrainedError()
        encoder = functools.partial(self.encode_ordinary)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def encode_batch(
        self, text: List[str], allowed_special: Set[str], num_threads: int = 8
    ):
        """Encodes a list of text into tokens, allowing specified special tokens."""
        if not hasattr(self, "_core"):
            raise NotTrainedError()
        encoder = functools.partial(self.encode, allowed_special=allowed_special)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))

    def decode(self, tokens: Sequence[int], errors: str = "replace"):
        """Decodes a list of tokens into a string."""
        if not hasattr(self, "_core"):
            raise NotTrainedError()
        return self._core.decode(tokens).decode("utf-8", errors=errors)

    def decode_batch(
        self,
        batch: Sequence[Sequence[int]],
        errors: str = "replace",
        num_threads: int = 8,
    ):
        """Decodes a batch (list of lists of tokens) into a list of strings."""
        if not hasattr(self, "_core"):
            raise NotTrainedError()
        decoder = functools.partial(self.decode, errors=errors)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(decoder, batch))

    def save(self, dir: str):
        """Save the vocabulary in a `.smtkn` file to a provided directory."""
        if not hasattr(self, "_core"):
            raise NotTrainedError()
        self._core.save(f"{self.name}.smtkn", dir)

    @classmethod
    def load(cls, path: str, *, pattern: str, special_tokens: Set[str]):
        """Loads the tokenizer vocabulary from a `.smtkn` file."""
        path: Path = Path(path)
        if path.is_dir():
            raise ValueError(
                f"{path} is a directory. Please provide a path to a file with .smtkn extension."
            )

        encoder = dict()
        with open(path, "rb") as f:
            for line in f:
                token, rank = line.split()
                token, rank = base64.b64decode(token), int(rank)
                encoder[token] = rank

        tok = cls(
            path.name,
            pattern=pattern,
            special_tokens=special_tokens,
            n_vocab=len(encoder),
        )
        tok._core = _smoltoken.BytePairTokenizer.load(encoder, pattern, special_tokens)
        return tok
