# SmolToken

SmolToken is a fast library for tokenizing text using the Byte Pair Encoding (BPE) algorithm. Inspired by OpenAI's [`tiktoken`](https://github.com/openai/tiktoken), SmolToken is designed to fill a critical gap by enabling BPE training from scratch while maintaining high performance for encoding and decoding tasks.

Unlike `tiktoken`, SmolToken supports training tokenizers on custom data. Up to **~4x faster** than the port of unoptimized educational implementation [`_educational.py`](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py) in rust.

## Benchmark Results

SmolToken is already faster than baseline educational implementation of BPE training:

| Implementation                 | Runtime (sec) |
| ------------------------------ | ------------- |
| **Unoptimized Implementation** | 36.94385      |
| **SmolToken Optimized**        | 17.63223      |
| **SmolToken (with rayon)**     | 7.489850      |

Tested on:

- Vocabulary size: **500**
- Dataset: **Tiny Stories (~18 MB)**

## Installation

Add smoltoken to your Rust project via [crates.io](https://crates.io/):

```bash
cargo add smoltoken
```

Or add smoltoken to your Python project via [PyPI](https://pypi.org/):

```bash
pip install smoltoken
```

## Roadmap

- [x] **Concurrency**: Add multi-threading support using `rayon` for faster training, encoding, and decoding.
- [x] **Python Bindings**: Integrate with Python using `PyO3` to make it accessible for Python developers.
- [x] **Serialization**: Add serialization support to save/load trained tokenizer vocabulary.

## Contributing

We very much welcome contributions to make Smoltoken fast, robust and efficient. Make a fork, create a feature branch if needed and sumbit your pull request. Since, the library itself is in its early release stage, I also expect to get community feedback to improve on. Just raise an issue here and we will fix them promptly.

## License

SmolToken is open source and licensed under the [MIT License](LICENSE).

## Acknowledgements

Special thanks to OpenAI's [`tiktoken`](https://github.com/openai/tiktoken) for inspiration and foundational ideas.
