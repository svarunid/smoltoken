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

## Features
- **Concurrency**: Multi-threading support with rayon for accelerated training, encoding, and decoding processes.
- **Python Bindings**: Seamless integration with Python via PyO3, enabling accessibility for Python developers.
- **Serialization**: Support for saving and loading trained tokenizer vocabulary through serialization.

## Contributing

We very much welcome contributions to make Smoltoken fast, robust and efficient. Make a fork, create a feature branch if needed and sumbit your pull request. Since, the library itself is in its early release stage, I also expect to get community feedback to improve on. Just raise an issue here and we will fix them promptly.

## License

SmolToken is open source and licensed under the [MIT License](LICENSE).

## Acknowledgements

Special thanks to OpenAI's [`tiktoken`](https://github.com/openai/tiktoken) for inspiration and foundational ideas.
