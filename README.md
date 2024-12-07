# SmolToken

SmolToken is a fast Rust library for tokenizing text using the Byte Pair Encoding (BPE) algorithm. Inspired by OpenAI's [`tiktoken`](https://github.com/openai/tiktoken), SmolToken is designed to fill a critical gap by enabling BPE training from scratch while maintaining high performance for encoding and decoding tasks.

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

You can add SmolToken to your Rust project via [crates.io](https://crates.io/crates/smoltoken):

```bash
cargo add smoltoken
```

## Example Usage

Here’s a quick example of how to use SmolToken in your Rust project:

```rust
use std::collections::HashSet;
//
use fancy_regex::Regex;
use smoltoken::BytePairTokenizer;
//
// Define a simple pattern and some training data.
let pattern = Regex::new(r"\w+|\S").unwrap();
let data = "hello hello world";
//
// Special tokens to be handled explicitly.
let special_tokens: HashSet<&str> = HashSet::from(["<unk>", "<pad>"]);
//
// Train a BPE tokenizer with a vocabulary size of 300.
let tokenizer = BytePairTokenizer::train(data, r"\w+|\S", 300, special_tokens.clone());
//
// Encode text into token ranks.
let encoded = tokenizer.encode("hello <unk> world", special_tokens.clone());
println!("Encoded: {:?}", encoded);
//
// Decode token ranks back into text.
let decoded = tokenizer.decode_ordinary(&encoded).unwrap();
println!("Decoded: {}", decoded);
```

## Roadmap

- [x] **Concurrency**: Add multi-threading support using `rayon` for faster training, encoding, and decoding.
- [ ] **Python Bindings**: Integrate with Python using `PyO3` to make it accessible for Python developers.
- [ ] **Further Optimizations**: Push for performance on par with HuggingFace's tokenizer.

## Contributing

We very much welcome contributions to make Smoltoken fast, robust and efficient. Make a fork, create a feature branch if needed and sumbit your pull request. Since, the library itself is in its early release stage, I also expect to get community feedback to improve on. Just raise an issue here and we will fix them promptly.

## License

SmolToken is open source and licensed under the [MIT License](LICENSE).

## Acknowledgements

Special thanks to OpenAI's [`tiktoken`](https://github.com/openai/tiktoken) for inspiration and foundational ideas.
