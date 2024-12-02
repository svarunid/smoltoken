//! SmolToken: A fast library for Byte Pair Encoding (BPE) tokenization.
//!
//! SmolToken is a fast lightweight tokenizer library designed to tokenize text using
//! Byte Pair Encoding (BPE), a widely-used subword tokenization algorithm. Inspired by OpenAI's
//! [`tiktoken`], SmolToken aims to provide a robust solution for encoding and decoding text,
//! with additional flexibility to train tokenizers from scratch on your own data.
//!
//! # Example
//!
//! ```rust
//! use std::collections::HashSet;
//!
//! use fancy_regex::Regex;
//! use smoltoken::BytePairTokenizer;
//!
//! // Define a simple pattern and some training data.
//! let pattern = Regex::new(r"\w+|\S").unwrap();
//! let data = "hello hello world";
//!
//! // Special tokens to be handled explicitly.
//! let special_tokens: HashSet<&str> = HashSet::from(["<unk>", "<pad>"]);
//!
//! // Train a BPE tokenizer with a vocabulary size of 300.
//! let tokenizer = BytePairTokenizer::train(data, r"\w+|\S", 300, special_tokens.clone());
//!
//! // Encode text into token ranks.
//! let encoded = tokenizer.encode("hello <unk> world", special_tokens.clone());
//! println!("Encoded: {:?}", encoded);
//!
//! // Decode token ranks back into text.
//! let decoded = tokenizer.decode_ordinary(&encoded).unwrap();
//! println!("Decoded: {}", decoded);
//! ```
//!
//! [`tiktoken`]: https://github.com/openai/tiktoken
#![feature(test)]

use std::collections::HashSet;
use std::string::FromUtf8Error;

// use rayon::prelude::*;
use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;

/// Alias for an unsigned 8-bit integer representing a byte.
type Byte = u8;
/// Alias for an unsigned 32-bit integer representing token IDs.
type Rank = u32;

/// Represents errors that may occur during decoding in the BPE algorithm.

#[derive(Debug, Clone)]
pub enum DecodeError {
    /// Error indicating an invalid (out-of-vocabulary) token.
    TokenError(Rank),
    /// Error indicating failure to convert bytes into a valid UTF-8 string.
    Utf8Error(FromUtf8Error),
}

impl std::fmt::Display for DecodeError {
    /// Formats the DecodeError for user-friendly output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenError(token) => write!(f, "Invalid token for decoding: {}", token),
            Self::Utf8Error(err) => std::fmt::Display::fmt(err, f),
        }
    }
}

impl From<FromUtf8Error> for DecodeError {
    /// Converts a FromUtf8Error into a DecodeError::Utf8Error.
    fn from(err: FromUtf8Error) -> Self {
        DecodeError::Utf8Error(err)
    }
}

/// A tokenizer that uses byte pair encoding algorithm to encode/decode text.
pub struct BytePairTokenizer {
    pattern: Regex,
    encoder: HashMap<Vec<Byte>, Rank>,
    decoder: HashMap<Rank, Vec<Byte>>,
    special_pattern: Regex,
    special_encoder: HashMap<Vec<Byte>, Rank>,
    special_decoder: HashMap<Rank, Vec<Byte>>,
}

impl BytePairTokenizer {
    fn encode_native(&self, bytes: &[Byte]) -> Vec<Rank> {
        if bytes.len() == 1 {
            return vec![self.encoder[bytes]];
        }

        // This is a vector of (start, rank).
        // The rank is of the pair starting at position start.
        let mut parts = Vec::with_capacity(bytes.len() + 1);

        // Note that we hash bytes when indexing into `ranks`, not token pairs. As long as we train BPE
        // the way we currently do, this is equivalent. An easy way to break this would be to decouple
        // merge priority from token index or to prevent specific token merges.
        let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
        for i in 0..bytes.len() - 1 {
            let rank = *self.encoder.get(&bytes[i..=i + 1]).unwrap_or(&Rank::MAX);
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
            parts.push((i, rank));
        }
        parts.push((bytes.len() - 1, Rank::MAX));
        parts.push((bytes.len(), Rank::MAX));

        let get_rank = {
            #[inline(always)]
            |parts: &Vec<(usize, Rank)>, i: usize| {
                if (i + 3) < parts.len() {
                    // Similar to `piece[i..=i+1]` above. The +2 is because we haven't yet deleted
                    // parts[i + 1], see comment in the main loop.
                    *self
                        .encoder
                        .get(&bytes[parts[i].0..=parts[i + 2].0])
                        .unwrap_or(&Rank::MAX)
                } else {
                    Rank::MAX
                }
            }
        };

        // If you have n parts and m merges, this does O(mn) work.
        // We could do something with a heap and do O(m log n) work.
        // n is often very small so considerations like cache-locality outweigh the algorithmic
        // complexity downsides of the `parts` vector.
        while min_rank.0 != Rank::MAX {
            let i = min_rank.1;
            // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
            // `parts.remove(i + 1)` will thrash the cache.
            if i > 0 {
                parts[i - 1].1 = get_rank(&parts, i - 1);
            }
            parts[i].1 = get_rank(&parts, i);
            parts.remove(i + 1);

            min_rank = (Rank::MAX, usize::MAX);
            for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
                if rank < min_rank.0 {
                    min_rank = (rank, i);
                }
            }
        }

        parts
            .windows(2)
            .map(|part| self.encoder[&bytes[part[0].0..part[1].0]])
            .collect()
    }

    fn decode_native(&self, tokens: &[Rank]) -> Result<Vec<Byte>, DecodeError> {
        let mut bytes: Vec<Byte> = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            let token_bytes = match self.decoder.get(&token) {
                Some(bytes) => bytes,
                None => self
                    .special_decoder
                    .get(&token)
                    .ok_or(DecodeError::TokenError(token))?,
            };
            bytes.extend(token_bytes);
        }
        Ok(bytes)
    }
}

impl BytePairTokenizer {
    /// Creates a new `BytePairTokenizer` with the specified pattern, encoder, decoder, and special tokens.
    pub fn new(
        pattern: Regex,
        encoder: HashMap<Vec<Byte>, Rank>,
        decoder: HashMap<Rank, Vec<Byte>>,
        special_tokens: HashSet<&str>,
    ) -> Self {
        let mut rank = encoder.len() as Rank;
        let mut special_encoder = HashMap::default();
        let mut special_decoder = HashMap::default();
        for &special in &special_tokens {
            let bytes = special.as_bytes().to_vec();
            special_encoder.insert(bytes.clone(), rank);
            special_decoder.insert(rank, bytes);

            rank += 1;
        }

        let special_pattern = Regex::new(
            &special_tokens
                .iter()
                .map(|&spl| fancy_regex::escape(spl))
                .collect::<Vec<_>>()
                .join("|"),
        )
        .unwrap();

        Self {
            pattern,
            encoder,
            decoder,
            special_pattern,
            special_encoder,
            special_decoder,
        }
    }

    /// Encodes a given text into token ranks using ordinary (non-special) tokens.
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let mut ranks = vec![];
        for part in self.pattern.find_iter(text) {
            let bytes = part.unwrap().as_str().as_bytes();
            match self.encoder.get(bytes) {
                Some(rank) => ranks.push(*rank),
                None => ranks.extend(self.encode_native(bytes)),
            }
        }
        ranks
    }

    /// Encodes a given text into token ranks, allowing specified special tokens.
    pub fn encode(&self, text: &str, allowed_special: HashSet<&str>) -> Vec<Rank> {
        let mut ranks = vec![];

        let mut start = 0;
        loop {
            // Find the next occurrence of a special token.
            let special = self.special_pattern.find_from_pos(text, start).unwrap();
            match special {
                Some(special) => {
                    // Skip if the special token is not allowed.
                    if !allowed_special.contains(&special.as_str()) {
                        start = special.start() + 1;
                        continue;
                    }
                    ranks.extend(self.encode_ordinary(&text[start..special.start()]));
                    ranks.push(self.special_encoder[special.as_str().as_bytes()]);
                    start += special.end();
                }
                None => {
                    // Encode remaining text if no special tokens are left.
                    if start != text.len() {
                        ranks.extend(self.encode_ordinary(&text[start..text.len()]));
                    }
                    break;
                }
            }
        }
        ranks
    }

    /// Decodes a sequence of token ranks into a string.
    pub fn decode_ordinary(&self, tokens: &[Rank]) -> Result<String, DecodeError> {
        // Decoupling the implementation earlier to avoid code repetition when
        // implementing variants of `decode` method later.
        Ok(String::from_utf8(self.decode_native(tokens)?)?)
    }

    /// Trains a Byte Pair Encoding tokenizer on the given data with the specified vocabulary size.
    pub fn train(
        data: &str,
        pattern: &str,
        vocab_size: Rank,
        special_tokens: HashSet<&str>,
    ) -> Self {
        // Ensure that the vocabulary size is at least 256 to cover all possible byte values.
        assert!(
            vocab_size >= 256,
            "Vocabulary size should atleast be 256 to cover all individual bytes"
        );

        // Initialize the encoder and decoder with all possible single-byte tokens.
        // The encoder maps sequences of bytes (Vec<Byte>) to their corresponding token IDs (Rank).
        let mut encoder: HashMap<Vec<Byte>, Rank> =
            (0..256).map(|b| (vec![b as Byte], b as Rank)).collect();

        // The decoder maps token IDs back to their byte sequences.
        let mut decoder: HashMap<Rank, Vec<Byte>> =
            (0..256).map(|b| (b as Rank, vec![b as Byte])).collect();

        // Use the provided regex pattern to split the input data into words.
        // Split the word into individual bytes and convert them to `Vec<Ranks>`.
        let pattern = Regex::new(pattern).unwrap();
        let mut parts: Vec<Vec<Rank>> = pattern
            .find_iter(data)
            .map(|part| {
                part.unwrap()
                    .as_str()
                    .as_bytes()
                    .iter()
                    .map(|&b| b as Rank)
                    .collect()
            })
            .collect();

        let mut stats: HashMap<(Rank, Rank), isize> = HashMap::default();
        for part in &parts {
            for pair in part.windows(2) {
                let pair = (pair[0], pair[1]);
                stats
                    .entry(pair)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        // Okay, now what happens here might seem a bit intimidating. So, let be break it
        // for ya. Whenever a merge operation is performed, a couple of things happen:
        // - The frequent pair is merged to form a new token and is removed from `stats`.
        // - The occurrences of the frequent pair in the `parts` is replaced by the new token.
        // - Frequencies of new pairs formed with the new token are added to `stats`.
        // - Frequencies of pairs that contained the ranks in the merged pairs are decremented
        //   from `stats`.
        // And that's exactly what happens inside the nested loops.
        while decoder.len() < vocab_size as usize {
            // Filters `stats` for entries with frqeuncy greater than `0`.
            stats.retain(|_, v| *v > 0);
            match stats.iter().max_by_key(|&(_, count)| count) {
                None => break,
                Some((&most_common_pair, _)) => {
                    let rank = decoder.len() as Rank; // Newly minted token.
                    stats.remove(&most_common_pair);

                    // Retrieve the byte sequences corresponding to the tokens in the most frequent pair.
                    // These byte sequences are obtained from the `decoder` mapping.
                    let mut bytes = decoder.get(&most_common_pair.0).unwrap().clone();
                    bytes.extend(decoder.get(&most_common_pair.1).unwrap().clone());

                    // Add the new symbol to the `encoder` and `decoder` mappings.
                    // This updates the vocabulary with the new merged token.
                    encoder.insert(bytes.clone(), rank);
                    decoder.insert(rank, bytes);

                    for part in &mut parts {
                        let mut i = 0;
                        while i + 1 < part.len() {
                            if part[i] == most_common_pair.0 && part[i + 1] == most_common_pair.1 {
                                // The pair getting merged is `(part[i], part[i+1])`
                                if i > 0 {
                                    // Decrement the frequency of pair `(part[i-1], part[i])`
                                    stats
                                        .entry((part[i - 1], part[i]))
                                        .and_modify(|count| *count -= 1);

                                    // Increment the frequency of pair `(part[i-1], rank)`. Also handles `(rank, rank)`
                                    // when a previous pair is merged.
                                    stats
                                        .entry((part[i - 1], rank))
                                        .and_modify(|count| *count += 1)
                                        .or_insert(1);
                                }

                                if i + 2 < part.len() {
                                    // Decrement the frequency of pair `(part[i+1], part[i+2])`
                                    stats
                                        .entry((part[i + 1], part[i + 2]))
                                        .and_modify(|count| *count -= 1);

                                    if i + 3 < part.len()
                                        && !(part[i + 2] == most_common_pair.0
                                            && part[i + 3] == most_common_pair.1)
                                    {
                                        // Increment the frequency of pair `(rank, part[i + 2])` only when the next pair is
                                        // not the `most_common_pair`
                                        stats
                                            .entry((rank, part[i + 2]))
                                            .and_modify(|count| *count += 1)
                                            .or_insert(1);
                                    }
                                }

                                part[i] = rank;
                                part.remove(i + 1);
                            }
                            i += 1;
                        }
                    }
                }
            }
        }

        Self::new(pattern, encoder, decoder, special_tokens)
    }
}

extern crate test;
#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> BytePairTokenizer {
        let pattern = Regex::new(r"\S+|\s+\S+").unwrap();
        let encoder: HashMap<Vec<Byte>, Rank> = HashMap::from_iter(vec![
            (b"He".to_vec(), 0),
            (b"ll".to_vec(), 1),
            (b"o".to_vec(), 2),
            (b",".to_vec(), 3),
            (b" w".to_vec(), 4),
            (b"or".to_vec(), 5),
            (b"ld".to_vec(), 6),
            (b"!".to_vec(), 7),
            (b"<|".to_vec(), 8),
            (b"en".to_vec(), 9),
            (b"do".to_vec(), 10),
            (b"ft".to_vec(), 11),
            (b"ex".to_vec(), 12),
            (b"t".to_vec(), 13),
            (b"|>".to_vec(), 14),
        ]);

        let decoder: HashMap<Rank, Vec<Byte>> = encoder
            .iter()
            .map(|(bytes, rank)| (*rank, bytes.clone()))
            .collect();
        let special_tokens: HashSet<&str> = HashSet::from(["<|endoftext|>"]);

        BytePairTokenizer::new(pattern, encoder, decoder, special_tokens)
    }

    #[test]
    fn encode_without_special_tokens() {
        let tok = setup();

        assert_eq!(
            tok.encode_ordinary("Hello, world!"),
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );

        assert_eq!(
            tok.encode_ordinary("Hello<|endoftext|>, world!"),
            &[0, 1, 2, 8, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn encode_with_allowed_special_tokens() {
        let tok = setup();
        let allowed_special = HashSet::from(["<|endoftext|>"]);

        assert_eq!(
            tok.encode("Hello<|endoftext|>, world!", allowed_special),
            &[0, 1, 2, 15, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn decode_tokens() {
        let tok = setup();

        assert_eq!(
            tok.decode_ordinary(&[0, 1, 2, 15, 3, 4, 5, 6, 7]).unwrap(),
            "Hello<|endoftext|>, world!"
        );

        assert_eq!(
            tok.decode_ordinary(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
            "Hello, world!"
        );
    }

    #[test]
    fn fail_to_decode() {
        let tok = setup();

        matches!(
            tok.decode_ordinary(&[200, 300]).unwrap_err(),
            DecodeError::TokenError(_)
        );
    }

    #[test]
    fn train_bpe() {
        let data = "abababcd";
        let pattern = r"\S+|\s+\S+";
        let vocab_size = 260;
        let special_tokens: HashSet<&str> = HashSet::from(["<|endoftext|>"]);
        let tokenizer = BytePairTokenizer::train(data, pattern, vocab_size, special_tokens);

        let encoder = tokenizer.encoder;
        let decoder = tokenizer.decoder;
        assert_eq!(encoder.len(), vocab_size as usize);
        assert_eq!(decoder.len(), vocab_size as usize);

        assert!(encoder.contains_key(&vec![b'a', b'b']));
        assert!(encoder.contains_key(&vec![b'a', b'b', b'a', b'b']));

        assert_eq!(decoder[&256], vec![b'a', b'b']);
        assert_eq!(decoder[&257], vec![b'a', b'b', b'a', b'b']);

        assert!(tokenizer
            .special_encoder
            .contains_key(&b"<|endoftext|>"[..]));

        assert!(tokenizer.special_pattern.is_match("<|endoftext|>").unwrap());
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use test::Bencher;

    #[bench]
    fn train(b: &mut Bencher) {
        let vocab_size = 500;
        let data = std::fs::read_to_string("../sample/code.txt").unwrap();
        let pattern =
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

        b.iter(|| {
            BytePairTokenizer::train(&data, pattern, vocab_size, HashSet::from(["<|endoftext|>"]))
        })
    }
}
