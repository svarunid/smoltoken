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
//! let name = String::from("simple_tokenizer");
//! let pattern = Regex::new(r"\w+|\S").unwrap();
//! let data = "hello hello world";
//!
//! // Special tokens to be handled explicitly.
//! let special_tokens: HashSet<&str> = HashSet::from(["<unk>", "<pad>"]);
//!
//! // Train a BPE tokenizer with a vocabulary size of 300.
//! let tokenizer = BytePairTokenizer::train(name, data, r"\w+|\S", 300, special_tokens.clone());
//!
//! // Encode text into token ranks.
//! let encoded = tokenizer.encode("hello <unk> world", &special_tokens);
//! println!("Encoded: {:?}", encoded);
//!
//! // Decode token ranks back into text.
//! let decoded = tokenizer.decode(&encoded).unwrap();
//! println!("Decoded: {}", decoded);
//! ```
//!
//! [`tiktoken`]: https://github.com/openai/tiktoken
use std::io::{BufRead, BufReader, BufWriter, Error, Write};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::string::FromUtf8Error;
use std::thread;
use std::{collections::HashSet, fs::File};

use base64::prelude::*;
use fancy_regex::Regex;
use kdam::{tqdm, BarExt};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

/// Alias for an unsigned 8-bit integer representing a byte.
type Byte = u8;
/// Alias for an unsigned 32-bit integer representing token IDs.
type Rank = u32;

const MAX_NUM_THREADS: usize = 128;

#[inline(always)]
fn increment(stats: &mut HashMap<(Rank, Rank), isize>, pair: (Rank, Rank)) {
    stats.entry(pair).and_modify(|c| *c += 1).or_insert(1);
}

#[inline(always)]
fn decrement(stats: &mut HashMap<(Rank, Rank), isize>, pair: (Rank, Rank)) {
    stats.entry(pair).and_modify(|c| *c -= 1).or_insert(-1);
}

/// Represents errors that may occur during decoding in the BPE algorithm.
#[derive(Debug, Clone)]
pub enum DecodeError {
    /// Error indicating an invalid (out-of-vocabulary) token.
    TokenError(Rank),
    /// Error indicating failure to convert bytes into a valid UTF-8 string.
    Utf8Error(FromUtf8Error),
}

pub struct FakeThreadId(NonZeroU64);
fn hash_current_thread() -> usize {
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TokenError(token) => write!(f, "Invalid token for decoding: {}", token),
            Self::Utf8Error(err) => std::fmt::Display::fmt(err, f),
        }
    }
}

impl From<FromUtf8Error> for DecodeError {
    fn from(err: FromUtf8Error) -> Self {
        DecodeError::Utf8Error(err)
    }
}

/// A tokenizer that uses byte pair encoding algorithm to encode/decode text.
pub struct BytePairTokenizer {
    name: String,
    pattern: Vec<Regex>,
    encoder: HashMap<Vec<Byte>, Rank>,
    decoder: HashMap<Rank, Vec<Byte>>,
    special_pattern: Vec<Regex>,
    special_encoder: HashMap<Vec<Byte>, Rank>,
    special_decoder: HashMap<Rank, Vec<Byte>>,
}

impl BytePairTokenizer {
    fn get_tl_regex(&self) -> &Regex {
        &self.pattern[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn get_tl_special_regex(&self) -> &Regex {
        &self.special_pattern[hash_current_thread() % MAX_NUM_THREADS]
    }

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
        name: String,
        pattern: Regex,
        encoder: HashMap<Vec<Byte>, Rank>,
        decoder: HashMap<Rank, Vec<Byte>>,
        special_tokens: HashSet<&str>,
    ) -> Self {
        let mut rank = encoder.len() as Rank;
        let pattern = (0..MAX_NUM_THREADS).map(|_| pattern.clone()).collect();
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
        let special_pattern = (0..MAX_NUM_THREADS)
            .map(|_| special_pattern.clone())
            .collect();

        Self {
            name,
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
        let regex = self.get_tl_regex();
        for part in regex.find_iter(text) {
            let bytes = part.unwrap().as_str().as_bytes();
            match self.encoder.get(bytes) {
                Some(rank) => ranks.push(*rank),
                None => ranks.extend(self.encode_native(bytes)),
            }
        }
        ranks
    }

    /// Encodes an array of text a vector of token ranks using ordinary (non-special) tokens.
    pub fn encode_ordinary_batch(&self, batch: &[&str]) -> Vec<Vec<Rank>> {
        batch
            .par_iter()
            .map(|text| self.encode_ordinary(text))
            .collect()
    }

    /// Encodes a given text into token ranks, allowing specified special tokens.
    pub fn encode(&self, text: &str, allowed_special: &HashSet<&str>) -> Vec<Rank> {
        let mut ranks = vec![];
        let special_regex = self.get_tl_special_regex();

        let mut start = 0;
        loop {
            // Find the next occurrence of a special token.
            let special = special_regex.find_from_pos(text, start).unwrap();
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

    /// Encodes an array of text into a vector of token ranks, allowing specified special tokens.
    pub fn encode_batch(&self, batch: &[&str], allowed_special: &HashSet<&str>) -> Vec<Vec<Rank>> {
        batch
            .par_iter()
            .map(|text| self.encode(text, &allowed_special))
            .collect()
    }

    /// Decodes a sequence of token ranks into a string.
    pub fn decode(&self, tokens: &[Rank]) -> Result<String, DecodeError> {
        // Decoupling the implementation earlier to avoid code repetition when
        // implementing variants of `decode` method later.
        Ok(String::from_utf8(self.decode_native(tokens)?)?)
    }

    /// Decode an array of sequence of token ranks into a vector of strings.
    pub fn decode_batch(&self, batch: &[&[Rank]]) -> Vec<Result<String, DecodeError>> {
        batch.par_iter().map(|tokens| self.decode(tokens)).collect()
    }

    /// Trains a Byte Pair Encoding tokenizer on the given data with the specified vocabulary size.
    pub fn train(
        name: String,
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

        // Use the provided regex pattern to split the input data into words.
        // Split the word into individual bytes and convert them to `Vec<Ranks>`.
        println!("Splitting data into sub-words based on the pattern...");
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
                increment(&mut stats, (pair[0], pair[1]));
            }
        }

        // Initialize the encoder and decoder with all possible single-byte tokens.
        // The encoder maps sequences of bytes (Vec<Byte>) to their corresponding token IDs (Rank).
        println!("Starting to build vocabulary...");
        let mut pb = tqdm!(total = vocab_size as usize);
        let mut encoder: HashMap<Vec<Byte>, Rank> =
            (0..256).map(|b| (vec![b as Byte], b as Rank)).collect();

        // The decoder maps token IDs back to their byte sequences.
        let mut decoder: HashMap<Rank, Vec<Byte>> =
            (0..256).map(|b| (b as Rank, vec![b as Byte])).collect();

        // Okay, now what happens here might seem a bit intimidating. So, let be break it
        // for ya. Whenever a merge operation is performed, a couple of things happen:
        // - The frequent pair is merged to form a new token and is removed from `stats`.
        // - The occurrences of the frequent pair in the `parts` is replaced by the new token.
        // - Frequencies of new pairs formed with the new token are added to `stats`.
        // - Frequencies of pairs that contained the ranks in the merged pairs are decremented
        //   from `stats`.
        // And that's exactly what happens inside the nested loops.
        let _ = pb.update(256);
        while decoder.len() < vocab_size as usize {
            // Ensure that the frequencies are not below 0.
            stats.retain(|_, v| *v > 0);

            let most_common_pair = match stats.par_iter().max_by_key(|&(_, count)| count) {
                None => {
                    println!("Warning: Ran out of pairs before reaching target vocabulary size");
                    println!("Final vocabulary size: {}", decoder.len());
                    break;
                }
                Some((&most_common_pair, _)) => {
                    stats.remove(&most_common_pair);
                    [most_common_pair.0, most_common_pair.1]
                }
            };

            let rank = decoder.len() as Rank; // Newly minted token.

            // Retrieve the byte sequences corresponding to the tokens in the most frequent pair.
            // These byte sequences are obtained from the `decoder` mapping.
            let mut bytes = decoder.get(&most_common_pair[0]).unwrap().clone();
            bytes.extend(decoder.get(&most_common_pair[1]).unwrap());

            // Add the new symbol to the `encoder` and `decoder` mappings.
            // This updates the vocabulary with the new merged token.
            encoder.insert(bytes.clone(), rank);
            decoder.insert(rank, bytes);

            let freqs: Vec<((Rank, Rank), isize)> = parts
                .par_iter_mut()
                .flat_map(|part| {
                    let mut i = 0;
                    let mut stats = HashMap::default();
                    while i + 1 < part.len() {
                        if part[i..i + 2] == most_common_pair {
                            if i > 0 {
                                decrement(&mut stats, (part[i - 1], part[i]));
                                increment(&mut stats, (part[i - 1], rank));
                            }

                            if i + 2 < part.len() {
                                decrement(&mut stats, (part[i + 1], part[i + 2]));

                                if i + 3 < part.len() && part[i + 2..i + 4] != most_common_pair {
                                    increment(&mut stats, (rank, part[i + 2]));
                                }
                            }

                            part[i] = rank;
                            part.remove(i + 1);
                        }
                        i += 1;
                    }
                    stats
                })
                .collect();

            for (pair, freq) in freqs {
                stats
                    .entry(pair)
                    .and_modify(|count| *count += freq)
                    .or_insert(freq);
            }
            let _ = pb.update(1);
        }
        println!("Vocabulary has been built successfully.");

        Self::new(name, pattern, encoder, decoder, special_tokens)
    }

    /// Save the vocabulary in a `.smtkn` file to a provided directory.
    pub fn save(&self, dir: &str) -> Result<(), Error> {
        let dir = Path::new(dir);

        let mut path = PathBuf::from(dir);
        path.push(Path::new(&format!("{}.smtkn", self.name)));

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let mut sorted: Vec<_> = self.encoder.clone().into_iter().collect();
        sorted.sort_by_key(|&(_, rank)| rank);

        for (token, rank) in sorted {
            let encoded_token = BASE64_STANDARD.encode(&token);
            writeln!(writer, "{} {}", encoded_token, rank)?;
        }

        Ok(())
    }

    /// Load the tokenizer from a vocabulary file.
    pub fn load(path: &str, pattern: &str, special_tokens: HashSet<&str>) -> Result<Self, Error> {
        let pattern = Regex::new(pattern).unwrap();

        let path = Path::new(path);
        if path.is_dir() {
            Err(Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "{} is a directory. Please provide a path to a file with .smtkn extension.",
                    path.to_str().unwrap()
                ),
            ))?
        }

        let name = String::from(path.file_stem().unwrap().to_str().unwrap());

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut encoder = HashMap::default();
        let mut decoder = HashMap::default();

        for line in reader.lines() {
            let line = line.unwrap();
            let parts: Vec<&str> = line.split(' ').collect();

            let token = BASE64_STANDARD.decode(parts[0]).unwrap();
            let rank = parts[1].parse::<u32>().unwrap();

            encoder.insert(token.clone(), rank);
            decoder.insert(rank, token);
        }
        Ok(Self::new(name, pattern, encoder, decoder, special_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> BytePairTokenizer {
        let name = String::from("test");
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

        BytePairTokenizer::new(name, pattern, encoder, decoder, special_tokens)
    }

    fn train_setup(vocab_size: Rank) -> BytePairTokenizer {
        let data = "abababcd";
        let pattern = r"\S+|\s+\S+";
        let name = String::from("test");
        let special_tokens: HashSet<&str> = HashSet::from(["<|endoftext|>"]);
        BytePairTokenizer::train(name, data, pattern, vocab_size, special_tokens)
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
            tok.encode("Hello<|endoftext|>, world!", &allowed_special),
            &[0, 1, 2, 15, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn decode_tokens() {
        let tok = setup();

        assert_eq!(
            tok.decode(&[0, 1, 2, 15, 3, 4, 5, 6, 7]).unwrap(),
            "Hello<|endoftext|>, world!"
        );

        assert_eq!(
            tok.decode(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap(),
            "Hello, world!"
        );
    }

    #[test]
    fn decode_fails_with_token_error() {
        let vocab_size = 260;
        let tokenizer = train_setup(vocab_size);
        assert!(matches!(
            tokenizer.decode(&[200, 300]).unwrap_err(),
            DecodeError::TokenError(_)
        ));
    }

    #[test]
    fn decode_fails_with_utf8_error() {
        let vocab_size = 260;
        let tokenizer = train_setup(vocab_size);
        assert!(matches!(
            tokenizer.decode(&[200]).unwrap_err(),
            DecodeError::Utf8Error(_)
        ));
    }

    #[test]
    fn train_bpe() {
        let vocab_size = 260;
        let tokenizer = train_setup(vocab_size);

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
    }

    #[test]
    fn save_and_load() {
        let vocab_size = 360;
        let path = "../tmp/";
        let pattern = r"\S+|\s+\S+";
        let special_tokens = HashSet::from(["<|endoftext|>"]);
        let tokenizer = train_setup(vocab_size);
        let _ = tokenizer.save(path);

        let tokenizer =
            BytePairTokenizer::load("../tmp/test.smtkn", pattern, special_tokens).unwrap();

        let decoder = tokenizer.decoder;

        assert_eq!(tokenizer.name, "test");
        assert_eq!(decoder[&256], vec![b'a', b'b']);
        assert_eq!(decoder[&257], vec![b'a', b'b', b'a', b'b']);
    }
}
