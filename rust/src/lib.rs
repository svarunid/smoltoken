//! A simple byte pair encoding tokenizer library. The code to encode text
//! was taken from [tiktoken](https://github.com/openai/tiktoken/) library.
//! You can use this to train BPE on your own data. :-)
#![feature(test)]

extern crate test;

// use rayon::prelude::*;
use fancy_regex::Regex;
use rustc_hash::FxHashMap as HashMap;

type Byte = u8;
type Rank = u32; // Type alias for token IDs

pub struct BytePairTokenizer {
    pattern: Regex,
    special_pattern: Regex,
    encoder: HashMap<Vec<Byte>, Rank>,
    decoder: HashMap<Rank, Vec<Byte>>,
}

#[derive(Debug, Clone)]
pub struct DecodeKeyError {
    token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

impl BytePairTokenizer {
    pub fn encode_native(&self, bytes: &[Byte]) -> Vec<Rank> {
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

    fn decode_native(&self, tokens: &[Rank]) -> Result<Vec<Byte>, DecodeKeyError> {
        let mut bytes: Vec<Byte> = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            bytes.extend(self.decoder.get(&token).ok_or(DecodeKeyError { token })?);
        }

        Ok(bytes)
    }
}

impl BytePairTokenizer {
    pub fn new(
        pattern: Regex,
        special_pattern: Regex,
        encoder: HashMap<Vec<Byte>, Rank>,
        decoder: HashMap<Rank, Vec<Byte>>,
    ) -> Self {
        BytePairTokenizer {
            pattern,
            special_pattern,
            encoder,
            decoder,
        }
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let mut ranks = vec![];
        for part in self.pattern.find_iter(text) {
            let bytes = part.unwrap().as_str().as_bytes();
            match self.encoder.get(bytes) {
                Some(token) => ranks.push(*token),
                None => ranks.extend(self.encode_native(bytes)),
            }
        }
        ranks
    }

    pub fn encode(&self, text: &str, allowed_special: Option<&[&str]>) -> Vec<Rank> {
        let mut ranks: Vec<Rank> = vec![];

        let mut start = 0;
        loop {
            let special = self.special_pattern.find_from_pos(text, start).unwrap();

            match special {
                Some(special) => {
                    if let Some(allowed_special) = allowed_special {
                        if !allowed_special.contains(&special.as_str()) {
                            start = special.start() + 1;
                            continue;
                        }
                    }

                    let text = &text[start..special.start()];
                    ranks.extend(self.encode_ordinary(text));
                    ranks.push(self.encoder[special.as_str().as_bytes()]);

                    start = special.end() + 1;
                }
                None => break,
            }
        }
        ranks
    }

    pub fn decode_ordinary(&self, tokens: &[Rank]) -> Result<Vec<Byte>, DecodeKeyError> {
        // Decoupling the implementation earlier to avoid code repetition when
        // implementing variants of `decode` method later.
        self.decode_native(tokens)
    }

    pub fn train(data: &str, pattern: &str, vocab_size: Rank, special_tokens: &[&str]) -> Self {
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

        // This is where the most of the time is spent. This loop needs to be optimized.
        while decoder.len() < vocab_size as usize {
            // We are calculating the frequencies of from scratch after each merge. Avoid
            // this behaviour. Rather incrementally add the frequency of newly merged
            // pair with the surrounding rank. Hence we can be moved this outside the loop.
            let mut stats: HashMap<(Rank, Rank), usize> = HashMap::default();
            for part in &parts {
                for pair in part.windows(2) {
                    let pair = (pair[0], pair[1]);
                    *stats.entry(pair).or_insert(0) += 1;
                }
            }

            match stats.iter().max_by_key(|&(_, count)| count) {
                None => break,
                Some((&most_common_pair, _)) => {
                    let rank = decoder.len() as Rank;

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
                        while i < part.len() - 1 {
                            if part[i] == most_common_pair.0 && part[i + 1] == most_common_pair.1 {
                                part[i] = rank;
                                part.remove(i + 1);
                            }
                            i += 1;
                        }
                    }
                }
            }
        }

        // Add special tokens to the end of vocabulary.
        for &special in special_tokens {
            let rank = decoder.len() as Rank;
            let bytes = special.as_bytes().to_vec();

            encoder.insert(bytes.clone(), rank);
            decoder.insert(rank, bytes);
        }

        let special_pattern = Regex::new(
            &special_tokens
                .iter()
                .map(|&spl| fancy_regex::escape(spl))
                .collect::<Vec<_>>()
                .join("|"),
        )
        .unwrap();

        BytePairTokenizer {
            pattern,
            special_pattern,
            encoder,
            decoder,
        }
    }
}
