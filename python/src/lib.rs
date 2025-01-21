use std::collections::HashSet;
use std::num::NonZeroU64;
use std::thread;

use fancy_regex::Regex;
use kdam::{tqdm, BarExt};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyBytes;
use pyo3::PyResult;
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;

type Byte = u8;
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

pub struct FakeThreadId(NonZeroU64);
fn hash_current_thread() -> usize {
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

#[derive(Debug, Clone)]
struct DecodeKeyError {
    token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

#[pyclass]
pub struct BytePairTokenizer {
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

    pub fn new(
        pattern: Regex,
        encoder: HashMap<Vec<Byte>, Rank>,
        decoder: HashMap<Rank, Vec<Byte>>,
        special_tokens: HashSet<String>,
    ) -> Self {
        let mut rank = encoder.len() as Rank;
        let pattern = (0..MAX_NUM_THREADS).map(|_| pattern.clone()).collect();
        let mut special_encoder = HashMap::default();
        let mut special_decoder = HashMap::default();
        for special in &special_tokens {
            let bytes = special.as_bytes().to_vec();
            special_encoder.insert(bytes.clone(), rank);
            special_decoder.insert(rank, bytes);

            rank += 1;
        }

        let special_pattern = Regex::new(
            &special_tokens
                .iter()
                .map(|spl| fancy_regex::escape(spl))
                .collect::<Vec<_>>()
                .join("|"),
        )
        .unwrap();
        let special_pattern = (0..MAX_NUM_THREADS)
            .map(|_| special_pattern.clone())
            .collect();

        Self {
            pattern,
            encoder,
            decoder,
            special_pattern,
            special_encoder,
            special_decoder,
        }
    }

    fn encode_native(&self, bytes: &[Byte]) -> Vec<Rank> {
        if bytes.len() == 1 {
            return vec![self.encoder[bytes]];
        }

        let mut parts = Vec::with_capacity(bytes.len() + 1);
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
                    *self
                        .encoder
                        .get(&bytes[parts[i].0..=parts[i + 2].0])
                        .unwrap_or(&Rank::MAX)
                } else {
                    Rank::MAX
                }
            }
        };

        while min_rank.0 != Rank::MAX {
            let i = min_rank.1;
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

    fn decode_native(&self, tokens: Vec<Rank>) -> Result<Vec<Byte>, DecodeKeyError> {
        let mut bytes: Vec<Byte> = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let token_bytes = match self.decoder.get(&token) {
                Some(bytes) => bytes,
                None => self
                    .special_decoder
                    .get(&token)
                    .ok_or(DecodeKeyError { token })?,
            };
            bytes.extend(token_bytes);
        }
        Ok(bytes)
    }
}

#[pymethods]
impl BytePairTokenizer {
    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        let regex = self.get_tl_regex();
        let mut ranks = vec![];
        for part in regex.find_iter(text) {
            let bytes = part.unwrap().as_str().as_bytes();
            match self.encoder.get(bytes) {
                Some(rank) => ranks.push(*rank),
                None => ranks.extend(self.encode_native(bytes)),
            }
        }
        ranks
    }

    pub fn encode(
        &self,
        py: Python,
        text: &str,
        allowed_special: HashSet<PyBackedStr>,
    ) -> Vec<Rank> {
        py.allow_threads(|| {
            let allowed_special: HashSet<&str> =
                allowed_special.iter().map(|s| s.as_ref()).collect();

            let mut ranks = vec![];
            let special_regex = self.get_tl_special_regex();

            let mut start = 0;
            loop {
                let special = special_regex.find_from_pos(text, start).unwrap();
                match special {
                    Some(special) => {
                        if !allowed_special.contains(special.as_str()) {
                            start = special.start() + 1;
                            continue;
                        }
                        ranks.extend(self.encode_ordinary(&text[start..special.start()]));
                        ranks.push(self.special_encoder[special.as_str().as_bytes()]);
                        start += special.end();
                    }
                    None => {
                        if start != text.len() {
                            ranks.extend(self.encode_ordinary(&text[start..text.len()]));
                        }
                        break;
                    }
                }
            }
            ranks
        })
    }

    pub fn decode(&self, py: Python, tokens: Vec<Rank>) -> Result<Py<PyBytes>, PyErr> {
        match py.allow_threads(|| self.decode_native(tokens)) {
            Ok(bytes) => Ok(PyBytes::new_bound(py, &bytes).into()),
            Err(e) => Err(pyo3::exceptions::PyKeyError::new_err(format!("{}", e))),
        }
    }

    #[new]
    pub fn train(
        data: &str,
        pattern: &str,
        vocab_size: Rank,
        special_tokens: HashSet<PyBackedStr>,
    ) -> Self {
        assert!(
            vocab_size >= 256,
            "Vocabulary size should atleast be 256 to cover all individual bytes"
        );

        let mut encoder: HashMap<Vec<Byte>, Rank> =
            (0..256).map(|b| (vec![b as Byte], b as Rank)).collect();
        let mut decoder: HashMap<Rank, Vec<Byte>> =
            (0..256).map(|b| (b as Rank, vec![b as Byte])).collect();

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

        println!("Starting to build vocabulary...");
        let mut pb = tqdm!(total = vocab_size as usize);
        while decoder.len() < vocab_size as usize {
            stats.retain(|_, v| *v > 0);
            let most_common_pair = match stats.par_iter().max_by_key(|&(_, count)| count) {
                None => break,
                Some((&most_common_pair, _)) => {
                    stats.remove(&most_common_pair);
                    [most_common_pair.0, most_common_pair.1]
                }
            };

            let rank = decoder.len() as Rank;
            let mut bytes = decoder.get(&most_common_pair[0]).unwrap().clone();
            bytes.extend(decoder.get(&most_common_pair[1]).unwrap());

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

        let special_tokens = special_tokens.iter().map(|s| s.to_string()).collect();
        Self::new(pattern, encoder, decoder, special_tokens)
    }
}

#[pymodule]
fn _smoltoken(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BytePairTokenizer>()?;
    Ok(())
}
