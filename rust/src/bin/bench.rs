use smoltoken::{BytePairTokenizer, TokenizerDataSource};
use std::collections::HashSet;
use std::time::Instant;
fn main() {
    let vocab_size = 500;
    let name = String::from("test");
    let data = std::fs::read_to_string("../sample/tinystories.txt").unwrap();
    let pattern =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

    let time = Instant::now();
    BytePairTokenizer::train(
        name,
        pattern,
        vocab_size,
        HashSet::from(["<|endoftext|>"]),
        TokenizerDataSource::Text(&data),
    )
    .unwrap();
    println!("Time taken: {} sec", time.elapsed().as_secs_f32());
}
