
use std::cell::RefCell;
use std::io::{self, BufRead};
use std::fs::File;
use std::path::Path;
use std::collections::{HashMap};
use serde::Serialize;
use roaring::RoaringBitmap;

use rand::seq::SliceRandom;
use rand::SeedableRng;
#[macro_use]
extern crate lazy_static;
extern crate partial_sort;

use partial_sort::PartialSort;

const NUM_HINTS: u8 = 3_u8.pow(5);
const MATCHING_HINT: u8 = NUM_HINTS - 1;
const PARTITION_INIT: Option<Vec<u16>> = None;
const BIG_YOSHI: usize = usize::MAX / 2;
const NUM_TRIALS: usize = 1000;
const BEAM_WIDTH: usize = 80;

fn load_index_word_lookups() -> (Vec<String>, HashMap<String, u16>) {
    let path = Path::new("words/wordlist.txt");
    let file = match File::open(&path) {
        Err(why) => panic!("{}", why),
        Ok(file) => file,
    };
    let mut index_to_word = Vec::new();
    let mut word_to_index = HashMap::new();
    let lines = io::BufReader::new(file).lines().enumerate();
    for (i, line) in lines {
        let val = line.unwrap();
        index_to_word.push(val.clone());
        word_to_index.insert(val.clone(), i as u16);
    }
    (index_to_word, word_to_index)
}

fn load_secrets() -> Vec<u16> {
    let path = Path::new("words/secret.txt");
    let file = match File::open(&path) {
        Err(why) => panic!("{}", why),
        Ok(file) => file,
    };
    let mut words = Vec::new();
    let lines = io::BufReader::new(file).lines().enumerate();
    for (_, line) in lines {
        let val = line.unwrap();
        words.push(WORD2INDEX[&val.to_owned()]);
    }
    words
}

fn load_hint_lookup() -> Vec<u8> {
    let path = Path::new("lookup.bin");
    std::fs::read(path).unwrap()
}

lazy_static! {

    static ref INDEX2WORD: Vec<String> = load_index_word_lookups().0;
    static ref WORD2INDEX: HashMap<String, u16> = load_index_word_lookups().1;
    static ref LIST_SIZE: usize = INDEX2WORD.len();
    static ref ALL_WORDS: Vec<u16> = (0..INDEX2WORD.len() as u16).collect();
    static ref SECRETS: Vec<u16> = load_secrets();
    static ref NUM_SECRETS: usize = SECRETS.len();

    static ref HINT_LOOKUP: Vec<u8> = load_hint_lookup();
}



#[inline(always)]
fn get_feedback(secret: u16, guess: u16) -> u8 {
    HINT_LOOKUP[guess as usize * INDEX2WORD.len() + secret as usize]
}

type CacheType = HashMap<Vec<u16>, (Strategy, usize)>;

#[derive(Clone, Serialize)]
struct Strategy {
    guess: u16,
    children: Box<HashMap<u8, Strategy>>,
}

impl Strategy {
    fn new() -> Strategy {
        Strategy {
            guess: u16::MAX,
            children: Box::new(HashMap::new()),
        }
    }
}

struct Partitions {
    partitions: [Option<Vec<u16>>; NUM_HINTS as usize],
    count: usize,
}

impl Partitions {
    fn from_targets(targets: &Vec<u16>, guess: u16) -> Partitions {
        let mut p = Partitions {
            partitions: [PARTITION_INIT; NUM_HINTS as usize],
            count: 0,
        };
        for secret in targets.iter() {
            let hint = get_feedback(*secret, guess);
            if p.partitions[hint as usize] == None {
                p.partitions[hint as usize] = Some(Vec::new());
                p.count += 1;
            }
            p.partitions[hint as usize].as_mut().unwrap().push(*secret);
        }
        return p;
    }
}

struct CountingPartitions {
    partitions: [u16; NUM_HINTS as usize],
    count: usize,
}

impl CountingPartitions {
    fn from_targets(targets: &Vec<u16>, guess: u16) -> CountingPartitions {
        let mut p = CountingPartitions {
            partitions: [0; NUM_HINTS as usize],
            count: 0,
        };
        for secret in targets.iter() {
            let hint = get_feedback(*secret, guess);
            if p.partitions[hint as usize] == 0 {
                p.count += 1;
            }
            let c = &mut p.partitions[hint as usize];
            *c += 1;
        }
        return p;
    }

    fn heuristic(&self) -> f32 {
        let mut result = 0.0_f32;
        for i in 0..NUM_HINTS {
            let p = self.partitions[i as usize].clone() as f32;
            if p == 0.0 {
                continue;
            }
            if i == MATCHING_HINT {
                result -= p;
            }
            result += p.ln() * p
        }
        result
    }
}

fn optimize(
    secrets: &Vec<u16>,
    choices: &Vec<u16>,
    depth: usize,
    cache: &mut CacheType,
) -> (Strategy, usize) {
    if secrets.len() == 1 {
        let strat = Strategy {
            guess: secrets[0],
            children: Box::new(HashMap::new()),
        };
        return (strat, 1);
    }
    if cache.contains_key(secrets) {
        return cache[secrets].clone();
    }
    let num_secrets = secrets.len();
    let mut found_best = false;
    // When three or less options, always optimal to try guessing one of the choices
    let choices = if secrets.len() <= 3 { secrets } else { choices };
    let mut guess_partitions = Vec::with_capacity(choices.len());
    let secret_set = std::collections::HashSet::<u16>::from_iter(secrets.clone());

    // Initial pass: look for perfect splitter
    for guess in secrets.iter() {
        let partitions: CountingPartitions = CountingPartitions::from_targets(secrets, *guess);
        if partitions.count <= 1 {
            // No information introduced
            continue;
        }
        if partitions.count == num_secrets {
            guess_partitions.clear();
            guess_partitions.push((partitions.heuristic(), *guess));
            found_best = true;
            break;
        } else {
            guess_partitions.push((partitions.heuristic(), *guess));
        }
    }

    // Second pass: try everything else
    if !found_best {
        for guess in choices.iter() {
            if secret_set.contains(guess) {
                continue;
            }
            let partitions = CountingPartitions::from_targets(secrets, *guess);
            if partitions.count <= 1 {
                // No information introduced
                continue;
            }
            if partitions.count == num_secrets {
                guess_partitions.clear();
                guess_partitions.push((partitions.heuristic(), *guess));
                break;
            } else {
                guess_partitions.push((partitions.heuristic(), *guess));
            }
        }
    }
    let top_n = std::cmp::min(BEAM_WIDTH, guess_partitions.len());
    guess_partitions.partial_sort(top_n, |a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut best_total_depth = BIG_YOSHI;
    let mut best_strategy = Strategy::new();
    for (_, guess) in guess_partitions.iter().take(top_n) {
        let mut strategy = Strategy {
            guess: *guess,
            children: Box::new(HashMap::new()),
        };
        let partitions = Partitions::from_targets(secrets, *guess);

        let mut total_depth = 0;
        for i in 0..NUM_HINTS {
            let sub_targets = &partitions.partitions[i as usize];
            if *sub_targets == None {
                continue;
            }
            let sub_secrets = sub_targets.as_ref().unwrap();
            let result;
            if cache.contains_key(sub_secrets) {
                let cached = cache[sub_secrets].clone();
                result = (cached.0, cached.1);
            } else {
                result = optimize(&sub_secrets, choices, depth + 1, cache);
            }
            cache.insert(sub_secrets.clone(), (result.0.clone(), result.1));
            let sub_depth = result.1;

            if i != MATCHING_HINT {
                total_depth += sub_depth;
            }
            strategy.children.insert(i, result.0);
        }
        total_depth += num_secrets;
        if total_depth < best_total_depth {
            best_total_depth = total_depth;
            best_strategy = strategy;
        }
    }
    cache.insert(secrets.clone(), (best_strategy.clone(), best_total_depth));
    (best_strategy, best_total_depth)
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct Observation {
    guess: u16,
    feedback: u8,
}

fn run_trial(rng: &mut rand::rngs::StdRng, cache: &mut CacheType) {
    // let starting_words: [&str; 2] = ["salet", "reast"];
    // let starting_words: [&str; 1] = ["salet"];
    // let starting_words: [&str; 1] = ["reast"];
    let starting_words: [&str; 10] = ["salet", "reast", "crate", "trace", "slate", "crane", "carle", "slane", "carte", "torse"];
    // let starting_words: [&str; 7] = ["salet", "reast", "crate", "trace", "carle", "slane", "torse"];
    // let starting_words: [&str; 15] = ["salet", "reast", "crate", "trace", "slate", "crane", "carle", "slane", "carte", "torse", "slant", "trice", "least", "trine", "prate"];
    let group_size = SECRETS.len() / starting_words.len() + if SECRETS.len() % starting_words.len() != 0 {1} else {0};
    let enable_conjugates_pruning = true;
    let mut secret_perm: Vec<u16> = SECRETS.clone();
    secret_perm.shuffle(rng);

    // Maps observations to a bitmap containing the possible words that match those observations.
    let mut observation_map: HashMap<Vec<Observation>, Box<RefCell<RoaringBitmap>>> = HashMap::new();
    // Maps observations to count of how many times we see that see that observation over the 2315 words.
    let mut observation_count: HashMap<Vec<Observation>, u16> = HashMap::new();
    // Observations we've seen so far at each of the 2315 positions.
    let mut observations: Vec<Vec<Observation>> = (0..SECRETS.len()).map(|_| Vec::new()).collect();

    // Initialize observation_map with "empty observation", and full set of secret words.
    let all_set = Box::new(RefCell::new(RoaringBitmap::new()));
    for i in SECRETS.iter() {
        all_set.borrow_mut().insert(*i as u32);
    }
    observation_map.insert(Vec::new(), all_set);
    observation_count.insert(Vec::new(), 1 as u16);

    let mut solved_count = 0;
    let mut misses = 0;

    // Initial guesses by mixing our starting words across the 2315 positions
    let mut guesses = vec![];
    for i in 0..SECRETS.len() {
        guesses.push(WORD2INDEX[starting_words[std::cmp::min(starting_words.len() - 1, i / group_size)]])
    }

    // Keep going until everything is solved
    while solved_count != SECRETS.len() {
        solved_count = 0;
        let mut new_observation_map: HashMap<Vec<Observation>, Box<RefCell<RoaringBitmap>>> = HashMap::new();
        let mut new_observation_count: HashMap<Vec<Observation>, u16> = HashMap::new();
        for i in 0..SECRETS.len() {
            let guess = guesses[i];
            let secret = secret_perm[i];
            let feedback = get_feedback(secret, guess);
            if feedback == MATCHING_HINT {
                solved_count += 1;
            } else {
                misses += 1;
            }
            let observation = Observation{
                guess: guess,
                feedback: feedback,
            };
            let prev_bitmap = observation_map[&observations[i]].borrow();
            observations[i].push(observation);
            let key = observations[i].clone();
            if !new_observation_map.contains_key(&key) {
                // Add new observation mapping by filter previous version of the bitmap
                let new_bitmap = Box::new(RefCell::new(RoaringBitmap::new()));
                for prev_secret in prev_bitmap.iter() {
                    if get_feedback(prev_secret as u16, guess) == feedback {
                        new_bitmap.borrow_mut().insert(prev_secret);
                    }
                }
                new_observation_map.insert(key.clone(), new_bitmap);
                new_observation_count.insert(key.clone(), 1);
            } else {
                new_observation_count.insert(key.clone(), new_observation_count[&key] + 1);
            }
        }

        // Prune naked conjugates
        let mut info_gained = true;
        let mut total_revised = 0;
        while enable_conjugates_pruning && info_gained {
            info_gained = false;
            let observations = new_observation_map.keys().cloned().collect::<Vec<_>>();
            for observation in &observations {
                let mut total_superset_of = 0;
                for observation2 in &observations {
                    if new_observation_map[observation].borrow().is_superset(&new_observation_map[observation2].borrow()) {
                        total_superset_of += new_observation_count[observation2];
                    }
                }
                let num_secrets = new_observation_map[observation].borrow().len() as u16;
                if total_superset_of == num_secrets {
                    for observation2 in &observations {
                        if observation2 == observation {
                            continue;
                        }
                        let mut other_secrets = new_observation_map[observation2].borrow_mut();
                        let secrets = new_observation_map[observation].borrow();
                        if secrets.is_superset(&other_secrets) {
                            continue;
                        }
                        for secret in secrets.iter() {
                            let found = other_secrets.remove(secret);
                            info_gained |= found;
                            total_revised += found as u32;
                        }
                    }
                } else if total_superset_of > num_secrets {
                    panic!("Shouldn't happen :D");
                }
            }
        }
        let mut total = 0;
        for i in new_observation_count.values() {
            total += i;
        }
        // println!("total={} total_revised={} solved_count={} score={}", total, total_revised, solved_count, misses + solved_count);
        observation_map = new_observation_map;
        observation_count = new_observation_count;

        // Update guesses
        for i in 0..SECRETS.len() {
            let secret_subset = observation_map[&observations[i]].borrow();
            let (strat, _) = optimize(&secret_subset.iter().map(|x| x as u16).collect(), &ALL_WORDS, 0, cache);
            guesses[i] = strat.guess;
        }
    }
    println!("{}", misses + solved_count)

}

fn main() {
    let mut rng = rand::rngs::StdRng::from_seed([1; 32]);
    let mut cache: CacheType = HashMap::new();
    for _ in 0..(NUM_TRIALS) {
        run_trial(&mut rng, &mut cache);
    }
}
