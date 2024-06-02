import time
from collections import OrderedDict
from multiprocessing import Process, Queue, Pipe
from argparse import ArgumentParser
import os

import regex as re
from datasets import load_dataset

from utils import GPT4_SPLIT_PATTERN, get_pair_counts, merge_counts, merge_tokens, render_token
from tokenizer import Tokenizer


def split_dataset(dataset, num_splits):
  """
  Split the dataset into multiple splits, with the last one containing more entries
  """

  size = len(dataset)
  split_size = size // num_splits
  splits = []
  for i in range(0, size, split_size):
      if (size - i) < 2 * split_size:
          split = dataset[i:-1]
      else:
          split = dataset[i:i+split_size]
      splits.append(split)

  return splits

def bpe(data_split, counts_queue, merge_pipe, num_merges):
  """
  Byte pair encoding algorithm on one node. For each iteration, get which ids to merge from master process,
  merge, and send back the count of pairs
  """

  gpt4_compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)

  text_chunks = []
  for text in data_split['text']:
    text = text.strip() 
    if len(text) > 0:
      text_chunks += re.findall(gpt4_compiled_pattern, text)

  ids = [list(ch.encode("utf-8")) for ch in text_chunks]
  for _ in range(num_merges):
    counts = {}
    for chunk_ids in ids:
      counts = get_pair_counts(chunk_ids, counts)
    counts_queue.put(counts)
    pair, idx = merge_pipe.recv()
    ids = [merge_tokens(chunk_ids, pair, idx) for chunk_ids in ids]

def parallel_bpe(n_vocab, dataset, num_workers, log=False):
  """
  Parallel implementation of byte pair encoding algorithm.
  Each worker will process a subset of the dataset and send back the counts of pairs to the master process.
  The master process tally up the counts and pick which pair to to merge, and send the decision back to the worker.
  """

  num_merges = n_vocab - 256
  counts_queue = Queue()
  merge_pipes = [Pipe() for _ in range(num_workers)]
  dataset_splits = split_dataset(dataset, num_workers)

  workers = [
    Process(target=bpe, args=(split, counts_queue, pipe[0], num_merges)) 
    for (split, pipe) in zip(dataset_splits, merge_pipes)
  ]
  for w in workers:
      w.start()

  merges = OrderedDict()
  vocabs = {idx: bytes([idx]) for idx in range(256)}
  for i in range(num_merges):
    start = time.perf_counter()
    counts= {}
    for w in workers:
      counts = merge_counts(counts, counts_queue.get())
    pair = max(counts, key=counts.get)
    idx = 256 + i
    merges[pair] = idx
    vocabs[idx] = vocabs[pair[0]] + vocabs[pair[1]]
    for pipe in merge_pipes:
      pipe[1].send((pair, idx))

    end = time.perf_counter()
    if log:
      print(f'{pair} -> {idx}     {end-start} secs')
  
    return merges, vocabs


def save_merges(merges, file_path):
  with open(file_path, 'w') as f:
    for pair, _ in merges.items():
      idx1, idx2 = pair 
      f.write(f'{idx1} {idx2}\n')


def save_vocabs(vocab, merges, file_path):
    inverted_merges = {idx: pair for pair, idx in merges.items()}
    with open(file_path, 'w') as f:
      for idx, token in vocab.items():
        if idx <= 256:
          continue
        pair = inverted_merges[idx]
        token = render_token(token)
        token1, token2 = map(lambda idx: render_token(vocab[idx]), pair)

        f.write(f'|{token1}|{token2}|'.ljust(10) + f'->  |{token}|'.ljust(16) + f'{idx}\n')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-w", "--workers", type=int, default=8,
                        help="number of processes to parallelize bpe")
    parser.add_argument("-n", "--num-vocab", type=int, default=300,
                        help="number of final vocab size")
    parser.add_argument("-d", "--dest-path", type=str, default='data_cache/',
                        help="path to save bpe results to")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    num_workers = args.workers
    num_vocab = args.num_vocab
    dest_folder = args.dest_path
    os.makedirs(dest_folder, exist_ok=True)

    DATASET = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")
    print('loaded dataset')

    merges, vocabs = parallel_bpe(num_vocab, DATASET, num_workers, log=True)

    merges_file = os.path.join(dest_folder, 'merges.txt')
    vocabs_file = os.path.join(dest_folder, 'vocabs.txt')
    save_merges(merges, merges_file)
    save_vocabs(vocabs, merges, vocabs_file)

    tokenizer = Tokenizer(merges_file)
