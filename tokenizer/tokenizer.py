
import regex as re
from collections import OrderedDict
from utils import get_counts, merge_tokens, GPT4_SPLIT_PATTERN

class Tokenizer():
  def __init__(self, file_path) -> None:
    self.file_path = file_path
    self.merges = self.load_merges(file_path)
    self.vocab = Tokenizer.get_vocabs(self.merges)


  def encode_chunk(self, text):
    ids = list(text.encode("utf-8"))
    while len(ids) > 2:
      counts = get_counts(ids)
      pair = min(counts, lambda p: self.merges.get(p, float('inf')))

      if pair not in self.merges:
        break
      idx = self.merges[pair]
      ids = merge_tokens(ids, pair, idx)

    return ids 

  def encode(self, text, merges):
    gpt4_compiled_pattern = re.compile(GPT4_SPLIT_PATTERN)
    text_chunks = re.findall(gpt4_compiled_pattern, text)

    ids = []
    for chunk in text_chunks:
      ids += self.encode_chunk(chunk, merges)

    return ids

  def decode(self, ids, vocab):
    text_bytes = b"".join(vocab[idx] for idx in ids)
    text = text_bytes.decode("utf-8", errors="replace")
    return text


  def load_merges(self, file_path):
    merges = OrderedDict()
    with open(file_path) as f:
      for idx, line in enumerate(f.readlines()):
        idx1, idx2 = map(int, line.strip().split())
        merges[(idx1, idx2)] = idx + 256

    return merges
  
  @staticmethod
  def get_vocabs(merges):
    vocabs = {idx: bytes([idx]) for idx in range(256)}
    for pair, idx in merges.items():
      vocabs[idx] = vocabs[pair[0]] + vocabs[pair[1]]

    return vocabs


