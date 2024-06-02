import unicodedata

# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_pair_counts(ids, counts=None):
  """
  Count up the occurrences of pairs, potentially additionally to exisiting counts
  """

  counts = {} if counts is None else counts
  for pair in zip(ids, ids[1:]): 
    counts[pair] = counts.get(pair, 0) + 1
  return counts


def merge_counts(counts, additional_counts):
  """
  Merge two dictionaries of counts
  """

  new_counts = {}
  for idx, count in additional_counts.items():
    new_counts[idx] = counts.get(idx, 0) + count

  return new_counts
    
def merge_tokens(ids, pair, idx):
  """
  Merge a pair into a single token for the input ids
  """

  newids = []
  i = 0
  while i < len(ids):
    # if not at the very last position AND the pair matches, replace it
    if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# Copied from minBPE implementation by Karpathy https://github.com/karpathy/minbpe
def replace_control_characters(s: str) -> str:
  chars = []
  for ch in s:
    if unicodedata.category(ch)[0] != "C":
      chars.append(ch) # this character is ok
    else:
      chars.append(f"\\u{ord(ch):04x}") # escape
  return "".join(chars)

def render_token(t: bytes) -> str:
  # pretty print a token, escaping control characters
  s = t.decode('utf-8', errors='replace')
  s = replace_control_characters(s)
  return s