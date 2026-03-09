from __future__ import annotations

import re
from typing import Dict, List, Tuple
from collections import Counter

class L104Tokenizer:
    """Byte-Pair Encoding style tokenizer with vocabulary building.

    Inspired by DeepSeek-V3's 102,400-token vocabulary.
    Builds subword units from corpus, handles unknown tokens via character fallback.
    """

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._special_tokens = {
            "<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
            "<SEP>": 4, "<CLS>": 5, "<MASK>": 6,
        }
        self._vocab_built = False
        self._word_freqs: Counter = Counter()
        self._merge_rules: List[Tuple[str, str]] = []
        # Initialize with special tokens
        for tok, idx in self._special_tokens.items():
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

    def _preprocess(self, text: str) -> List[str]:
        """Normalize and split text into words."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.split()

    def build_vocab(self, corpus: List[str], min_freq: int = 2):
        """Build BPE vocabulary from corpus."""
        # Count character-level frequencies
        char_vocab: Counter = Counter()
        word_splits: Dict[str, List[str]] = {}

        for text in corpus:
            for word in self._preprocess(text):
                self._word_freqs[word] += 1
                chars = list(word) + ["</w>"]
                word_splits[word] = chars
                for c in chars:
                    char_vocab[c] += self._word_freqs[word]

        # Initialize vocab with characters
        next_id = len(self._special_tokens)
        for char, freq in char_vocab.most_common():
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

        # BPE merge iterations (capped for performance on large corpora)
        max_merges = min(self.vocab_size - next_id, 500)
        for _ in range(max_merges):
            pairs = self._count_pairs(word_splits)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_freq:
                break

            self._merge_rules.append(best_pair)
            merged = best_pair[0] + best_pair[1]

            if merged not in self.token_to_id:
                self.token_to_id[merged] = next_id
                self.id_to_token[next_id] = merged
                next_id += 1

            # Apply merge to all word splits
            word_splits = self._apply_merge(word_splits, best_pair)

            if next_id >= self.vocab_size:
                break

        self._vocab_built = True

    def _count_pairs(self, word_splits: Dict[str, List[str]]) -> Counter:
        """Count adjacent token pairs across vocabulary."""
        pairs = Counter()
        for word, splits in word_splits.items():
            freq = self._word_freqs.get(word, 1)
            for i in range(len(splits) - 1):
                pairs[(splits[i], splits[i + 1])] += freq
        return pairs

    def _apply_merge(self, word_splits: Dict[str, List[str]],
                     pair: Tuple[str, str]) -> Dict[str, List[str]]:
        """Apply a BPE merge to all word splits."""
        new_splits = {}
        for word, splits in word_splits.items():
            new_word = []
            i = 0
            while i < len(splits):
                if i < len(splits) - 1 and splits[i] == pair[0] and splits[i + 1] == pair[1]:
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(splits[i])
                    i += 1
            new_splits[word] = new_word
        return new_splits

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text into token IDs."""
        tokens = [self._special_tokens["<CLS>"]]
        for word in self._preprocess(text):
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        tokens.append(self._special_tokens["<EOS>"])
        return tokens

    def _tokenize_word(self, word: str) -> List[int]:
        """Tokenize a single word using BPE merges."""
        chars = list(word) + ["</w>"]

        # Apply learned merges
        for pair in self._merge_rules:
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    new_chars.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars

        return [self.token_to_id.get(c, self._special_tokens["<UNK>"]) for c in chars]

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(tid, "<UNK>") for tid in token_ids
                  if tid not in self._special_tokens.values()]
        text = "".join(tokens).replace("</w>", " ").strip()
        return text

    @property
    def vocab_count(self) -> int:
        return len(self.token_to_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2: SEMANTIC ENCODER — TF-IDF + Embedding Similarity
# ═══════════════════════════════════════════════════════════════════════════════
