# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal CLIP BPE tokenizer and text embedder. No external dependencies beyond stdlib + torch.

Replicates the tokenization behavior of CLIPTokenizer / CLIPTokenizerFast
for use with traced OWLv2 models.
"""

import io
import json
import os
import re
from functools import lru_cache

import torch
import torch.nn.functional as F

_CKPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ckpts",
    "owlv2-base-patch16-ensemble.pt",
)


@lru_cache()
def _bytes_to_unicode():
    """Returns mapping from bytes (0-255) to unicode strings, avoiding control/whitespace chars."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word):
    """Return set of symbol pairs in a word (tuple of symbols)."""
    pairs = set()
    prev = word[0]
    for char in word[1:]:
        pairs.add((prev, char))
        prev = char
    return pairs


_PAT = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]|[^\sa-zA-Z0-9]+""",
    re.IGNORECASE,
)


class CLIPTokenizer:
    """Minimal CLIP BPE tokenizer."""

    def __init__(self, vocab=None, merges=None, max_length=16):
        """
        Args:
            vocab: Either a dict (token->id mapping) or a path to vocab.json.
            merges: Either a string (merges.txt contents) or a path to merges.txt.
            max_length: Max token sequence length (default 16 for OWLv2).
        """
        if isinstance(vocab, dict):
            self.encoder = vocab
        else:
            with open(vocab) as f:
                self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        if isinstance(merges, str) and "\n" in merges:
            lines = merges.strip().split("\n")
        else:
            with open(merges) as f:
                lines = f.read().strip().split("\n")
        # Skip header line
        merge_list = [tuple(line.split()) for line in lines[1:]]
        self.bpe_ranks = {m: i for i, m in enumerate(merge_list)}

        self.byte_encoder = _bytes_to_unicode()
        self.cache = {}

        self.sot_token = self.encoder.get("<|startoftext|>", 49406)
        self.eot_token = self.encoder.get("<|endoftext|>", 49407)
        self.pad_token = 0
        self.max_length = max_length

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i + 1 < len(word) and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        """Encode text string to list of token IDs."""
        text = text.lower().strip()
        tokens = []
        for match in _PAT.finditer(text):
            token = "".join(
                self.byte_encoder[b] for b in match.group(0).encode("utf-8")
            )
            bpe_tokens = self._bpe(token).split(" ")
            tokens.extend(self.encoder[t] for t in bpe_tokens)
        return tokens

    def __call__(self, texts, return_tensors="pt"):
        """Tokenize a list of strings, returning input_ids and attention_mask tensors."""
        if isinstance(texts, str):
            texts = [texts]

        all_ids = []
        all_masks = []
        for text in texts:
            ids = [self.sot_token] + self.encode(text) + [self.eot_token]
            # Truncate (keeping SOT at start)
            if len(ids) > self.max_length:
                ids = ids[: self.max_length - 1] + [self.eot_token]
            mask = [1] * len(ids)
            # Pad
            pad_len = self.max_length - len(ids)
            ids = ids + [self.pad_token] * pad_len
            mask = mask + [0] * pad_len
            all_ids.append(ids)
            all_masks.append(mask)

        result = {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }
        return result


class TextEmbedder:
    """Compute normalized text embeddings using the OWLv2 CLIP text encoder.

    Uses the same traced text encoder from the OWLv2 checkpoint — no additional
    dependencies beyond torch.
    """

    def __init__(self):
        if not os.path.exists(_CKPT_PATH):
            raise FileNotFoundError(
                f"OWLv2 checkpoint not found at {_CKPT_PATH}. "
                "Run 'python owl/export_owl.py' first."
            )
        checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

        self.text_encoder = torch.jit.load(
            io.BytesIO(checkpoint["text_encoder"]), map_location="cpu"
        )
        self.text_encoder.eval()

        self.tokenizer = CLIPTokenizer(
            vocab=checkpoint["tokenizer_vocab"],
            merges=checkpoint["tokenizer_merges"],
            max_length=config["max_seq_length"],
        )

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings into L2-normalized embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            Tensor of shape (len(texts), embed_dim) with L2-normalized rows.
        """
        tokens = self.tokenizer(texts)
        embeds = self.text_encoder(tokens["input_ids"], tokens["attention_mask"])
        return F.normalize(embeds, dim=-1)
