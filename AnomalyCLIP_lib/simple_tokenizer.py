import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path="AnomalyCLIP_lib/bpe_simple_vocab_16e6.txt.gz"):
        self.encoder = {}
        self.decoder = {}
        self.bpe_ranks = {}
        self.cache = {}
        self.bpe_path = bpe_path

        with gzip.open(bpe_path) as f:
            for i, line in enumerate(f):
                line = line.decode("utf-8").strip()
                if line:
                    self.encoder[line] = i
                    self.decoder[i] = line

        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|\S\S+|\S|\n|\w+|\d+|\S\S|\S"""
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = set(zip(word, word[1:]))

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
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
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word
            pairs = set(zip(word, word[1:]))

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, ftfy.fix_text(text)):
            bpe_token_strs = [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ') if bpe_token in self.encoder]
            bpe_tokens.extend(bpe_token_strs)
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join(self.decoder[token] for token in tokens)
        return text
