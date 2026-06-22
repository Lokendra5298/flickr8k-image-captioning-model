# src/vocabulary.py

import re
from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=5):
        self.min_freq = min_freq

        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        self.stoi = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3,
        }

        self.itos = {
            0: self.pad_token,
            1: self.start_token,
            2: self.end_token,
            3: self.unk_token,
        }

    def __len__(self):
        return len(self.stoi)

    def tokenizer(self, text):
        """
        Simple tokenizer:
        - lowercase
        - keep only letters and numbers
        """
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_vocab(self, captions):
        """
        Builds vocabulary from all training captions.
        Words appearing fewer than min_freq times are ignored.
        """
        counter = Counter()

        for caption in captions:
            tokens = self.tokenizer(caption)
            counter.update(tokens)

        idx = len(self.stoi)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, caption):
        """
        Converts caption text to token ids.
        """
        tokens = self.tokenizer(caption)

        numericalized = [self.stoi[self.start_token]]

        for token in tokens:
            if token in self.stoi:
                numericalized.append(self.stoi[token])
            else:
                numericalized.append(self.stoi[self.unk_token])

        numericalized.append(self.stoi[self.end_token])

        return numericalized

    def decode_indices(self, indices):
        """
        Converts token ids back to words.
        """
        words = []

        for idx in indices:
            word = self.itos.get(int(idx), self.unk_token)

            if word == self.end_token:
                break

            if word not in [self.start_token, self.pad_token]:
                words.append(word)

        return " ".join(words)