import nltk
from collections import Counter
import pickle
import os
import json # Ensure json is imported as it's used in build_vocab

class Vocabulary(object):
    def __init__(self, vocab_threshold, annotations_file, start_word, end_word, unk_word, pad_word, annotations_file_test=None): # Added pad_word
        self.vocab_threshold = vocab_threshold
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word # Store pad_word
        self.annotations_file = annotations_file
        self.annotations_file_test = annotations_file_test
        self.get_vocab()

    def get_vocab(self):
        if os.path.exists('vocab.pkl'):
            with open('vocab.pkl', 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
                # Also ensure special tokens are correctly set from loaded vocab if they were dynamic
                self.start_word = vocab.start_word
                self.end_word = vocab.end_word
                self.unk_word = vocab.unk_word
                self.pad_word = vocab.pad_word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open('vocab.pkl', 'wb') as f:
                pickle.dump(self, f)

    def build_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        # Add special tokens first to ensure consistent indices (pad=0, bos=1, eos=2, unk=3)
        self.add_word(self.pad_word) # Index 0
        self.add_word(self.start_word) # Index 1
        self.add_word(self.end_word) # Index 2
        self.add_word(self.unk_word) # Index 3

        with open(self.annotations_file, 'r') as f:
            caption_data = json.load(f)

        counter = Counter()
        for i, annotation in enumerate(caption_data['annotations']):
            caption = annotation['caption']
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

        words = [word for word, count in counter.items() if count >= self.vocab_threshold]

        for word in words:
            self.add_word(word)

        print(f"Total vocabulary size: {len(self.word2idx)}")

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    # Added denumericalize method
    def denumericalize(self, token_ids):
        words = []
        for token_id in token_ids:
            word = self.idx2word.get(token_id)
            # Only append if word exists and is not a special token other than unk (if unk is to be shown)
            # Exclude pad_word, start_word, and end_word as they are typically stripped from references/hypotheses
            if word and word != self.pad_word and word != self.start_word and word != self.end_word:
                 words.append(word)
        return words
