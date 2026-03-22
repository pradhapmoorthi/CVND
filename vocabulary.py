import nltk
from pycocotools.coco import COCO
import pickle
import os
import numpy as np
from collections import Counter

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, vocab_threshold, annotations_file, vocab_from_file=False,
                 start_word="<bos>", end_word="<eos>", unk_word="<unk>", pad_word="<pad>"):
        """Initialize the vocabulary."""
        self.vocab_threshold = vocab_threshold
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file or build it from captions."""
        if os.path.exists('./vocab.pkl') and self.vocab_from_file:
            with open('./vocab.pkl', 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file!')
        else:
            self.build_vocab()
            with open('./vocab.pkl', 'wb') as f:
                pickle.dump(self, f)
        
    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        # Ensure special tokens get distinct, sequential IDs at the beginning
        self.add_word(self.pad_word)
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

        print('Finished building vocabulary of %d words' % len(self))

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a word to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def denumericalize(self, tokens):
        """Convert a list of token IDs to a list of words."""
        words = []
        for token_id in tokens:
            word = self.idx2word.get(token_id, self.unk_word)
            # Filter out special tokens like start, end, pad from the output caption
            if word in [self.start_word, self.end_word, self.pad_word]: 
                continue
            words.append(word)
        return words

    def __call__(self, word):
        """Return the index of a word."""
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.word2idx)''')
