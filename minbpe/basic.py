from .base import get_stats, merge_vocab

class BasicTokenizer:
    """Basic Byte Pair Encoding tokenizer
    args:
        vocab: list of integers
        merge_id: integer
    """
    def __init__(self, text, vocab_size):
        self.vocab = self._build_vocab()
        self.merges = {}

    def _build_vocab(self):
        """building the initial vocabulary with the range of 256
        args:
            None
        
        returns: initial vocabulary dictionary
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        return vocab
    
    def fit(self, text, vocab_size, verbose=False):
        """train the tokenizer given the text corpus and vocab size
        args:
            text: string
            vocab_size: integer
        
        returns: 
        """
        num_merges = vocab_size - 256
        ids = list(tokens) # copy so we don't destroy the original list
        tokens = text.encode("utf-8") # raw bytes
        tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
        merges = {} # (int, int) -> int
        vocab = self.vocab.copy()
        for i in range(num_merges):
            stats = get_stats(ids)
            most_frequent_pair = max(stats, key=stats.get)
            idx = len(vocab)+i
            vocab = merge_vocab(vocab, stats,idx)
            merges[most_frequent_pair] = idx
            vocab[idx] = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]

             # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {most_frequent_pair} -> {idx} ({vocab[idx]}) had {stats[most_frequent_pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encoder(self, text):
        """encode a text
        args:
            text: string
        
        returns: list of integers
        """
        pass
    
    def decoder(self, ids):
        """decode a list of ids
        args:
            ids: list of integers
        
        returns: string
        """
        pass