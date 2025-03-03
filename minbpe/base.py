def get_stats(ids, counts=None):
    """returns a dictionary of counts of consecutive pairs of ids
    args:
        ids: list of integers
    
    returns: dictionary of counts of consecutive pairs of ids
    """
    counts = {} if counts is None else counts
    
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_vocab(ids, pair, merge_id):
    """merge the most frequent pair of ids in the vocabulary
    args:
        vocab: list of integers
        stats: dictionary of counts of consecutive pairs of ids
        merge_id: integer
    
    returns: list of integers after merging the most frequent pair
    """
    new_vocab = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            new_vocab.append(merge_id)
            i += 2
        else:
            new_vocab.append(ids[i])
            i += 1
    return new_vocab

class Tokenizer:
    """base class for tokenizers"""
    
    def __init__(self):
        self.merges = {}
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab()

    def fit(self, text, vocab_size, verbose=False):
         # train a vocabulary of size vocab_size from given text
         raise NotImplementedError
    
    def encoder(self, text):
        """encode a text
        args:
            text: string
        
        returns: list of integers
        """
        raise NotImplementedError
    
    def decoder(self, ids):
        """decode a list of integers
        args:
            ids: list of integers
        
        returns: string
        """
        raise NotImplementedError

    def _build_vocab(self):
        """building the initial vocabulary with the range of 256
        args:
            None
        
        returns: initial vocabulary dictionary
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        # print("initial vocab",vocab)
        return vocab
    

