from tqdm import tqdm
from .base import get_stats, merge_vocab, Tokenizer

class BasicTokenizer(Tokenizer):
    """Basic Byte Pair Encoding tokenizer
    args:
        vocab: list of integers
        merge_id: integer
    """
    def __init__(self):
        # self.merges = {}
        # self.vocab = self._build_vocab()
        super().__init__()
    
    def fit(self, text, vocab_size, verbose=False):
        """train the tokenizer given the text corpus and vocab size
        args:
            text: string
            vocab_size: integer
        
        returns: 
        """
        num_merges = vocab_size - 256
        tokens = text.encode("utf-8") # raw bytes
        ids = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
        merges = {} # (int, int) -> int
        vocab = self.vocab.copy()

        print(f"training started...")
        for i in tqdm(range(num_merges)):
            stats = get_stats(ids)
            most_frequent_pair = max(stats, key=stats.get)
            idx = len(vocab)+i
            ids = merge_vocab(ids, most_frequent_pair, idx)
            merges[most_frequent_pair] = idx
            # print("\n", len(vocab), idx, vocab)
            vocab[idx] = vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]]

             # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {most_frequent_pair} -> {idx} ({vocab[idx]}) had {stats[most_frequent_pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

        print(f"trained vocab with {len(vocab)} tokens")


    def encoder(self, text):
        """encode a text
        args:
            text: string
        
        returns: list of integers
        """

        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = merge_vocab(tokens, pair, idx)
        
        return tokens
    
    def decoder(self, ids):
        """decode a list of ids
        args:
            ids: list of integers
        
        returns: string
        """
        for id in ids:
            print("id:", id, "vocab:", self.vocab[id], "decode:", self.vocab[id].decode("utf-8", errors="replace"), "\n")

        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text