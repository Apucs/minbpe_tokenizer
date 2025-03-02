def get_stats(ids):
    """returns a dictionary of counts of consecutive pairs of ids
    args:
        ids: list of integers
    
    returns: dictionary of counts of consecutive pairs of ids
    """
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_vocab(vocab, stats, merge_id):
    """merge the most frequent pair of ids in the vocabulary
    args:
        vocab: list of integers
        stats: dictionary of counts of consecutive pairs of ids
        merge_id: integer
    
    returns: list of integers after merging the most frequent pair
    """
    new_vocab = []
    i = 0
    while i < len(vocab):
        if i < len(vocab) - 1 and (vocab[i], vocab[i + 1]) in stats:
            new_vocab.append(merge_id)
            i += 2
        else:
            new_vocab.append(vocab[i])
            i += 1
    return new_vocab

# class Tokenizer:
#     def __init__(self):
#         self.merges = {}
#         self.stats = {}

