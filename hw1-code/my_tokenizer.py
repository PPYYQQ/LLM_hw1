class Tokenizer:
    def __init__(self):
        pass

    def train(self, text, vocab_size):
        """
        Train the tokenizer using BPE algorithm.
        Params:
            text (str): string-type data used to run BPE.
            vocab_size (int): the size of final vocabulary.

        Return:
            None
        """
        pass

    def encode(self, text):
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.

        Return:
            ids (list): list of integer-type tokens.
        """
        pass

    def decode(self, ids):
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.

        Return:
            text (str): string-type data.
        """
        pass


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    Params:
        ids (list): list of the ints(ids).
        replace pair with token idx

    Return:
        newids (list): list of the ints(ids) after replacement
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids-1):
            continue
        if ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
    return newids
    

if __name__ == '__main__':
    text = 'The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to'
    tokens = text.encode("utf-8")
    # print(tokens)
    tokens = list(map(int, tokens))
    # print(tokens)
    
    stats = get_stats(tokens)
    # print(sorted(((v,k) for k,v in stats.items()), reverse=True))
    top_pair = max(stats, key = stats.get)
    print(top_pair)