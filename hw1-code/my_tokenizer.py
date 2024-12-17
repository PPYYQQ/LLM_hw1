from tqdm import tqdm

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
    # print(ids)
    while i < len(ids):
        # print(i, len(ids))
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
        # print(i)
        # print(newids)
    return newids

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
        tokens = text.encode("utf-8")
        # print(tokens)
        tokens = list(map(int, tokens))
        num_merges = vocab_size - 256
        ids = list(tokens)
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in tqdm(range(num_merges)):
            stats = get_stats(ids)
            pair = max(stats, key = stats.get)
            idx = 256 + i
            # print(f"Merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    
        self.merges = merges
        self.vocab = vocab

        
    def encode(self, text):
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.

        Return:
            ids (list): list of integer-type tokens.
        """
        tokens = list(text.encode("utf-8"))
        merges = self.merges
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key = lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            idx = merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens
            
    def decode(self, ids):
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.

        Return:
            text (str): string-type data.
        """
        vocab = self.vocab
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors = "replace")
        return text

if __name__ == '__main__':
    # tokenizer = Tokenizer()
    # text = 'The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to'
    # tokenizer.train(text, 280)
    # # # print(tokens)
    # # tokens = list(map(int, tokens))
    # # # print(tokens)
    
    # # stats = get_stats(tokens)
    # # # print(sorted(((v,k) for k,v in stats.items()), reverse=True))
    # # top_pair = max(stats, key = stats.get)
    # # print(top_pair)
    # print(tokenizer.decode(tokenizer.encode('hello world')))
    # ood_text = text
    # print(tokenizer.decode(tokenizer.encode(ood_text)) == ood_text)

    


    pass