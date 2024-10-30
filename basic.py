import unicodedata

def get_state(idx,counts=None):
    """
    :param idx: 输入一个文本的encode格式
    :return: 每个相邻字符的出现次数
    """
    counts = {} if counts is None else counts
    for pair in zip(idx,idx[1:]):
        counts[pair] = counts.get(pair,0) + 1

    return counts

def merge(ids,pair,idx):
    """
    :param ids: 词表
    :param pair: 上一组最常见的字符组合
    :param idx:index
    :return:merge后的数组
    """
    new_idx = []
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_idx.append(idx)
            i += 2
        else:
            new_idx.append(ids[i])
            i += 1
    return new_idx


class Tokenizer:
    def __init(self):
        self.merge = {}
        self.pattern = ""  # 正则分开不同符号的作用，gpt2和gpt4使用的方法
        self.vocab = self._build_vocab()

    def train(self,vocab_size,text):
        raise  NotImplementedError

    def encode(self,text):
        raise NotImplementedError

    def decode(self,idx):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0,p1), idx in self.merge:
            vocab[idx] = p0 + p1

        return vocab


