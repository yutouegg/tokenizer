from basic import (Tokenizer, get_state, merge)

class BaseTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self,vocab_size,text):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        byte_text = text.encode("utf-8")
        ids = list(byte_text)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            idx = 256 + i

            stats = get_state(ids)
            if len(stats) <= 2:
                break
            pair = max(stats, key=stats.get)
            ids = merge(ids,pair,idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merge = merges
        self.vocab = vocab



    def encode(self,text):
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)

        while len(ids) >= 2:
            # 找到可以合并的相邻对
            stats = get_state(ids)
            # 选择具有最低合并索引的对
            pair = min(stats, key=lambda p: self.merge.get(p, float("inf")))

            # 如果没有可以合并的对了，就退出
            if pair not in self.merge:
                break

            # 执行合并
            idx = self.merge[pair]
            ids = merge(ids, pair, idx)

        return ids

    def decode(self,idx):
        text_bytes = b"".join(self.vocab[i] for i in idx)
        text = text_bytes.decode("utf-8",errors="replace")

        return text


def test_tokenizer():
    # 创建测试文本
    test_text = """
    This dataset is focused on improving LLM logical reasoning skills and was used to train the Platypus2 models. It is comprised of the following datasets, which were filtered using keyword search and then Sentence Transformers to remove questions with a similarity above 80%
    We've removed approximately 200 questions that appear in the Hugging Face benchmark test sets. Please see our paper and project webpage for additional information."""

    # 初始化tokenizer
    tokenizer = BaseTokenizer()

    # 训练tokenizer (设置一个合理的词汇表大小)
    vocab_size = 1000
    print("开始训练tokenizer...")
    tokenizer.train(vocab_size, test_text)
    print(f"词汇表大小: {len(tokenizer.vocab)}")

    # 测试编码
    print("\n测试编码:")
    test_samples = [
        "This dataset is focused on improving ",
        "LLM logical ",
        "train the Platypus2"
    ]

    for sample in test_samples:
        print(f"\n原文本: {sample}")
        # 编码
        encoded = tokenizer.encode(sample)
        print(f"编码结果: {encoded}")
        # 解码
        decoded = tokenizer.decode(encoded)
        print(f"解码结果: {decoded}")



test_tokenizer()

