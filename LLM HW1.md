# LLM HW1

## Tokenizer

### BPE算法

BPE是一种数据压缩和子词分割算法，广泛用于训练语言模型的tokenizer。其核心思想是基于字符级别的统计信息，将最常见的字符对（或子词对）合并为更大的子词单元，从而减少词表大小并提高模型的泛化能力。

主要步骤

1. 把所有字符转换成byte，此时vocab size为256
2. 计算byte pair出现的频率
3. 合并最频繁的byte pair，并扩充一个vocab
4. 重复2、3直到vocab size达到预期

### 基于BPE算法训练LLM tokenizer的流程

1. 收集语料，清洗并统一编码
2. 确定目标词表大小
3. BPE进行训练，并保存正向的词表， 和逆向的翻译回去的mapping
4. 生成Tokenizer对象，方便未来的使用

### My Tokenizer

#### Train

```python
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
```

先转换为utf-8编码的byte，此时vocab size为256，则需要做vocab size-256次合并。计算byte pair出现的频率、合并最频繁的byte pair，并扩充一个vocab、重复直到vocab size达到预期。过程中记录下merges和vocab以确定tokenize的encode和decode怎么做。

#### Encode

```python
	def get_stats(ids):
    	counts = {}
    	for pair in zip(ids, ids[1:]):
        	counts[pair] = counts.get(pair, 0) + 1
    		return counts

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
```

利用train中得到的merges对转换成bytes的text进行encode。要注意按照顺序进行encode，因为新增的vocab之间可能存在依赖关系。

#### Decode

```python
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
```

利用train中得到的vocab进行反向的转换成bytes，再从bytes转换成字符。

#### Main

```python
from my_tokenizer import Tokenizer
corpus_path = "./bpe/manual.txt"
with open(corpus_path, 'r') as f:
    text = f.read()
f.close()
tokenizer = Tokenizer()
tokenizer.train(text, 1024)
print(tokenizer.decode(tokenizer.encode(text)))
```

读入和训练，并指定vocab size，利用encode再decode来验证训练的正确性。

##### 两种语言在两种tokenizer上的比较

![image-20241230160939545](/Users/yongqianpeng/Library/Application Support/typora-user-images/image-20241230160939545.png)

可以看到英文在GPT2 encoder上的encode length比较短，在我训练的encoder上encode的length比较长，这是因为我们在训练的过程中，大多数在sentence1中的bytes pattern都没有被我们的encoder见过，自然它在压缩这类的文本的时候，压缩率不高。而GPT2的训练文本经历大量的英文，而且它的vocab size比较大，自然encode出来就会比较短。

![image-20241230161257997](/Users/yongqianpeng/Library/Application Support/typora-user-images/image-20241230161257997.png)

可以看到，在encode中文文本的时候结论反过来了，即使GPT2有很大的vocab size，但我的encoder因为在特殊的文本上train过，更贴近sentence2的分布，所以encode length会更短。（见下图，有sentence2的子串在训练文本中出现了）

![image-20241230161459520](/Users/yongqianpeng/Library/Application Support/typora-user-images/image-20241230161459520.png)



### 回答问题

1. 查看字符的Unicode：ord() 函数。将Unicode转换为字符：chr() 函数。
   ![image-20241230162211100](/Users/yongqianpeng/Library/Application Support/typora-user-images/image-20241230162211100.png)

2. **Tokenizer的vocab size大小的好处和坏处**
   **大词汇表：**

   ​	优点：更能捕捉细粒度语义，减少分词后的碎片。

   ​	缺点：占用更多内存和存储空间，训练和推理速度较慢。

   **小词汇表：**

   ​	优点：效率更高，适合资源受限环境。

   ​	缺点：可能导致过多分词，语义表达不够精确。

3.  **为什么 LLM 不能处理非常简单的字符串操作任务，比如反转字符串？**
   tokenizer会将文本分割为子词或字节级别的单元，而不是单字符。这导致模型对输入的理解是基于这些token，而非逐字逐符。而且语言模型的设计目标是基于语义和上下文预测，而不是直接处理低层次的操作，如逐字符的字符串修改。

4.  **为什么 LLM 在非英语语言（例如日语）上表现较差？**

   训练语料中英语占主导地位，非英语语言（如日语）数据较少，语料覆盖和多样性不足。此外，非英语语言可能有更复杂的语法结构或多音文字（如汉字）。从tokenization的角度讲，vocab是基于大量英语语料库构建的，对其他语言的覆盖度较低。这导致非英语语言的token化后，单个字符或分词单位可能占据多个token，导致信息表达变得冗长。

5. **为什么 LLM 在简单算术问题上表现不好？**
   数字的Tokenization通常按字符分割（如123被拆分为"1", "2", "3"），而非整体处理。算术问题要求Token的数值含义，而现有的LLM对数字Token的处理仅基于概率上下文，忽视了数字的符号特性。Tokenizer对符号操作（如+, -, *）的分词方式也可能不利于算术推理。

6. **为什么 GPT-2 在编写 Python 代码时遇到比预期更多的困难？**

   代码中的特殊符号（如缩进、括号等）在Tokenization时会被分成独立Token，使得代码的逻辑结构被破坏小词汇表可能。导致常见代码片段（如for i in range）被拆分成多个Token，难以捕捉全局模式。对于大词汇表，Tokenizer可能覆盖不常见的代码关键字，模型无法泛化。

7.  **为什么 LLM 遇到字符串 “<|endoftext|>” 时会突然中断？**
   Tokenizer将<|endoftext|>作为特殊Token，明确表示结束符号。在生成任务中，一旦预测到该Token，解码器会立即停止输出，这是Tokenizer约定好的语义。

8. **为什么当问 LLM 关于 “SolidGoldMagikarp” 的问题时 LLM 会崩溃？**
   在tokenization train set中存在一个 u/SolidGoldMagikarp，而在training set可能不会出现这个token，那么在训练过程中这个token就未经训练，就会引发奇怪的行为。

9. **为什么在使用 LLM 时应该更倾向于使用 YAML 而不是 JSON？**

   JSON 中的双引号 " 和大括号 {} 在分词时通常会单独作为 Token 被模型处理，而 YAML 省略了这些符号，因此YAML 具有更紧凑的语法。例如，在 YAML 中无需像 JSON 那样反复使用大括号 {}、引号 "" 或逗号 , 来分隔结构。

10. **为什么 LLM 实际上不是端到端的语言建模？**

    自然语言在进入模型前必须经过 tokenization 处理，这一过程将连续的语言encode为离散的 token 序列最终还需要decode出来，而模型实际建模的对象是这些 tokens 而非原始语言本身。另外BPE的离散化可能会导致语义信息部分丢失。



## LLM Implementation

![image-20241230201729569](/Users/yongqianpeng/Library/Application Support/typora-user-images/image-20241230201729569.png)

![image-20241230201749622](/Users/yongqianpeng/Library/Application Support/typora-user-images/image-20241230201749622.png)

![image-20241230201805004](/Users/yongqianpeng/Library/Application Support/typora-user-images/image-20241230201805004.png)

![image-20241230201815421](/Users/yongqianpeng/Library/Application Support/typora-user-images/image-20241230201815421.png)

### section 1

首先搭出了GPT2的框架，然后可以注意一下把所有的参数的名字和hugging face上的GPT2的参数都设置成一样的，方便我们可以load它的参数进行验证。然后搭建完之后把forward一写，就可以写一个from pretrain的函数去把huggingface上的GPT2参数load一下验证一下，验证下来发现也没什么问题。

之后拿一个小的dataset Shakespeare去写训练的代码。先去写一个简单的dataloader把数据load进来。设置一下loss function和optimizer，再跟着paper去做参数的初始化，之后就可以做训练了。

### section 2

第二部分是去给训练做加速。首先是把精度调一下，调成TF32。之后加上model.compile()。再利用上flash attention的方法，之后再去把vocab size调成一个2的幂的magic number。加速效果如下，GPU为3090：

default
	Step 10, loss: 9.794076919555664, dt: 532.956362ms tokens/sec: 15370.8644602419
TF32
	Step 10, loss: 9.794072151184082, dt: 403.121948ms tokens/sec: 20321.394148150954
BF16
	Step 10, loss: 9.794027328491211, dt: 299.741745ms tokens/sec: 27330.19386449975
compile
	Step 10, loss: 9.794479370117188, dt: 179.153919ms tokens/sec: 45726.04403893397
Flash attention
	Step 10, loss: 9.79465103149414, dt: 141.292334ms tokens/sec: 57979.08344428077
vocab size
	Step 10, loss: 9.899351119995117, dt: 137.336016ms tokens/sec: 59649.32037796708

### Section 3、4

去进一步follow paper里的训练参数，增加clip loss让训练更稳定一点，加入scheduler让learning rate随着训练逐渐降低。

接着去利用上多个gpu，具体地要使用ddp这个库，去告诉程序我目前的gpu世界是怎么样的，并给每一个进程打上标识目前使用的是哪一个gpu。有了这些标识后，去修改data'loader，把这个标识传进去，确保每个gpu load的数据不一样。

最后加上hellswag，在一定数量的训练后验证一下我这个模型的训练怎么样。

## Lora Fine-tuning





