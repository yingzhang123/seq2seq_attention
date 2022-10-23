import jieba
from utils import normalizeString
from utils import cht_to_chs

SOS_token = 0  # 起始符
EOS_token = 1  # 终止符
MAX_LENGTH = 10  # 将长度过长的句子去掉


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}  # 记录词对应的索引
        self.word2count = {}  # 记录每个词的词频
        self.index2word = {
            0: "SOS", 1: "EOS"
        }  # 记录索引到词
        self.n_words = 2  # 记录语料库中有多少种词，初始值为2(起始符+终止符)

    # 对词进行统计
    def addWord(self, word):
        if word not in self.word2index:  # 如果词不在统计表中，添加进统计表
            self.word2index[word] = self.n_words     # 词的索引为该词是第几种的词
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1    # 字典中的词数量+1
        else:                         # 该词在统计表中
            self.word2count[word] += 1

    # 对句子进行分词
    def addSentence(self, sentence):
        for word in sentence.split(" "):     # 将 "你 吃饭 了 吗 ？"  分割为 ["你"，“吃”,“吃饭”," 了"," 吗"] 的list数组
            self.addWord(word)        # 依次将每个词统计


# 文本解析
def readLangs(lang1, lang2, path):
    lines = open(path, encoding='utf-8').readlines()  # 拿到文本的所有行

    lang1_cls = Lang(lang1)
    lang2_cls = Lang(lang2)

    pairs = []  # 记录样本对
    for l in lines:  # 逐行处理
        l = l.split("\t")  # 以Tab分割
        sentence1 = normalizeString(l[0])  # 英文，英文文本处理(大写转小写，过滤非法字符等)
        sentence2 = cht_to_chs(l[1])   # 中文，繁体转简体
        seg_list = jieba.cut(sentence2, cut_all=False)   # 调用结巴分词对中文进行分割，得到分词后的数组
        sentence2 = " ".join(seg_list)   #将中文句子分词后的数组拼接为字符串。join() 方法用于把数组中的所有元素放入一个字符串。元素是通过指定的分隔符进行分隔的。
        # 英文文本是天然分词的，不需要分词                 # 向英文一样，通过空格拼接中文分词结果

        if len(sentence1.split(" ")) > MAX_LENGTH:   # 过滤一些长句,大于10个词的的不统计
            continue     # 忽略当前的一次循环

        if len(sentence2.split(" ")) > MAX_LENGTH:
            continue

        pairs.append([sentence1, sentence2])      # [[“what are you doing?”,"你 在 干 什么"],....]
        lang1_cls.addSentence(sentence1)      # 统计每种语言的词频
        lang2_cls.addSentence(sentence2)

    return lang1_cls, lang2_cls, pairs


# 测试
lang1 = "en"
lang2 = "cn"
path = "../data/cmn.txt"
lang1_cls, lang2_cls, pairs = readLangs(lang1, lang2, path)

print(len(pairs))
print(lang1_cls.n_words)
print(lang1_cls.index2word)

print(lang2_cls.n_words)
print(lang2_cls.index2word)