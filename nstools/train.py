import random
import time

import torch
import torch.nn as nn
from torch import optim
from datasets import readLangs, SOS_token, EOS_token, MAX_LENGTH
from models import EncoderRNN, AttenDecoderRNN
from utils import timeSince

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH += 1   # 添加了终止符,比dataset中的的最大长度多1，因为要加入终止符

# 本任务完成英文到中文的翻译。若要倒过来，则要修改lang1和lang2的位置，还有pairs中的中英文词样本对的位置
lang1 = "en"
lang2 = "cn"
path = "../data/cmn.txt"

input_lang, output_lang, pairs = readLangs(lang1, lang2, path)
# print(len(pairs))
# print(input_lang.n_words)
# print(input_lang.index2word)
# print(output_lang.n_words)
# print(output_lang.index2word)

def listTotensor(input_lang, data):
    indexes_in = [input_lang.word2index[word] for word in data.split(" ")]  #得到句子所对应的索引列表[3,6,3,...]，经过embedding层，变为二维向量
    indexes_in.append(EOS_token)              # 在最后加入终止符,所以要比dataset中得MAX_LENGTH大1
    input_tensor = torch.tensor(indexes_in,
                                dtype=torch.long,
                                device=device).view(-1, 1)
    return input_tensor       # 转换为张量并输出

#把pairs下的序列转换为输入tensor，并在tensor中插入一个终止符
# 将一个样本对转化为tensor
def tensorsFromPair(pair):
    input_tensor = listTotensor(input_lang, pair[0])     # 将样本对前半部分英文转化为索引列表
    output_tensor = listTotensor(output_lang, pair[1])     # 将样本对后半部分中文转化为索引列表
    return (input_tensor, output_tensor)

# 计算loss
def loss_func(input_tensor, output_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,criterion):
    encoder_hidden = encoder.initHidden()  #初始化隐藏层

    encoder_optimizer.zero_grad()  #优化器梯度置零
    decoder_optimizer.zero_grad()

    input_len = input_tensor.size(0)   # 输入输出长度，input_tensor,output_tensor均为二维张量。# 一句话的长度，
    output_len = output_tensor.size(0)   # input_tensor.size(1):为一个词的表示维度(embedding层的输出大小)

    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)  # encoder的输出

    #每次从input_tensor中取一个出来利用隐藏层信息进行encoder
    for ei in range(input_len):            # 将一个一句话的每个词依次编码
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]  #编码结果， # encoder_output为3维的向量
        # encoder_outputs为一个句子的编码结果，为二维张量[[],[]...]

    # 定义解码器
    decoder_hidden = encoder_hidden
    decoder_input = torch.tensor([[SOS_token]], device=device)  #第一个解码输入定义为起始符SOS_token

    # 加入随机因子，随机修改当前隐藏层的输入为真实的label，让模型收敛更快
    use_teacher_forcing = True if random.random() < 0.5 else False

    loss = 0    #loss初始化为0
    if use_teacher_forcing:          # 满足条件，使用
        for di in range(output_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )                                                    # encoder_outputs:要解码的内容
            loss += criterion(decoder_output, output_tensor[di])   # 计算loss, output_tensor:期待的输出(也就是label)

            decoder_input = output_tensor[di]   #下一次循环的输入直接定义为真实的label
    else:
        for di in range(output_len):         # 不满足条件
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, output_tensor[di])

            # 定义下一次的输入为当前的预测结果
            topV, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            # 判断解码是否结束
            if decoder_input.item() == EOS_token:        # 等于终止符，解码结束
                break

    loss.backward()  #梯度传播
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / output_len

######
# 定义网络
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttenDecoderRNN(hidden_size, output_lang.n_words,
                          max_len = MAX_LENGTH,
                          dropout_p=0.1).to(device)

lr = 0.01
encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)     # 编码器优化器
decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)     # 解码器优化器


#设置学习率调整  # 学习率的调整策略
scheduler_encoder = torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                    step_size=1,
                                                    gamma=0.95)
scheduler_decoder = torch.optim.lr_scheduler.StepLR(decoder_optimizer,
                                                    step_size=1,
                                                    gamma=0.95)
# 定义损失函数
criterion = nn.NLLLoss()

# 不使用dataset,dataloader
# 直接生成样本对训练
n_iters = 10000      # 最大迭代次数
training_pairs = [
    tensorsFromPair(random.choice(pairs)) for i in range(n_iters)   # 挑选1000000个样本对
]

print_every = 1000  # 每迭代1000词打印一次信息
save_every = 10000

print_loss_total = 0
start = time.time()

for iter in range(1, n_iters+1):
    training_pair = training_pairs[iter - 1]
    input_tensor = training_pair[0]
    output_tensor = training_pair[1]

    loss = loss_func(input_tensor,
                     output_tensor,
                     encoder,
                     decoder,
                     encoder_optimizer,
                     decoder_optimizer,
                     criterion)
    print_loss_total += loss

    if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print("{},{},{},{}".format(timeSince(start, iter/n_iters),
                                   iter, iter / n_iters * 100,
                                   print_loss_avg))

    #保存模型
    if iter % save_every == 0:
        torch.save(encoder.state_dict(),
                   "../models/encoder_{}.pth".format(iter))
        torch.save(decoder.state_dict(),
                   "../models/decoder_{}.pth".format(iter))

    #更新学习率
    if iter % 1000:
        scheduler_encoder.step()
        scheduler_decoder.step()

