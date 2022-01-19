# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from data import LCQMC_Dataset, load_embeddings, load_vocab, get_word_list
from model import ABCNN
from utils import test
import numpy as np
import pandas as pd

def main(p_sentence,h_sentence,vocab_file, embeddings_file, pretrained_file, max_length=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for predicting ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    embeddings = load_embeddings(embeddings_file)

    print("p_sentence:",p_sentence)

    # print(20 * "=", " Preparing for predicting ", 20 * "=")
    # if platform == "linux" or platform == "linux2":
    #     checkpoint = torch.load(pretrained_file)
    # else:
    #     checkpoint = torch.load(pretrained_file, map_location=device)
    # # Retrieving model parameters from checkpoint.
    # embeddings = load_embeddings(embeddings_file)
    print("\t* Process predict data...")
    word2idx, _, _ = load_vocab(vocab_file)
    p = [word2idx[word] for word in p_sentence if word in word2idx.keys()]
    print("p:",p)
    h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
    p_list = pad_sequences(p, maxlen=max_length)
    h_list = pad_sequences(h, maxlen=max_length)
    print("p_list:",p_list.shape)
    print("h_list:",h_list.shape)

    model = ABCNN(embeddings, device=device).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    device = model.device
    print(20 * "=", " Predicting ABCNN model on device: {} ".format(device), 20 * "=")
    # batch_time, total_time, accuracy, auc = test(model, test_loader)
    # print(
    #     "\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(
    #         batch_time, total_time, (accuracy * 100), auc))

    with torch.no_grad():
        # p_list = torch.from_numpy(p_list).to(device)
        # h_list = torch.from_numpy(h_list).to(device)
        p_list = torch.tensor(p_list,dtype=torch.long).to(device)
        h_list = torch.tensor(h_list,dtype=torch.long).to(device)
        print("h_list.type:",type(h_list))
        _, probs = model(p_list, h_list)
        print("probs= ",probs) #probs=  tensor([[0.9928, 0.0072]])
        print("probs= ",type(probs))#probs=  <class 'torch.Tensor'>
        print("probs= ",probs.shape)#probs=  torch.Size([1, 2])
        # print("probs= ",probs.numpy())#probs=  [[0.99283546 0.00716446]]
        _, out_classes = probs.max(dim=1)
        print("out_classes:",out_classes.shape) #out_classes: torch.Size([1])
        print("out_classes:",out_classes) #out_classes: tensor([0])
        print("out_classes:",type(out_classes)) #out_classes: <class 'torch.Tensor'>
        print("out_classes:",out_classes.item())  #输出正确  保留 out_classes: 0



def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences
    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。
    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值
    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)
    """
    sequences = [sequences]
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def isSimilar(sentence,lib_list,vocab_file, embeddings_file, pretrained_file, max_length=50):
    """

    :param sentence: 待比较的句子
    :param lib_list: 句子库
    :param vocab_file: word2idx
    :param embeddings_file: embedding模型所在路径
    :param pretrained_file: ABCNN模型路径
    :param max_length: 句子最大长度  截断
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for isSimilar ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    embeddings = load_embeddings(embeddings_file)

    model = ABCNN(embeddings, device=device,max_length=max_length).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    device = model.device

    # 遍历现有句子库
    for sen in lib_list:
        print("sen:", sen)
        print("\t* Process predict data...")
        word2idx, _, _ = load_vocab(vocab_file)
        p = [word2idx[word] for word in sentence if word in word2idx.keys()]
        print("正在对比的sen:", sen)
        h = [word2idx[word] for word in sen if word in word2idx.keys()]
        p_list = pad_sequences(p, maxlen=max_length)
        h_list = pad_sequences(h, maxlen=max_length)
        print("p_list:", p_list.shape)
        print("h_list:", h_list.shape)

        # model = ABCNN(embeddings, device=device).to(device)
        # model.load_state_dict(checkpoint["model"])
        # model.eval()
        # device = model.device
        print(20 * "=", " Predicting ABCNN model on device: {} ".format(device), 20 * "=")


        with torch.no_grad():
            # p_list = torch.from_numpy(p_list).to(device)
            # h_list = torch.from_numpy(h_list).to(device)
            p_list = torch.tensor(p_list, dtype=torch.long).to(device)
            h_list = torch.tensor(h_list, dtype=torch.long).to(device)
            # print("h_list.type:", type(h_list))
            _, probs = model(p_list, h_list)
            # print("probs= ", probs)  # probs=  tensor([[0.9928, 0.0072]])
            # print("probs= ", type(probs))  # probs=  <class 'torch.Tensor'>
            # print("probs= ", probs.shape)  # probs=  torch.Size([1, 2])
            # print("probs= ",probs.numpy())#probs=  [[0.99283546 0.00716446]]
            _, out_classes = probs.max(dim=1)
            # print("out_classes:", out_classes.shape)  # out_classes: torch.Size([1])
            # print("out_classes:", out_classes)  # out_classes: tensor([0])
            # print("out_classes:", type(out_classes))  # out_classes: <class 'torch.Tensor'>
            print("isSimilar:", out_classes.item())  # 输出正确  保留 out_classes: 0

def testIsSimilar(sentence,vocab_file, embeddings_file, pretrained_file, max_length=50):
    df = pd.read_csv("../data/LCQMC_test.csv", delimiter="\t")
    df.columns = ['sentence1', 'sentence2']
    p = map(get_word_list, df['sentence1'].values[0:max_length])
    # q = map(get_word_list, df['sentence2'].values[0:max_length])
    print("type(df):",type(df))
    print("df.shape:",df.shape)
    lib_list = df['sentence1'].values.tolist()
    print("type(lib_list):",type(lib_list))
    isSimilar(sentence,lib_list,"../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")


if __name__ == "__main__":
    # main("../data/LCQMC_test.csv", "../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")
    p_sentence = '尾号4位多少'
    h_sentence = '尾号是多少后4位'

    # p_sentence = '过年送礼送什么好'
    # h_sentence = '过年前什么时候送礼？'
    p_sentence = '2021年群众斗殴事件造成3人死亡，6人重伤，直接经济损失200万'
    h_sentence = '提醒大家不要聚众斗殴'
    # main(p_sentence,h_sentence,"../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")
    lib_list = ['送自己做的闺蜜什么生日礼物好','据极目新闻此前报道，2021年12月6日，河北邢台17岁少年刘学州在网上寻亲，二十多天后，他终于见到了亲生父母。但1月17日，刘学州称目前已被亲生母亲拉黑，希望得到法律援助。18日中午，刘学州又称决定暂时放弃民事诉讼。','闺蜜要过生日了','闺蜜送我什么礼物好']
    sentence = '19日凌晨，寻亲男孩刘学州发文称，本考虑放弃诉讼，但因亲生父母“颠倒黑白”，决定起诉，要与亲生父母“法庭见”。此前，他于12月16日在网上寻找父母，后得与亲生父母见面，但因其向父母要房子一事被生母拉黑。律师表示，刘学州亲生父母或涉嫌遗弃罪，刘学州向他们要房子不合理但可以理解。'
    isSimilar(sentence,lib_list,"../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar",max_length=50)#这里max_length只能设为50，因为预先训练好的模型参数就是50，如果要更改需要重新训练
    # testIsSimilar(sentence,"../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")