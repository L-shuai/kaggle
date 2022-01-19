# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from data import LCQMC_Dataset, load_embeddings, load_vocab
from model import ABCNN
from utils import test
import numpy as np

def main(p_sentence,h_sentence,vocab_file, embeddings_file, pretrained_file, max_length=50):
    print("p_sentence:",p_sentence)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for predicting ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    embeddings = load_embeddings(embeddings_file)
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
        p_list = torch.from_numpy(p_list).to(device)
        h_list = torch.from_numpy(h_list).to(device)
        _, probs = model(p_list, h_list)
        print("probs= ",probs) #probs=  tensor([[0.9928, 0.0072]])
        print("probs= ",type(probs))#probs=  <class 'torch.Tensor'>
        print("probs= ",probs.shape)#probs=  torch.Size([1, 2])
        print("probs= ",probs.numpy())#probs=  [[0.99283546 0.00716446]]
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

if __name__ == "__main__":
    # main("../data/LCQMC_test.csv", "../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")
    p_sentence = '尾号4位多少'
    h_sentence = '尾号是多少后4位'

    # p_sentence = '过年送礼送什么好'
    # h_sentence = '过年前什么时候送礼？'
    main(p_sentence,h_sentence,"../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")