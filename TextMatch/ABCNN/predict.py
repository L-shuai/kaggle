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

def main2(test_file, vocab_file, embeddings_file, pretrained_file, max_length=50, gpu_index=0, batch_size=128):

    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    # Retrieving model parameters from checkpoint.
    embeddings = load_embeddings(embeddings_file)
    print("\t* Loading test data...")    
    test_data = LCQMC_Dataset(test_file, vocab_file, max_length)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    print("\t* Building model...")
    model = ABCNN(embeddings, device=device).to(device)
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing ABCNN model on device: {} ".format(device), 20 * "=")
    batch_time, total_time, accuracy, auc = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))

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
        print("probs= ",probs)
        print("probs= ",type(probs))
        print("probs= ",probs.shape)
        print("probs= ",probs.numpy())
        _, out_classes = probs.max(dim=1)
        print("out_classes:",out_classes.shape)
        print("out_classes:",out_classes)
        print("out_classes:",type(out_classes))
        print("out_classes:",out_classes.item())


def pad_sequences1(sequences, maxlen=None, dtype='int32', padding='post',
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
    # lengths = [len(s) for s in sequences]
    lengths = 1
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = len(sequences)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    # for idx, s in enumerate(sequences):
    if len(sequences) == 0:
        return x # empty list was found
    if truncating == 'pre':
        trunc = sequences[-maxlen:]
    elif truncating == 'post':
        trunc = sequences[:maxlen]
    else:
        raise ValueError("Truncating type '%s' not understood" % padding)
    if padding == 'post':
        x[0, :len(trunc)] = trunc
    elif padding == 'pre':
        x[0, -len(trunc):] = trunc
    else:
        raise ValueError("Padding type '%s' not understood" % padding)
    return x

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
    # p_sentence = '尾号4位多少'
    # h_sentence = '尾号是多少后4位'

    p_sentence = '过年送礼送什么好'
    h_sentence = '过年前什么时候送礼？'
    main(p_sentence,h_sentence,"../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")