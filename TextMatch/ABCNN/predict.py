# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:30:07 2020

@author: zhaog
"""
import torch
from sys import platform
from torch.utils.data import DataLoader
from data import LCQMC_Dataset, load_embeddings, load_vocab, pad_sequences
from model import ABCNN
from utils import test

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
    h = [word2idx[word] for word in h_sentence if word in word2idx.keys()]
    p_list = pad_sequences(p, maxlen=max_length)
    h_list = pad_sequences(h, maxlen=max_length)

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
        p_list = p_list.to(device)
        h_list = h_list.to(device)
        _, probs = model(p_list, h_list)
        print("probs= ",probs)

if __name__ == "__main__":
    # main("../data/LCQMC_test.csv", "../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")
    p_sentence = '我今天去超市了'
    h_sentence = '最近的超市在哪里？'
    main(p_sentence,h_sentence,"../data/vocab.txt", "../data/token_vec_300.bin", "models/best.pth.tar")