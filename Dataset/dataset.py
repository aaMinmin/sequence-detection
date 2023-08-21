import numpy as np
import pickle
import os
import csv

def read(root, filename):  # root: C:\Users\Administrator\Desktop\DNA\DNAScoring

    seqs, scores = [],[]
    with open(os.path.join(root,filename), 'r') as f3:
        reader = csv.reader(f3)
        for i in reader:
            score, seq = i[0], i[1]
            scores.append(score)
            seqs.append(seq)
    return seqs, scores

def load_DNAScore(root, mode):
    scores3 = []
    seqs3 =[]
    # reverse_seqs=[]
    seqs, scores = read(root, 'length110.csv')  # sefold.csv
    # for j in range(0,len(seqs)):
        # reverse_seqs.append("".join(reversed(seqs[j])))
    for i in range(0, len(scores)):
        x = scores[i]
        x = float(x)
        scores3.append(x)

    if mode == 'train':  # 60%
        seqs3 = seqs[:int(0.95 * len(seqs))]
        scores3 = scores3[:int(0.95 * len(seqs))]

    elif mode == 'test':  # 20% = 80%->100%
        seqs3 = seqs[int(0.95 * len(seqs)):]
        scores3 = scores3[int(0.95 * len(seqs)):]
    return seqs3, scores3

def base_to_vec(seq, pos):
    if seq[pos] == 'A':
        return [1,0,0,0]
    if seq[pos] == 'G':
        return [0,1,0,0]
    if seq[pos] == 'C':
        return [0,0,1,0]
    if seq[pos] == 'T':
        return [0,0,0,1]
    else:
        return [0,0,0,0]

def embedding(seqs):
    data = []
    for seq in seqs:
        data_seq = []
        for pos in range(0,len(seq)):
            data_seq.append(base_to_vec(seq, pos))
        data.append(data_seq)
    return data


if __name__ == '__main__':
    path_ = "./"
    seqs_train2, scores_train = load_DNAScore(path_, mode='train')
    # seqs_train3, scores_train3 = load_DNAScore(path_, mode='train')
    seqs_test2, scores_test = load_DNAScore(path_, mode='test')
    seqs_train = embedding(seqs_train2)
    seqs_test = embedding(seqs_test2)
    # seqs_train33 = embedding(seqs_train3)
    # 当前文件所在路径

    #Train Test
    with open(os.path.join(path_, 'mysavedata/traindata.pkl'), 'wb+') as f:
        pickle.dump(seqs_train, f)
    with open(os.path.join(path_, 'mysavedata/trainscore.pkl'), 'wb+') as f1:
        pickle.dump(scores_train, f1)
    with open(os.path.join(path_, 'mysavedata/trainseq.pkl'), 'wb+') as f2:
        pickle.dump(seqs_train2, f2)#序列
    with open(os.path.join(path_, 'mysavedata/testdata.pkl'), 'wb+') as f3:
        pickle.dump(seqs_test, f3)#嵌入后的数据
    with open(os.path.join(path_, 'mysavedata/testscore.pkl'), 'wb+') as f4:
        pickle.dump(scores_test, f4)#自由能分数
    with open(os.path.join(path_, 'mysavedata/testseq.pkl'), 'wb+') as f5:
        pickle.dump(seqs_test2, f5)#序列
    # with open(os.path.join(path_, 'mysavedata/Train_data2.pkl'), 'wb+') as f6:
    #     pickle.dump(seqs_train33, f6)
    # with open(os.path.join(path_, 'mysavedata/Train_score2.pkl'), 'wb+') as f7:
    #     pickle.dump(scores_train3, f7)
    # with open(os.path.join(path_, 'mysavedata/Train_seq2.pkl'), 'wb+') as f8:
    #     pickle.dump(scores_train3, f8)

