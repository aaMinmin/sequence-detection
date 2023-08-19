import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from os import path
#from Mfold import seqfold

DNA_base = ['A', 'G', 'C', 'T']

DNA_Sequence = []

DIR = path.dirname(path.realpath(__file__))

i=110
with open(path.join(DIR,'new_generate' + str(i) + '.csv'), 'w', newline = '') as file_output:

    n = 0
    while n < 1000000:
        seq = ""
        GC_num = 0
        seq_len = i
        j = 0
        while j < seq_len:
            Nth = random.randint(0,3)
            if Nth == 1 or Nth == 2:
                GC_num += 1
            seq += DNA_base[Nth]
            j += 1
            #限制多聚物
            if len(seq) >= 4 and len(set(seq[-4:])) == 1:#set(seq[-4:]) 最后四个是由多少种元素组成 ； ==1 就是只有1种 也就是有均聚物
                seq = seq[:-1] #去掉序列最后一个 使其不形成均聚物
                j -= 1

        #限制GC含量45%-55%
        if float(GC_num/seq_len) >= 0.45 and float(GC_num/seq_len) <= 0.55:
            file_output.write(seq + "\n")
            n += 1
            DNA_Sequence.append(seq)
