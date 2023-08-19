import os
import csv
import random

import numpy as np
q=[]
for index in range(50,150,5):
    j=0
    with open(os.path.join('./','together/seq'+str(index)+'.csv'),'r',encoding='utf-8-sig') as f:
       reader = csv.reader(f)
       for i in reader:
            if j<=15000:
                score,seq=i[0],i[1]
                q.append([score]+[seq])
                j+=1
            else:
                j+= 1
            # score=i[0]+'\n'
            # q.append(score)
    print(index)
random.shuffle(q)
with open(os.path.join('./','Testdata1w5k.csv'),'w',newline='') as f3:
    writer = csv.writer(f3)
    for item in q:
        writer.writerow(item)