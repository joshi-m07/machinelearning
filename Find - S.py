import pandas as pd
import numpy as np

data=pd.read_csv('ws.csv',header=None)

concepts = np.array(data)[:,:-1]
target = np.array(data)[:,-1]

def train(c,t):
    for i,val in enumerate(t):
        if val=='Yes':
            hyp = c[i].copy()
            break
    for i ,val in enumerate(c):
        if t[i]=='Yes':
            for x in range(len(hyp)):
                if val[x]!=hyp[x]:
                    hyp[x]='?'
                else:
                    pass
    return hyp
print(train(concepts,target))