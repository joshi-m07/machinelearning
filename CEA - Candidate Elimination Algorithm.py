import numpy as np
import pandas as pd

data = pd.read_csv('ws.csv')

concepts=np.array(data)[:,:-1]
target=np.array(data)[:,-1]



def learn(concepts,target):
    specific_h = concepts[0].copy()
    print('Initialize specific_h and general_h')
    print("Specific boundary:" ,specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print("Generic boundary:", general_h)

    for i,h in enumerate(concepts):
        print("\nInstance",i+1,"is",h)
        if target[i]=='Yes':
            print("Positive instance")
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x]='?'
                    general_h[x][x]='?'
        if target[i]=='No':
            print('Negative instance')
            for x in range(len(specific_h)):
                if h[x]!= specific_h[x]:
                    general_h[x][x]=specific_h[x]
                else:
                    general_h[x][x]='?'
        
        print("specific after",i+1,"instance is", specific_h)
        print("general after",i+1,"instance is", general_h)
        print('\n')
    indices = [i for i,val in enumerate(general_h) if val==['?','?','?','?','?','?']]
    for i in indices:
        general_h.remove(['?','?','?','?','?','?'])
    return specific_h,general_h

s_final, g_final=learn(concepts,target)
print("FINAL SPECIFIC",s_final)
print("FINAL GENERIC", g_final)