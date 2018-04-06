import pandas as pd
import math
import numpy as np
df = pd.read_csv('train_data.data')
#FIND NO OF DIFFERENT LABELS

#find count of each labl
def entropy(dframe):
    l = set(dframe['label'])
    dict={}
    if(len(l)!=1):
        for label in l:
            c=0
            for i in dframe['label']:
                if i==label:
                    c+=1
            dict[label] = c
    #find measure of uncertanity in dataset
    hsl = [-dict[key]/len(dframe)*math.log2(dict[key]/len(dframe))for key in dict]
    #find the information gain of each attribute
    hs=sum(hsl)
    return hs
    #Information gain for each attribute
def remainingAttributes(dframe):
    colums = dframe.columns
    colums = colums[:len(colums)-1]
    return colums
#attribute 1
infG=0
def calattribute(col,dframe):
    a1 = set(dframe[col])
    lpp = set(dframe['label'])
    d1 = {}
    ct=0
    badidict={}
    if (len(lpp) != 1):
        for lkam in a1:
            c = 0
            for i in dframe[col]:
                if i == lkam:
                    c += 1
            d1[lkam] = c
        # print(d1)
        l = dframe['label']
        lablcount={}
        for lkam in a1:
            lablcount={}
            for label in l:
                ct=0

                for i in range(len(dframe[col])):
                    if dframe[col].iloc[i] == lkam and dframe['label'].iloc[i] == label:
                        ct+=1
                lablcount[label] = ct
            badidict[lkam]=lablcount
    # find measure of uncertanity in dataset
    # print(badidict)
    else:
        return lpp.pop()
    hsl = [d1[key] / len(dframe) for key in d1]
    s1=0
    le = 0
    # find the information gain of each attribute
    for key in badidict:
        le = 0
        for chotikey in badidict[key]:
            if(badidict[key][chotikey]!=0):
                le += (-badidict[key][chotikey]/d1[key]*math.log2(badidict[key][chotikey]/d1[key]))
        s1 += le*d1[key]/len(dframe)

    return s1
def findMaxGainAttribute(dframe,colums,hs):
    max=-9999
    attribute=0
    for cols in colums:
        p=calattribute(cols,dframe)
        if(p=="yes" or p=="no"):
            attribute=p
            break
        p=hs-p
        if(p>max):
            max=p
            attribute = cols
    return attribute
def createDecisionTree(dframe,dtree):
    cols=remainingAttributes(dframe)
    prob = entropy(dframe)
    asd = (findMaxGainAttribute(dframe,cols,prob))
    timepaas = {}

    if(asd=="yes" or asd=="no" ):
        dtree.append(asd)
        return asd
    values = set(dframe[asd])
    for pp in values:
        d1=dframe[dframe[asd]==pp]
        del d1[asd]

        timepaas[pp]=d1
        dtree.append(asd+pp)
        createDecisionTree(timepaas[pp],dtree)
    # print(timepaas)
    # print("\n\n")
    return dtree
if __name__ == '__main__':
    dtree=[]
    createDecisionTree(df,dtree)
    print(dtree)
