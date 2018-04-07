import numpy as np
import pandas as pd
traindata = pd.read_csv('train_data.data')
# print(traindata)
###
#No training
#Used for binary class classification means only 2 type of labels we have
###
def calLabelProbability(dframe):
    #using group by rather than applying **for** loop .
    x=dframe.groupby('label')['label'].count()
    dics={}
    dics[x.index[0]]=x[0]/len(dframe)
    dics[x.index[1]]=x[1]/len(dframe)
    # print(dics)
    return dics
def conditionalprobEachattribute(dframe,adict):
    lbp = calLabelProbability(dframe)
    cp1=0
    cp2=0
    tp=0
    likelihoodc1,likelihoodc2=1,1
    for key in adict:
        cp1,cp2,tp=0,0,0
        for i in range(len(dframe[key])):
            if(dframe[key].iloc[i] == adict[key]):
                tp+=1
                if(dframe['label'].iloc[i]=='yes'):
                    cp1+=1
                else:
                    cp2+=1
        likelihoodc1 *= cp1/tp
        likelihoodc2 *=cp2/tp
    # print(likelihoodc1*lbp['yes'],likelihoodc2*lbp['no'])
    if likelihoodc1> likelihoodc2 :
        return "Yes"
    else:
        return "no"
def predict(dframe,adict):
    return(conditionalprobEachattribute(dframe,adict))
if __name__ == '__main__':
    #dataset for testing
    # testdata = pd.read_csv('test_data.data')
    attributeVals={'outlook':'sunny','temperature':'hot','humidity':'high','wind':'weak'}
    y_pred = predict(traindata,attributeVals)
    print(y_pred)
