import numpy as np
import scipy.io as sio
import pandas as pd
import torch

def baseline_remove(datasets):
    data_in = datasets[0:40,0:32,0:8064]
    base_signal = (data_in[0:40,0:32,0:128]+data_in[0:40,0:32,128:256]+data_in[0:40,0:32,256:384])/3
    data = data_in[0:40,0:32,384:8064]
    ### baseline removal
    for i in range(0,60):
        data[0:40,0:32,i*128:(i+1)*128]=data[0:40,0:32,i*128:(i+1)*128]-base_signal
    return data

def label_preprocess(emotion):
    for i in range(0,40):
        if emotion[i]>5:
            emotion[i]=1
        else:
            emotion[i]=0
    return emotion
# output:[2400,32,128]
def DEAP_preprocess(dir,emotion):
    data =sio.loadmat(dir)
    datasets=data['data']
    labels=data['labels']
    data = baseline_remove(datasets)
    labels = labels.transpose(1,0)
    if emotion == "valence":
        label = labels[0]
    elif emotion == "arousal":
        label = labels[1]
    else:
        print("label is not founded")
    label = label_preprocess(label)
    data_eeg=np.zeros([40,60,32,128])
    label_eeg=np.zeros([40,60,1])
    for i in range(0,40):
        for j in range(0,60):
            data_eeg[i][j]=data[i,0:32,i*128:(i+1)*128]
            label_eeg[i][j]=label[i]
    data_eeg = data_eeg.reshape(-1,32,128)
    label_eeg = label_eeg.astype(np.int64).reshape(-1)
    return data_eeg, label_eeg
def adjacency_preprocess():
    dataset = pd.read_table("preprocess/electrode.txt")
    data1 = torch.tensor(dataset['dist'].values)
    data2 = torch.tensor(dataset['arg'].values)
    adj = torch.randn(32,32)
    for i in range(len(data1)):
        for j in range(len(data2)):
            adj[i][j] = torch.where(abs(data1[i]-data1[j])<0.1 and abs(data2[i]-data2[j])<45,1,0)
    return adj