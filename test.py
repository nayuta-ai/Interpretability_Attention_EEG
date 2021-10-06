import numpy as np
import scipy.io as sio
from scipy.fftpack import fft,ifft
from preprocess.preprocessed import DEAP_preprocess,adjacency_preprocess
from preprocess.feature import feature_extract

dataset_dir = "../data_preprocessed_matlab/s01.mat"
emotion = "valence"
data,label = DEAP_preprocess(dataset_dir,emotion)
param = {'stftn':128,'fStart':[4,8,14,31],'fEnd':[7,13,30,50],'window':1,'fs':128}
de = feature_extract(data,param)
data_fix = np.zeros([800,3,32,4])
label_fix = np.zeros(800)
for i in range(800):
    data_fix[i]=de[3*i:3*(i+1)]
    label_fix[i] = label[3*i]
print(data_fix.shape) # [800,3,32,4]
print(label_fix.shape) # [800,]
