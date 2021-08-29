import numpy as np
import scipy.io as sio
from scipy.fftpack import fft,ifft
from preprocess.preprocessed import DEAP_preprocess
from preprocess.Feature_Extraction import feature_extract

dataset_dir = "../data_preprocessed_matlab/s01.mat"
emotion = "valence"
data,label = DEAP_preprocess(dataset_dir,emotion)
param = {'stftn':128,'fStart':[4,8,14,31],'fEnd':[7,13,30,50],'window':1,'fs':128}
de = feature_extract(data,param)
# reshape re2:[40,20,3,32,4]
re1 = np.zeros([800,3,32,4])
for i in range(800):
    re1[i]=de[3*i:3*(i+1)]
re2 = np.zeros([40,20,3,32,4])
for i in range(40):
    re2[i]=re1[20*i:20*(i+1)]
print(re2.shape)