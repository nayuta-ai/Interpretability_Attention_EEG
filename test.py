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
print(de.shape)