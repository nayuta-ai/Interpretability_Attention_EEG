import scipy.io as sio
import torch
data =sio.loadmat("../../data_preprocessed_matlab/s01.mat")
datasets=data['data'][:, 0:32, 3 * 128:]
original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6',
                'CP2', 'P4', 'P8', 'PO4', 'O2']

adjacency_matrix

print(len(original_order))
