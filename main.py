import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

from graph import *
from model import SpaceFrequencyTemporalAttention
from module.frequency_attention import FrequencyAttentionLayer
from module.graph_attention import GraphAttentionLayer
from module.temporal_attention import TemporalAttentionLayer,TransformerBlock,PositionalEncoder
from preprocess.preprocessed import DEAP_preprocess,adjacency_preprocess
from preprocess.feature import feature_extract
from torch.autograd import Variable
import torch.optim as optim
def fix_seed(seed):
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def all_preprocess(data,emotion):
    dataset_dir = "../data_preprocessed_matlab/"+data+".mat"
    data, label = DEAP_preprocess(dataset_dir,emotion)
    param = {'stftn':128,'fStart':[4,8,14,31],'fEnd':[7,13,30,50],'window':1,'fs':128}
    de = feature_extract(data,param)
    data_fix = np.zeros([800,3,32,4])
    label_fix = np.zeros(800)
    for i in range(800):
        data_fix[i]=de[3*i:3*(i+1)]
        label_fix[i] = label[3*i]
    # data_fix: [800,3,32,4]
    # label_fix: [800,]
    return data_fix, label_fix
"""
dataset_dir = "../../data_preprocessed_matlab/s01.mat"
emotion = "valence"
a,b = all_preprocess(dataset_dir,emotion)
print(a.shape)
"""

if __name__ == '__main__':
    # random state
    SEED = 42
    fix_seed(SEED)
    # parameter
    ## training
    training_epochs = 10000
    batch_size = 10
    emotion = "arousal"
    # deap_subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11','s12', 's13','s14','s15', 's16', 's17','s18', 's19', 's20','s21', 's22', 's23', 's24', 's25', 's26','s27', 's28', 's29', 's30', 's31', 's32']
    deap_subjects = ['s01']
    ## loss
    learning_rate = 1e-8
    # gpu setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # network
    freq = 4
    FA = FrequencyAttentionLayer(freq)
    adj = adjacency_preprocess(device=device)
    GA = GraphAttentionLayer(in_features=batch_size*3,out_features=batch_size*3,alpha=0.2,dropout=0.5,adj=adj)
    tra = TransformerBlock()
    pos = PositionalEncoder()
    TA = TemporalAttentionLayer(pos,tra)
    model = SpaceFrequencyTemporalAttention(device,FA,GA,TA)
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for list in deap_subjects:
        # dataset load
        datasets, labels = all_preprocess(list,emotion)
        datasets = torch.from_numpy(datasets).float()
        labels = torch.from_numpy(labels).long()
        fold = 10
        test_accuracy_all_fold = np.zeros(shape=[0],dtype=float)
        total_train_acc=[]
        total_train_loss=[]
        total_test_acc=[]
        total_test_loss=[]
        # fold '0-9'
        for curr_fold in range(fold):
            fold_size = datasets.shape[0]//fold
            indexes_list = [i for i in range(len(datasets))]
            indexes = np.array(indexes_list)
            split_list = [i for i in range(curr_fold*fold_size,(curr_fold+1)*fold_size)]
            split = np.array(split_list)
            test_y = labels[split]
            test_x = datasets[split]
            split_set = set(indexes_list)^set(split_list)
            split = [x for x in split_set]
            split = np.array(split)
            train_x = datasets[split]
            train_y = labels[split]

            # set train batch number per epoch
            batch_num_epoch_train = train_x.shape[0]//batch_size
            batch_num_epoch_test = test_x.shape[0]//batch_size
            train_acc = []
            test_acc = []
            train_loss = []
            test_loss = []
            # training_epochs
            for epoch in range(training_epochs):
                loss = None
                """
                correct_train = 0
                loss_train_total = 0
                batch_train_loss = []
                """
                # training process
                model.train(True)
                for batch in range(batch_num_epoch_train):
                    offset = (batch*batch_size%(train_y.shape[0]-batch_size))
                    batch_x = train_x[offset:(offset+batch_size),:,:,:]
                    batch_x = batch_x.reshape(len(batch_x),3,32,4)
                    # print(batch_x.shape)
                    batch_y = train_y[offset:(offset+batch_size)]
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_x = Variable(batch_x)
                    # print(batch_y)
                    # print(batch_y.shape)
                    optimizer.zero_grad()
                    output, _, _, _ = model(batch_x)
                    # print(output)
                    # print(output.shape)
                    # print(output.shape)
                    # target = torch.empty(batch_size,dtype=torch.long).random_(2) # 修正必要
                    loss = loss_function(output,batch_y)
                    # batch_train_loss.append(loss.item())  
                    pred_train = output.argmax(dim=1,keepdim=True)
                    loss_train = loss.item()
                    correct_train = (pred_train == batch_y).sum().item()
                    """
                    loss_train_total += loss.item()
                    correct_train += (pred_train == batch_y).sum().item()
                    """
                    loss.backward()
                    optimizer.step()
                """    
                avg_loss_train = loss_train_total/batch_num_epoch_train
                avg_acc_train = correct_train/batch_num_epoch_train
                print('Training log: {} fold. {} epoch. Loss: {}'.format(curr_fold+1,epoch+1,avg_loss_train))
                train_loss.append(avg_loss_train)
                train_acc.append(avg_acc_train)
                """
                print('List: {} list. Training log: {} fold. {} epoch. Loss: {}'.format(list,curr_fold+1,epoch+1,loss_train))
                train_loss.append(loss_train)
                train_acc.append(correct_train)
                # print(train_loss)

                # test process
                model.eval()
                """
                loss_test_total = 0
                correct_test = 0
                """
                with torch.no_grad():
                    for batch in range(batch_num_epoch_test):
                        offset = (batch*batch_size%(test_y.shape[0]-batch_size))
                        batch_x = test_x[offset:(offset+batch_size),:,:,:]
                        batch_x = batch_x.reshape(len(batch_x),3,32,4)
                        # print(output.shape)
                        batch_y = test_y[offset:(offset+batch_size)]
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        output, _, _, _ = model(batch_x)
                        #loss_test_total += loss_function(output,batch_y).item()
                        loss_test = loss_function(output,batch_y).item()
                        pred_test = output.argmax(dim=1,keepdim=True)
                        #correct_test += (pred_test == batch_y).sum().item()
                        correct_test = (pred_test == batch_y).sum().item()
                """
                avg_loss_test = loss_test_total/batch_num_epoch_test
                avg_acc_test = correct_test/batch_num_epoch_test
                test_acc.append(avg_acc_test)
                test_loss.append(avg_loss_test)
                print('Train Loss: {}. Train Accuracy {}.'.format(avg_loss_train,avg_acc_train))
                print('Test Loss: {}. Test Accuracy: {}.'.format(avg_loss_test,avg_acc_test))
                """
                test_acc.append(correct_test)
                test_loss.append(loss_test)
                print('Train Loss: {}. Train Accuracy {}.'.format(loss_train,correct_train))
                print('Test Loss: {}. Test Accuracy: {}.'.format(loss_test,correct_test))
                PATH='./param/model.pth'
                torch.save(model.state_dict(),PATH)
            total_train_acc.append(train_acc)
            total_train_loss.append(train_loss)
            total_test_acc.append(test_acc)
            total_test_loss.append(test_loss)
        with open("./data/train_loss_"+list+".pickle",mode="wb") as f:
            pickle.dump(total_train_loss,f)
        with open("./data/test_loss_"+list+".pickle",mode="wb") as f:
            pickle.dump(total_test_loss,f)
        with open("./data/train_acc_"+list+".pickle",mode="wb") as f:
            pickle.dump(total_train_acc,f)
        with open("./data/test_acc_"+list+".pickle",mode="wb") as f:
            pickle.dump(total_test_acc,f)