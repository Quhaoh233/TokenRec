import os
from torch.utils.data import DataLoader, Dataset
import utils
import sys
import torch
import pandas as pd
import random


def read_data(data_name):
    file_path = '../data/'+data_name+'/train_codebook.txt'
    train_codebook_data = pd.read_csv(file_path, header=0)

    file_path = '../data/'+data_name+'/train.txt'
    with open(file_path, 'r') as f:
        train_data = f.readlines()
    
    file_path = '../data/'+data_name+'/test_codebook.txt'
    test_codebook_data = pd.read_csv(file_path, header=0)

    file_path = '../data/'+data_name+'/test.txt'
    with open(file_path, 'r') as f:
        test_data = f.readlines()

    item_list = pd.read_csv('../data/'+data_name+'/item_list.txt', header=0)
    user_list = pd.read_csv('../data/'+data_name+'/user_list.txt', header=0)
    item_num = len(item_list)
    user_num = len(user_list)

    return train_data, test_data, train_codebook_data, test_codebook_data, item_num, user_num


def read_cold_or_warm_data(data_name, mode='cold'):
    file_path = '../data/'+data_name+'_'+mode+'/train_codebook.txt'
    train_codebook_data = pd.read_csv(file_path, header=0)

    file_path = '../data/'+data_name+'_'+mode+'/train.txt'
    with open(file_path, 'r') as f:
        train_data = f.readlines()
    
    file_path = '../data/'+data_name+'_'+mode+'/test_codebook.txt'
    test_codebook_data = pd.read_csv(file_path, header=0)

    file_path = '../data/'+data_name+'_'+mode+'/test.txt'
    with open(file_path, 'r') as f:
        test_data = f.readlines()
    return train_data, test_data, train_codebook_data, test_codebook_data


class VQDataset(Dataset):
    def __init__(self, embs):
        super().__init__()
        self.embs = embs

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, index: int):
        emb = self.embs[index]
        return emb


class RecDataset(Dataset):
    def __init__(self, data, user_emb, user_vq, item_emb, item_vq, max_item_num=1e10):
        super().__init__()
        self.data = data
        self.max_item_num = max_item_num
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.user_vq = user_vq
        self.item_vq = item_vq

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # orginal idx
        sample = self.data[index]
        sample = sample.split()  # split the list
        user = sample[0]
        item_list = sample[1:]
        if len(item_list) > self.max_item_num:
            item_list = item_list[:self.max_item_num]
        
        user_id = int(user)
        item_idx = [int(x) for x in item_list]

        # codebook idx
        user_emb = self.user_emb[user_id].unsqueeze(0)  # [1, n_emb]
        item_emb = self.item_emb[item_idx]
        with torch.no_grad():
            user_vq_id = self.user_vq.encode(user_emb)  # [1, 3]
            item_vq_idx = self.item_vq.encode(item_emb)  # [n_item, 3]

        user_cb_id = utils.user_codebook_to_str(user_vq_id)
        item_cb_id = utils.item_codebook_to_str(item_vq_idx)
        return user_id, user_cb_id, item_cb_id
    

class LLM4RecDataset(Dataset):
    def __init__(self, data, codebook_data, no_shuffle=False):
        super().__init__()
        self.data = data
        self.codebook_data = codebook_data
        self.no_shuffle = no_shuffle
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # orginal idx
        sample = self.data[index].split(" ")
        user = sample[0]
        item_list = sample[1:]
        user_id = int(user)
        target_id = int(item_list[-1])
        item_id = " ".join(item_list[:-1])

        # ------------------ codebook id ------------------
        codebook_sample = self.codebook_data[index].split(" ")
        user_cb_id = codebook_sample[0]  # string
        item_cb_id = codebook_sample[1:]
        target_cb_id = item_cb_id[-1]
        items = item_cb_id[:-1]
        if self.no_shuffle is False:
            random.shuffle(items)
            item_cb_id_list = " ".join(items)
        elif self.no_shuffle is True:
            item_cb_id_list = " ".join(items) 
        else:
            NotImplementedError
    
        return user_id, item_id, target_id, user_cb_id, item_cb_id_list, target_cb_id



class LLM4RecTrainDataset(Dataset):
    def __init__(self, data, codebook_data, no_shuffle=False):
        super().__init__()
        self.data = data
        self.codebook_data = codebook_data
        self.no_shuffle = no_shuffle
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # ------------------ org_id --------------------
        sample = self.data[index].split(" ")
        user = sample[0]
        item_list = sample[1:]
        user_id = int(user)
        # train_item_list = " ".join(item_list[:-2])
        # valid_item_list = " ".join(item_list[:-1])
        train_target_id = int(item_list[-2])
        valid_target_id = int(item_list[-1])

        # ------------------ codebook id ------------------
        codebook_sample = self.codebook_data[index].split(" ")
        user_cb_id = codebook_sample[0]  # string
        item_cb_id_list = codebook_sample[1:]
        train_target_cb_id = item_cb_id_list[-2]
        valid_target_cb_id = item_cb_id_list[-1]
        train_item_cb_list = item_cb_id_list[:-2]
        valid_item_cb_list = item_cb_id_list[:-1]

        # shuffle the item list
        if self.no_shuffle is False:
            random.shuffle(train_item_cb_list)
            random.shuffle(valid_item_cb_list)
            train_item_cb_list = " ".join(train_item_cb_list)
            valid_item_cb_list = " ".join(valid_item_cb_list)
        elif self.no_shuffle is True:
            train_item_cb_list = " ".join(train_item_cb_list)
            valid_item_cb_list = " ".join(valid_item_cb_list)   
        else:
            NotImplementedError

        return user_id, train_target_id, valid_target_id, user_cb_id, train_item_cb_list, train_target_cb_id, valid_item_cb_list, valid_target_cb_id