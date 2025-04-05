import torch
import pandas as pd
import numpy as np
import csv
import torch.nn as nn
import dataset
import utils
import sys
import model
import train
from torch.utils.data import DataLoader, Dataset
import os


def learning(args):
    # hype-params
    lgn_dim = 64
    codebook_dim = 512
    device = torch.device("cuda:0" if True and torch.cuda.is_available() else "cpu")
    data_name = args.dataset
    lgn_name = 'lgn-'+ data_name + '-' + str(lgn_dim)
    vq_name = 'MQ-' + lgn_name
    print('Process: VQ is working:', vq_name)

    # read lgn-embeddings
    LightGCN = torch.load('../src/lgn/'+ lgn_name +'.pth.tar')
    user_emb = LightGCN['embedding_user.weight']  # requires_grad = False
    item_emb = LightGCN['embedding_item.weight']  # requires_grad = False
    print('total number of items:', item_emb.shape[0])
    print('total number of users:', user_emb.shape[0])


    # codebook initial
    if args.vq_model == 'RQ':
        user_vq = model.ResidualVQVAE(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.ResidualVQVAE(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
    elif args.vq_model == 'MQ':
        user_vq = model.MQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.MQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)   
    else:
        NotImplementedError

    # ---------------------- training -------------------------------
    # item vq
    item_vq_name = 'item-' + vq_name
    if args.train_vq is True:
        train.vqvae(item_vq, item_vq_name, device, item_emb, n_embedding=args.n_token, m_book=args.n_book)
    item_vq.load_state_dict(torch.load('../checkpoints/vq/' + item_vq_name + '.pth'))
    item_vq.to(device)

    # user vq
    user_vq_name = 'user-' + vq_name
    if args.train_vq is True:
        train.vqvae(user_vq, user_vq_name, device, user_emb, n_embedding=args.n_token, m_book=args.n_book)
    user_vq.load_state_dict(torch.load('../checkpoints/vq/' + user_vq_name + '.pth'))
    user_vq.to(device)
    
    # -------------------- output -------------------------
    file_path = '../data/' + data_name + '/train.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    
    item_vq.eval()
    user_vq.eval()
    train_rec_dataset = dataset.RecDataset(data, user_emb, user_vq, item_emb, item_vq)
    train_rec_loader = DataLoader(train_rec_dataset, batch_size=256, shuffle=False)

    item_list = []
    user_list = []
    for i, sample in enumerate(train_rec_loader):
        user_id, user_cb_id, item_cb_id = sample
        user_list += user_cb_id
        item_list += item_cb_id

    data = {'user_cb_id': user_list, "item_cb_id": item_list}
    df = pd.DataFrame(data)
    df.to_csv('../data/'+data_name+'/train_codebook.txt')


    file_path = '../data/' + data_name + '/test.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    test_rec_dataset = dataset.RecDataset(data, user_emb, user_vq, item_emb, item_vq)
    test_rec_loader = DataLoader(test_rec_dataset, batch_size=256, shuffle=False)

    item_list = []
    user_list = []
    for i, sample in enumerate(test_rec_loader):
        user_id, user_cb_id, item_cb_id = sample
        user_list += user_cb_id
        item_list += item_cb_id

    data = {'user_cb_id': user_list, "item_cb_id": item_list}
    df = pd.DataFrame(data)
    df.to_csv('../data/'+data_name+'/test_codebook.txt')


def learning_cold(args):
    
    data_name = args.dataset
    file_path = '../data/'+data_name+'/test.txt'
    test_data = pd.read_csv(file_path, header=None)
    shred = 7

    interaction_list = []
    for i in range(len(test_data)):
        ids = test_data.iloc[i, 0].split(" ")
        user = ids[0]
        items = ids[1:]
        interaction_list.append(len(items))
    interaction_list = np.array(interaction_list)
    cold = np.squeeze(np.argwhere(interaction_list <= shred))
    warm = np.squeeze(np.argwhere(interaction_list > shred))

    # hype-params
    lgn_dim = 64
    codebook_dim = 512
    device = torch.device("cuda:"+str(args.cuda) if True and torch.cuda.is_available() else "cpu")
    lgn_name = 'lgn-'+ data_name + '-' + str(lgn_dim)
    vq_name = 'MQ-' + lgn_name
    print('Process: VQ is working:', vq_name)

    # read lgn-embeddings
    LightGCN = torch.load('../src/lgn/'+ lgn_name +'.pth.tar')
    user_emb = LightGCN['embedding_user.weight']  # requires_grad = False
    item_emb = LightGCN['embedding_item.weight']  # requires_grad = False
    warm_user_emb = user_emb[warm]

    # cold_user_emb = user_emb[cold]
    # cold_item_emb = item_emb[cold]

    print('total number of warm users:', warm_user_emb.shape[0])


    # codebook initial
    if args.vq_model == 'RQ':
        user_vq = model.ResidualVQVAE(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.ResidualVQVAE(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
    elif args.vq_model == 'MQ':
        user_vq = model.MQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = model.MQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)   
    else:
        NotImplementedError

    # ---------------------- training -------------------------------
    # item vq
    item_vq_name = 'item-' + vq_name
    if args.train_vq is True:
        train.vqvae_cold(item_vq, item_vq_name, device, item_emb, n_embedding=args.n_token, m_book=args.n_book)
    item_vq.load_state_dict(torch.load('../checkpoints/vq_cold/' + item_vq_name + '.pth'))
    item_vq.to(device)

    # user vq
    user_vq_name = 'user-' + vq_name
    if args.train_vq is True:
        train.vqvae_cold(user_vq, user_vq_name, device, warm_user_emb, n_embedding=args.n_token, m_book=args.n_book)
    user_vq.load_state_dict(torch.load('../checkpoints/vq_cold/' + user_vq_name + '.pth'))
    user_vq.to(device)
    
    # -------------------- output -------------------------
    # warm
    file_path = '../data/' + data_name + '_warm/train.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    
    item_vq.eval()
    user_vq.eval()
    train_rec_dataset = dataset.RecDataset(data, user_emb, user_vq, item_emb, item_vq)
    train_rec_loader = DataLoader(train_rec_dataset, batch_size=256, shuffle=False)

    item_list = []
    user_list = []
    for i, sample in enumerate(train_rec_loader):
        user_id, user_cb_id, item_cb_id = sample
        user_list += user_cb_id
        item_list += item_cb_id

    data = {'user_cb_id': user_list, "item_cb_id": item_list}
    df = pd.DataFrame(data)
    df.to_csv('../data/'+data_name+'_warm/train_codebook.txt')


    file_path = '../data/' + data_name + '_warm/test.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    test_rec_dataset = dataset.RecDataset(data, user_emb, user_vq, item_emb, item_vq)
    test_rec_loader = DataLoader(test_rec_dataset, batch_size=256, shuffle=False)

    item_list = []
    user_list = []
    for i, sample in enumerate(test_rec_loader):
        user_id, user_cb_id, item_cb_id = sample
        user_list += user_cb_id
        item_list += item_cb_id

    data = {'user_cb_id': user_list, "item_cb_id": item_list}
    df = pd.DataFrame(data)
    df.to_csv('../data/'+data_name+'_warm/test_codebook.txt')

    # cold
    file_path = '../data/' + data_name + '_cold/train.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    
    item_vq.eval()
    user_vq.eval()
    train_rec_dataset = dataset.RecDataset(data, user_emb, user_vq, item_emb, item_vq)
    train_rec_loader = DataLoader(train_rec_dataset, batch_size=256, shuffle=False)

    item_list = []
    user_list = []
    for i, sample in enumerate(train_rec_loader):
        user_id, user_cb_id, item_cb_id = sample
        user_list += user_cb_id
        item_list += item_cb_id

    data = {'user_cb_id': user_list, "item_cb_id": item_list}
    df = pd.DataFrame(data)
    df.to_csv('../data/'+data_name+'_cold/train_codebook.txt')


    file_path = '../data/' + data_name + '_cold/test.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    test_rec_dataset = dataset.RecDataset(data, user_emb, user_vq, item_emb, item_vq)
    test_rec_loader = DataLoader(test_rec_dataset, batch_size=256, shuffle=False)

    item_list = []
    user_list = []
    for i, sample in enumerate(test_rec_loader):
        user_id, user_cb_id, item_cb_id = sample
        user_list += user_cb_id
        item_list += item_cb_id

    data = {'user_cb_id': user_list, "item_cb_id": item_list}
    df = pd.DataFrame(data)
    df.to_csv('../data/'+data_name+'_cold/test_codebook.txt')