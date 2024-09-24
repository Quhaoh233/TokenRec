import torch
import numpy as np
import pandas as pd
import sys
import train
from tqdm import tqdm
import dataset
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import utils
import test
from parse import parse_args
import vq


# hype-params
args = parse_args()

# pipeline: phase 1: vq, phase 2: llm
data_name = args.dataset
print('iLLMRec is working on', data_name)

use_cuda = True
device = torch.device("cuda:" + str(args.cuda) if use_cuda and torch.cuda.is_available() else "cpu")

if args.vq is True:
    print(device)
    vq.learning(args)

# --------------------- read data -----------------------
lgn_dim = 64
model_name = 'lgn'  # mf or lgn
checkpoint_name = model_name + '-'+ data_name + '-' + str(lgn_dim)
user_emb, item_emb = utils.read_cf_embeddings(model_name, checkpoint_name)
train_data, test_data, train_codebook_data, test_codebook_data, item_num, user_num = dataset.read_data(data_name)


valid_data, valid_codebook_data = utils.data_construction(test_data, test_codebook_data, args.item_limit)
test_data, test_codebook_data = utils.data_construction(test_data, test_codebook_data, args.item_limit)  # list

if args.no_data_augment is False:
    train_data, train_codebook_data = utils.data_augment(train_data, train_codebook_data, item_limit=args.item_limit)  # list
elif args.no_data_augment is True:
    train_data, train_codebook_data = utils.data_construction(train_data, train_codebook_data, args.item_limit)  # list
else:
    NotImplementedError

train_rec_dataset = dataset.LLM4RecTrainDataset(train_data, train_codebook_data, args.no_shuffle)
train_rec_loader = DataLoader(train_rec_dataset, batch_size=args.batch, shuffle=True, drop_last=False)
valid_rec_dataset = dataset.LLM4RecDataset(valid_data, valid_codebook_data, args.no_shuffle)
valid_rec_loader = DataLoader(valid_rec_dataset, batch_size=args.batch, shuffle=False, drop_last=False)
test_rec_dataset = dataset.LLM4RecDataset(test_data, test_codebook_data, no_shuffle=True)
test_rec_loader = DataLoader(test_rec_dataset, batch_size=args.batch, shuffle=False, drop_last=False)
print('item number =', item_num)
print('user number =', user_emb.shape[0])

if args.no_train == False:
    train.backbone(data_name, train_rec_loader, valid_rec_loader, user_emb.to(device), item_emb.to(device), item_num, args, device)
test.backbone(data_name, test_rec_loader, user_emb.to(device), item_emb.to(device), item_num, args, device)


        