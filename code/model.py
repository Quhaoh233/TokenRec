import torch
import torch.nn as nn
import utils
import sys


class MQ(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book, mask_ratio=0.1):
        super(MQ, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, dim),
            )

        self.m_book = m_book
        self.codebook = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
            self.codebook.append(codebook)
        
        self.pos = nn.Embedding(1, input_dim)
        self.pos.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
        self.mask_ratio = mask_ratio
            
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(), 
            nn.Linear(128, input_dim),
            )

    def forward(self, x):  # shape = [batch, emb]
        # masked
        b, e =x.shape
        mask = x[0, :].bernoulli_(self.mask_ratio).bool()
        x = torch.masked_fill(x, mask, 0)
        x += self.pos.weight

        # encode    
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebook[0].weight
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebook[m].weight  # It should be learnable!
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list
    
    def valid(self, x):  # shape = [batch, emb]
        x += self.pos.weight.data
        # encode    
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebook[0].weight.data
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebook[m].weight.data  # It should be learnable!
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = zq

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list
    
    def encode(self, x):
        x += self.pos.weight.data
        nearest_neighbor_list = []
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebook[0].weight.data
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
                nearest_neighbor_list.append(nearest_neighbor)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebook[m].weight.data
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
                nearest_neighbor_list.append(nearest_neighbor)
        # zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)  # shape = [batch_size, n_codebook]
        return codeword_idx
    

class ResidualVQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book):
        super(ResidualVQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, dim),
            )

        self.m_book = m_book
        self.codebook = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
            self.codebook.append(codebook)
            
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(), 
            nn.Linear(128, input_dim),
            )

    def forward(self, x):
        # encode
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebook[0].weight
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebook[m].weight  # It should be learnable!
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list

    def valid(self, x):
        # encode
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebook[0].weight.data
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebook[m].weight.data  # It should be learnable!
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list

    def encode(self, x):
        nearest_neighbor_list = []
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebook[0].weight
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
                nearest_neighbor_list.append(nearest_neighbor)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebook[m].weight
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebook[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
                nearest_neighbor_list.append(nearest_neighbor)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)  # shape = [batch_size, n_codebook]
        return codeword_idx


class projection(nn.Module):
    def __init__(self, input_dim, output_dim, target_length, hidden_dim=256):
        super(projection, self).__init__()
        self.l1 = nn.Linear(int(target_length*input_dim), hidden_dim)  # input_dim = 512, output_dim = 64
        self.l2 = nn.Linear(hidden_dim, output_dim)  # input_dim = 512, output_dim = 64
        self.flatten = nn.Flatten()  # default 1 to -1. 0: batch
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.dropout(self.l1(x)))
        x = self.l2(x)
        return x
