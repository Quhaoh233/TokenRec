import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from kmeans_pytorch import kmeans
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5EncoderModel, T5ForConditionalGeneration, T5Model
import pandas as pd
import model
import sys
import time
from tqdm import tqdm
import myevaluate
import utils


def backbone(data_name, test_rec_loader, user_emb, item_emb, item_num, args, device):

    max_source_length = args.source_length
    k = args.k
    # --------------------------- evaluate ------------------------------------------
    linear_projection = model.projection(input_dim=512, output_dim=item_emb.shape[1], target_length=args.target_length)
    t5 = T5Model.from_pretrained('../checkpoints/backbone/' + data_name)
    tokenizer = T5Tokenizer.from_pretrained("../checkpoints/backbone/" + data_name, legacy=False)
    linear_projection.load_state_dict(torch.load('../checkpoints/backbone/' + data_name +'/projection.pt'))

    t5.to(device)
    linear_projection.to(device)

    t5.eval()
    linear_projection.eval()
    n_batch = 0
    metrics = torch.zeros([4, 2]).to(device)
    for i, data in enumerate(tqdm(test_rec_loader)):
        user_id, item_id, target_id, user_cb_id, item_cb_id, target_cb_id = data
        input_sentences = utils.prompt(user_cb_id, item_cb_id, is_test=True, is_unseen=args.is_unseen)
        # target = utils.get_target_emb(item_emb, target_id)
        if i == 0:
            print('Input Example =', input_sentences[0])
        input_encoding = tokenizer(input_sentences, return_tensors='pt', max_length=max_source_length, padding="max_length", truncation=True)
        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        decoder_input_encoding = tokenizer([args.decoder_prepend for _ in range(len(target_cb_id))], return_tensors="pt", max_length=args.target_length, padding="max_length", truncation=True)
        decoder_input_ids, decoder_attention_mask = decoder_input_encoding.input_ids, decoder_input_encoding.attention_mask
        decoder_input_ids = t5._shift_right(decoder_input_ids)

        outputs = t5(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), decoder_input_ids=decoder_input_ids.to(device))
        last_hidden_states = outputs.last_hidden_state  # shape = [batch, max_source_length, embedding]
        predicts = linear_projection(last_hidden_states)

        if args.similarity == 'cos':
            scores = utils.similarity_score(predicts, item_emb, item_id)
            results = torch.argsort(scores, dim=1, descending=True)  # the bigger the better
        elif args.similarity == 'MSE':
            scores = utils.MSE_distance(predicts, item_emb)
            results = torch.argsort(scores, dim=1, descending=False)  # the less the better  
        else:
            NotImplementedError

        for j in range(metrics.shape[0]):
            metr, batch = myevaluate.get_metrics(target_id, results, device, (j+1)*10)
            metrics[j, :] += metr
        n_batch += batch

    print(data_name)
    for j in range(metrics.shape[0]):
        print('test_hit@%s =' % ((j+1)*10), metrics[j, 0].item()/n_batch, 'test_ndcg@%s =' % ((j+1)*10), metrics[j, 1].item()/n_batch)