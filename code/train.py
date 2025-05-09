import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from kmeans_pytorch import kmeans
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5EncoderModel, T5ForConditionalGeneration, T5Model, AutoTokenizer
import pandas as pd
import model
import sys
from tqdm import tqdm
import dataset
import myevaluate
import utils


def vqvae(model, model_name, device, co_emb, n_embedding, kmean_epoch=50, m_book=3, valid_ratio=0.2, batch_size=512, lr=1e-3, n_epochs=1000, l_w_embedding=1, l_w_commitment=0.25):
    # random sampling
    idx = torch.rand_like(co_emb[:, 0])
    sh = torch.quantile(idx, valid_ratio)
    valid_emb = []
    train_emb = []
    for n in range(co_emb.shape[0]):
        if idx[n] <= sh:
            valid_emb.append(co_emb[n])
        else:
            train_emb.append(co_emb[n])
    valid_emb = torch.stack(valid_emb, dim=0)
    train_emb = torch.stack(train_emb, dim=0)
    train_dataset = dataset.VQDataset(train_emb)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dataset = dataset.VQDataset(valid_emb)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_emb.shape[0], shuffle=True)
    model.to(device)
    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    mse_loss = nn.MSELoss()
    tic = time.time()
    valid_loss = 10
    for e in range(n_epochs):
        total_loss = 0
        model.train()
        for i, x in enumerate(train_dataloader):
            current_batch_size = x.shape[0]
            x = x.to(device)
            x_hat, res_dict, ce_dict = model(x)
            # codebook kmeans initialization
            if e % kmean_epoch == 0 and i == 0:
            # if e in [0, 10, 20] and i == 0:
                for m in range(m_book):
                    ids, centers = kmeans(res_dict[m], n_embedding, distance='euclidean', device=device)
                    model.codebooks[m].weight.data = centers.to(device)
            l_reconstruct = mse_loss(x_hat, x)  # reconstruction_loss / data_variance?
            loss = l_reconstruct
            for m in range(m_book):
                l_embedding = mse_loss(res_dict[m].detach(), ce_dict[m])
                # l_commitment = F.mse_loss(res_dict[m], ce_dict[m].detach())
                loss += l_w_embedding * l_embedding # + l_w_commitment * l_commitment
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(train_dataloader.dataset)

        model.eval()
        epoch_loss = 0
        for x in valid_dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            x_hat, res_dict, ce_dict = model.valid(x)
            l_reconstruct = mse_loss(x_hat, x)
            loss = l_reconstruct
            for m in range(m_book):
                l_embedding = mse_loss(res_dict[m], ce_dict[m])
                l_commitment = mse_loss(res_dict[m], ce_dict[m])
                loss += l_w_embedding * l_embedding + l_w_commitment * l_commitment
            epoch_loss += loss.item() * current_batch_size
        epoch_loss /= len(valid_dataloader.dataset)

        if epoch_loss < valid_loss:
            # print('Pass the validation, Save the MQ model.')
            valid_loss = epoch_loss
            torch.save(model.state_dict(), '../checkpoints/vq/' + model_name + '.pth')
            
        toc = time.time()
        if e % 10 == 0:
            print(f'VQ: epoch {e} train_loss: {total_loss} valid_loss: {valid_loss} elapsed {(toc - tic):.2f}s')


def backbone(data_name, train_rec_loader, valid_rec_loader, user_emb, item_emb, item_num, args, device):
    # read checkpoints
    if args.train_from_checkpoint is True:
        linear_projection = model.projection(input_dim=512, output_dim=item_emb.shape[1], target_length=args.target_length)
        t5 = T5Model.from_pretrained('../checkpoints/backbone/' + data_name)
        tokenizer = AutoTokenizer.from_pretrained("../checkpoints/backbone/" + data_name, legacy=False)
        linear_projection.load_state_dict(torch.load('../checkpoints/backbone/' + data_name +'/projection.pt'))
        grouped_params = utils.group_model_params(t5, linear_projection, decay=args.decay)
    # read pretrained llms
    else:
        linear_projection = model.projection(input_dim=512, output_dim=item_emb.shape[1], target_length=args.target_length)
        tokenizer = AutoTokenizer.from_pretrained("../src/t5-small", legacy=False, local_files_only=True)
        t5 = T5Model.from_pretrained("../src/t5-small", local_files_only=True)
        add_tokens = utils.codebook_tokens(args.n_book, args.n_token)
        num_added_toks = tokenizer.add_tokens(add_tokens)  # add tne tokens to the tokenizer vocabulary
        t5.resize_token_embeddings(len(tokenizer))  # add new, random embeddings for the new tokens
        print('added token number =', num_added_toks)
        grouped_params = utils.group_model_params(t5, linear_projection, decay=args.decay)

    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr)
    # loss_func = torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
    # loss_func = torch.nn.MSELoss()
    loss_func = torch.nn.CosineEmbeddingLoss()
    t5.to(device)
    linear_projection.to(device)

    # ------------------------ training --------------------------------
    max_epoch = args.epochs
    global_metric = 0
    loss_list = []
    metric_list = []
    for epoch in range(max_epoch):
        t5.train()
        linear_projection.train()
        loss_record = 0
        batch_record = 0
        for i, sample in enumerate(train_rec_loader):
            # train
            user_id, train_target_id, valid_target_id, user_cb_id, train_item_cb_id, train_target_cb_id, valid_item_cb_id, valid_target_cb_id = sample  # tuple text sequence [batch]
            train_batch = len(train_item_cb_id)
            input_sentences = utils.prompt(user_cb_id, train_item_cb_id)
            if len(input_sentences) == 0:  # if the list is empty, skip the batch
                continue
            targets = utils.get_target_emb(item_emb, train_target_id)
            input_encoding = tokenizer(input_sentences, return_tensors='pt', max_length=args.source_length, padding="max_length", truncation=True)  # padding to max model input length
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            decoder_input_encoding = tokenizer([args.decoder_prepend for _ in range(len(train_target_cb_id))], return_tensors="pt", max_length=args.target_length, padding="max_length", truncation=True)
            decoder_input_ids, decoder_attention_mask = decoder_input_encoding.input_ids, decoder_input_encoding.attention_mask
            decoder_input_ids = t5._shift_right(decoder_input_ids)

            outputs = t5(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), decoder_input_ids=decoder_input_ids.to(device))
            last_hidden_states = outputs.last_hidden_state  # shape = [batch, max_source_length, embedding]
            predicts = linear_projection(last_hidden_states)  # predicts = [batch, emb],

            # negative sampling, 1:1
            current_batch = predicts.shape[0]
            neg_idx = torch.randint(0, item_num, (current_batch, ))
            neg_sample = item_emb[neg_idx, :]
            
            # concat
            predicts = torch.cat((predicts, predicts), 0)
            samples = torch.cat((targets, neg_sample), 0)
            labels = torch.ones([2*current_batch])
            labels[current_batch:] = -1
            loss = loss_func(predicts, samples.to(device), labels.to(device))  # targets = [batch, embedding]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_record += loss.item()*train_batch
            batch_record += train_batch
            # score = utils.similarity_score(predicts, item_emb)
            if i % 100 == 0:
                print('epoch =', epoch, 'batch =', i, 'train_loss =', loss.item())
        
        loss_list.append(loss_record/batch_record)

        
        # ----------------- validation -----------------------------------
        if epoch % 10 == 0:
            metrics = torch.zeros([2]).to(device)
            n_batch = 0
            t5.eval()
            linear_projection.eval()
            for i, sample in enumerate(tqdm(valid_rec_loader)):
                user_id, item_id, target_id, user_cb_id, item_cb_id, target_cb_id = sample
                input_sentences = utils.prompt(user_cb_id, item_cb_id)
                targets = utils.get_target_emb(item_emb, target_id)
                input_encoding = tokenizer(input_sentences, return_tensors='pt', max_length=args.source_length, padding="max_length", truncation=True)
                input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
                decoder_input_encoding = tokenizer([args.decoder_prepend for _ in range(len(target_cb_id))], return_tensors="pt", max_length=args.target_length, padding="max_length", truncation=True)
                decoder_input_ids, decoder_attention_mask = decoder_input_encoding.input_ids, decoder_input_encoding.attention_mask
                decoder_input_ids = t5._shift_right(decoder_input_ids)

                with torch.no_grad():
                    outputs = t5(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), decoder_input_ids=decoder_input_ids.to(device))
                    last_hidden_states = outputs.last_hidden_state  # shape = [batch, max_source_length, embedding]
                    predicts = linear_projection(last_hidden_states)  # shape = [batch, emb]
                if args.similarity == 'cos':  # default
                    scores = utils.similarity_score(predicts, item_emb, item_id)  # the bigger the better
                    results = torch.argsort(scores, dim=1, descending=True)
                elif args.similarity == 'MSE':
                    scores = utils.MSE_distance(predicts, item_emb) # the less the better  
                    results = torch.argsort(scores, dim=1, descending=False) 
                else:
                    NotImplementedError

                metr, batch = myevaluate.get_metrics(target_id, results, device, args.k)
                metrics += metr
                n_batch += batch
                # valid_loss += loss.item() * batch

            # valid_loss = valid_loss / n_batch
            print(data_name, 'valid_hit@%s =' % args.k, metrics[0].item()/n_batch, 'valid_ndcg@%s =' %  args.k, metrics[1].item()/n_batch)
            
            metric_list.append(metrics/n_batch)
            # save checkpoints
            if torch.mean(metrics/n_batch) > global_metric:
                global_metric = torch.mean(metrics/n_batch)
                print('Pass the validation, save checkpoints ...')
                t5.save_pretrained('../checkpoints/backbone/' + data_name)
                tokenizer.save_pretrained("../checkpoints/backbone/" + data_name)
                torch.save(linear_projection.state_dict(), '../checkpoints/backbone/'  + data_name + '/projection.pt')

            metric_output = torch.stack(metric_list, dim=0)
            metric_save = pd.DataFrame(metric_output.detach().cpu().numpy(), columns=['hit@%s' % args.k, 'ncdg@%s' % args.k])
            metric_save.to_csv('../results/'+data_name+'/valid_metric_record.csv', index=False)
            loss_output = pd.DataFrame(loss_list, columns=['loss'])
            loss_output.to_csv('../results/'+data_name+'/train_loss_record.csv', index=False)