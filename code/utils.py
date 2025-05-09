import torch
import torch.nn as nn
import sys
import numpy as np
import pandas as pd
import random
import time


def reconstruct(model, emb, device):
    model.to(device)
    model.eval()
    mse_loss = nn.MSELoss()
    with torch.no_grad():
        emb_hat, _, _ = model(emb)
        loss = mse_loss(emb_hat, emb)
        print('test loss:', loss.item())
    return emb_hat


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

	
def item_codebook_to_str(vq_id):
	id_num = vq_id.shape[0]
	codebook_num = vq_id.shape[1]
	sample = []
	for i in range(id_num):
		temp = ['item_']
		for j in range(codebook_num):
			token = "".join(["<", str(j), '-', str(vq_id[i, j].item()), '>'])
			temp.append(token)
		temp = "".join(temp)
		sample.append(temp)
	sample = " ".join(sample)
	return sample


def user_codebook_to_str(vq_id):
	user_head = ['a', 'b', 'c', 'd', 'e']
	id_num = vq_id.shape[0]
	codebook_num = vq_id.shape[1]
	sample = []
	for i in range(id_num):
		temp = ['user_']
		for j in range(codebook_num):
			token = "".join(["<", user_head[j], '-', str(vq_id[i, j].item()), '>'])
			temp.append(token)
		temp = "".join(temp)
		sample.append(temp)
	sample = " ".join(sample)
	return sample


def group_model_params(model1, model2, decay):
	grouped_params = [
            {
                "params": [
                    p
                    for n, p in model1.named_parameters()
                ],
                "weight_decay": decay,
            },
            {
                "params": [
                    p
                    for n, p in model2.named_parameters()
                ],
                "weight_decay": decay,
            },
        ]

	return grouped_params


def group_model_emb_params(model1, model2, emb, decay):
	grouped_params = [
            {
                "params": [
                    p
                    for n, p in model1.named_parameters()
                ],
                "weight_decay": decay,
            },
            {
                "params": [
                    p
                    for n, p in model2.named_parameters()
                ],
                "weight_decay": decay,
            },
			{
                "params": emb.parameters(),
                "weight_decay": decay,
            },
        ]

	return grouped_params


def seq_construct(item_id, train_item_cb_id, user_cb_id):
	user_list = []
	seq_list = []
	label_list = []
	for tx_id, cb_id, user in zip(item_id, train_item_cb_id, user_cb_id):

		items = cb_id.split(' ')  # item_<x-xx><x-xxx><x-xx>
		ids = tx_id.split(' ')
		temp_list = []
		for j, item in enumerate(items):
			temp_list.append(item)
			if j >= 2:
				temp = " ".join(temp_list)
				seq_list.append(temp)
				label_list.append(int(ids[j+1]))
				user_list.append(user)

	label_list = torch.tensor(np.array(label_list))
	return seq_list, label_list, user_list


def seq_construct_back(item_id, train_item_cb_id, user_cb_id):
	user_list = []
	seq_list = []
	label_list = []
	seq_num = 0
	for tx_id, cb_id, user in zip(item_id, train_item_cb_id, user_cb_id):  # batch
		if seq_num < 20:
			items = cb_id.split(' ')  # item_<x-xx><x-xxx><x-xx>
			ids = tx_id.split(' ')
			for j in range(1, len(items)):  # seq
				if len(items) - j > 2:
					temp_list  = items[:-j]
					temp = " ".join(temp_list)
					seq_list.append(temp)
					label_list.append(int(ids[-j-1]))
					user_list.append(user)
					seq_num += 1
		else:
			break

	label_list = torch.tensor(np.array(label_list))
	return seq_list, label_list, user_list


def seq_construct_v2(item_id, train_item_cb_id, user_cb_id):
	user_list = []
	seq_list = []
	label_list = []
	label_cb_list = []
	for tx_id, cb_id, user in zip(item_id, train_item_cb_id, user_cb_id):
		items = cb_id.split(' ')  # item_<x-xx><x-xxx><x-xx>
		ids = tx_id.split(' ')
		temp_list = []
		for j in range(len(items)-1):
			item = items[j]
			temp_list.append(item)
			if j > 2:
				temp = " ".join(temp_list)
				seq_list.append(temp)
				label_cb_list.append(items[j+1])
				label_list.append(int(ids[j+1]))
				user_list.append(user)

	label_list = torch.tensor(np.array(label_list))
	return seq_list, label_list, user_list, label_cb_list


def prompt(user_batch, items_batch, is_test=False, is_unseen=False):
    prefix = prefix_prompt()
    if is_test is False:
        sentences = [prefix + train_prompt(user, items) for user, items in zip(user_batch, items_batch)]
    elif is_test is True:
        if is_unseen is False:
            sentences = [prefix + seen_prompt(user, items) for user, items in zip(user_batch, items_batch)]
        elif is_unseen is True:
            sentences = [prefix + unseen_prompt(user, items) for user, items in zip(user_batch, items_batch)]
        else:
            NotImplementedError
    else:
        NotImplementedError
    return sentences


def train_prompt(user, items):
	prompts = dict()
	prompts[0] = f'Given the following purchase history for the {user}: {items}. Predict the user preferences.'
	prompts[1] = f'I find the purchase history list for the {user}: {items}. I wonder what the user will like. Can you help me decide?'
	prompts[2] = f'Considering the {user} has interacted with {items}. What are the user preferences?'
	prompts[3] = f'According to what items the {user} has purchased: {items}. Can you describe the user preferences?'
	prompts[4] = f"By analyzing the {user}'s purchase of {items}, what are the expected preferences of the user?"
	prompts[5] = f"Given the {user}'s previous interactions with the {items}, what are the user preferences?"
	prompts[6] = f"Taking into account the {user}'s engagement with the {items}, what are the user potential interests?"
	prompts[7] = f"In light of the {user}'s interactions with the {items}, what might the user be interested in?"
	prompts[8] = f"Considering the {user}'s past interactions with the {items}, what are the user likely preferences?"
	prompts[9] = f"With the {user}'s history of engagement with the {items}, what would the user be inclined to like?"
	idx = int(np.random.randint(len(prompts), size=1))

	return prompts[idx]

def unseen_prompt(user, items):
	prompt = f"Based on the {user}'s historical engagement with the {items}, what would the user likely be interested in?"
	return prompt

def seen_prompt(user, items):
	prompt = f"Given the {user}'s previous interactions with the {items}, what are the user preferences?"  # 5
	return prompt

def prefix_prompt(RP=False, ICL=False, CoT=False):
    output = ''
    if RP:
        output += 'You are an expert at recommending products to users based on their purchase histories. '
    if ICL:
        output += 'Here is an example format for recommendations. ### Input: Given the following purchase history: [previous items]. I wonder what the user will like. Can you help me decide? ### Response: The interaction history shows that the user might like [predictive item]. '
    if CoT:
        output += 'Let’s think step by step. '
    return output + '\n'

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def read_cf_embeddings(model_name, checkpoint_name):
	model = torch.load('../src/'+model_name+'/'+ checkpoint_name +'.pth.tar')
	user_emb = model['embedding_user.weight']  # requires_grad = False
	item_emb = model['embedding_item.weight']  # requires_grad = False

	return user_emb, item_emb


def get_target_emb(item_emb, labels):
	target = item_emb[labels]
	return target


def codebook_tokens(n_book, n_token):
	add_tokens = []
	# item
	for i in range(n_book):
		for j in range(n_token):
			token = "<" + str(i) + '-' + str(j) + '>'
			add_tokens.append(token)
	# user
	user_head = ['a', 'b', 'c', 'd', 'e']
	for i in range(n_book):
		for j in range(n_token):
			token = "<" + user_head[i] + '-' + str(j) + '>'
			add_tokens.append(token)
	return add_tokens


def similarity_score(predicts, item_emb, item_id):
	'''
	predicts.shape = [batch, emb]
	item_emb.shape = [item_num, emb]
	items.shape = [batch, num]
	'''
	score = []
	cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
	batch = predicts.shape[0]
	for i in range(batch):
		items = item_id[i].split(" ")
		items = [int(item) for item in items]
		temp = cos(predicts[i, :].unsqueeze(0), item_emb)
		temp[items] = 0
		score.append(temp)
	score = torch.stack(score, dim=0)
	return score


def MSE_distance(predicts, item_emb):
	'''
	predicts.shape = [batch, emb]
	item_emb.shape = [item_num, emb]
	'''
	score = []
	batch, dim = predicts.shape
	item_num, _ = item_emb.shape
	for i in range(batch):
		temp = predicts[i, :].unsqueeze(0).expand(item_num, -1)  # [item_num, emb]
		dis = (temp - item_emb).pow(2).sum(1).sqrt()
		score.append(dis)
	score = torch.stack(score, dim=0)
	return score


def whole_word_embedding(tokenizer, emb, input_ids, n_book):
	'''
	emb.shape = torch.tensor([source_length, 512])
	input_ids.shape = torch.size([batch, source_length])
	'''
	# tic = time.time()
	batch_whole_word_emb = []
	batch, source_l = input_ids.shape
	mark_id = tokenizer.convert_tokens_to_ids('_')
	for i in range(batch):
		sentence = input_ids[i, :]
		ids = torch.arange(len(sentence))+1
		ids[(sentence == 0).nonzero(as_tuple=True)] = 0
		marks = (sentence == mark_id).nonzero(as_tuple=False)
		curr = torch.max(ids)
		for j in range(marks.shape[0]):
			curr += 1
			temp = marks[j, :]
			idx = torch.arange(-1, n_book+1, 1) + temp 
			ids[idx] = curr
		curr_emb = torch.stack([emb.weight[ids[l]] for l in range(source_l)], dim=0)
		batch_whole_word_emb.append(curr_emb)
	batch_whole_word_emb = torch.stack(batch_whole_word_emb, dim=0)
	# toc = time.time()
	# print(f'elapsed {(toc - tic):.2f}s')
	return batch_whole_word_emb


def whole_word_embedding_v2(tokenizer, emb, input_ids, n_book):
	'''
	emb.shape = torch.tensor([source_length, 512])
	input_ids.shape = torch.size([batch, source_length])
	'''
	tic = time.time()
	batch, source_l = input_ids.shape
	mark_id = tokenizer.convert_tokens_to_ids('_')
	whole_word_ids = torch.arange(source_l)+1
	whole_word_ids = whole_word_ids.unsqueeze(dim=0).expand(batch, -1)
	whole_word_ids[(input_ids == 0).nonzero(as_tuple=True)] = 0
	marks = (input_ids == mark_id).nonzero(as_tuple=True)
	rows, columns = marks
	for n in range(-1, n_book+1, 1):
		whole_word_ids[rows, columns+n] = whole_word_ids[rows, columns]

	batch_whole_word_emb = []
	for b in range(batch):
		sentence_emb = []
		for l in range(source_l):
			sentence_emb.append(emb.weight[whole_word_ids[b, l]])
		sentence_emb = torch.stack(sentence_emb, dim=0)
		batch_whole_word_emb.append(sentence_emb)
	batch_whole_word_emb = torch.stack(batch_whole_word_emb, dim=0)
	toc = time.time()
	print(f'elapsed {(toc - tic):.2f}s')
	return batch_whole_word_emb  # shape = [batch, source_l, 512]


def data_augment(id_list, codebook_id_list, shred=2, item_limit=20):
	'''
	id_list = list
	codebook_id_list = pd.Dataframe
	'''
	num = len(id_list)
	samples = []
	codebook_samples = []
	for n in range(num):
		ids = id_list[n].strip('\n').split(" ")
		user_id = ids[0]
		item_id = ids[1:]
		user_codebook_id = codebook_id_list['user_cb_id'][n]
		item_codebook_id = codebook_id_list['item_cb_id'][n].strip('\n').split(" ")

		temp_sample = []
		temp_codebook_sample = []
		temp_sample.append(user_id)
		temp_codebook_sample.append(user_codebook_id)
		for k in range(len(item_id)):
			if k > item_limit:
				break
			temp_sample.append(item_id[k])
			temp_codebook_sample.append(item_codebook_id[k])
			if k > shred:
				sample = " ".join(temp_sample)
				codebook_sample = " ".join(temp_codebook_sample)
				samples.append(sample)
				codebook_samples.append(codebook_sample)


	output_samples = samples
	output_codebook_samples = codebook_samples
	return output_samples, output_codebook_samples


def data_construction(id_list, codebook_id_list, item_limit=100):
	'''
	id_list = lisn
	codebook_id_list = pd.Dataframe
	'''
	num = len(id_list)
	samples = []
	codebook_samples = []
	for n in range(num):
		ids = id_list[n].strip('\n').split(" ")
		user_id = ids[0]
		item_id = ids[1:]
		user_codebook_id = codebook_id_list['user_cb_id'][n]
		item_codebook_id = codebook_id_list['item_cb_id'][n].strip('\n').split(" ")

		temp_sample = []
		temp_codebook_sample = []
		temp_sample.append(user_id)
		temp_codebook_sample.append(user_codebook_id)
		for k in range(len(item_id)):
			if k > item_limit:
				break
			temp_sample.append(item_id[k])
			temp_codebook_sample.append(item_codebook_id[k])

		sample = " ".join(temp_sample)
		codebook_sample = " ".join(temp_codebook_sample)
		samples.append(sample)
		codebook_samples.append(codebook_sample)

	return samples, codebook_samples
