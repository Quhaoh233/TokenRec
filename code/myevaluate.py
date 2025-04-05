import math
import torch


def get_metrics(targets, results, device, k):
    metrics = torch.zeros([2]).to(device)
    hits, batch = hit_at_k(targets, results, k)
    ndcg, _ = ndcg_at_k(targets, results, k)
    metrics[0] = hits
    metrics[1] = ndcg
    return metrics, batch


def hit_at_k(labels, results, k):
    '''
    labels.shape = [batch]
    results.shape = [batch, item_num]
    '''
    hit = 0.0
    batch = results.shape[0]
    for i in range(batch):
        res = results[i, :k]
        label = labels[i]
        if label in res:
            hit += 1
    return hit, batch


def ndcg_at_k(labels, results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    batch = results.shape[0]
    for i in range(batch):
        res = results[i, :k]
        label = labels[i]
        one_ndcg = 0.0
        rel = torch.where(res == label, 1, 0)
        for j in range(len(rel)):
            one_ndcg += rel[j] / math.log(j+2,2)
        ndcg += one_ndcg
    return ndcg, batch


def hit_at_k_v2(results, labels, k):
    '''
    len(predicts) = batch*k, list
    len(labels) = batch, list
    '''
    hit = 0
    batch = len(labels)
    for b in range(batch):
        topk = results[b*k:(b+1)*k]
        if labels[b] in topk:
            hit += 1
    return hit, batch


def ncdg_at_k_v2(results, labels, k):
    '''
    len(predicts) = batch*k, list
    len(labels) = batch, list
    '''
    ndcg = 0.0
    batch = len(labels)
    for b in range(batch):
        label = labels[b]
        topk = results[b*k:(b+1)*k]
        one_ndcg = 0.0
        for i in range(len(topk)):
            if topk[i] == label:
                one_ndcg += 1 / math.log(i+2,2)
        ndcg += one_ndcg
    return ndcg, batch
