import pickle as pkl
import json
import yaml
import gc
import torch
import torch.nn.functional as F


############### utils ################

def print_dict(d):
    for key in d:
        print(key, ":", d[key])


def bpr_loss(pos_score, neg_score):
    return torch.mean(F.softplus(neg_score - pos_score))


################# io #################

def save_pickle(filename, obj):
    with open(filename, "wb") as f:
        pkl.dump(obj, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        gc.disable()
        obj = pkl.load(f)
        gc.enable()
    return obj


def save_json(filename, obj):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    return obj


def load_yaml(filename):
    with open(filename, 'r') as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    return obj


def save_yaml(filename, obj):
    with open(filename, 'w') as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)



def bpr_loss(pos_score, neg_score):
    if pos_score.shape != neg_score.shape:
        num_neg = neg_score.shape[-1]
        pos_score = pos_score.repeat_interleave(
            num_neg, dim=-1).reshape(neg_score.shape)
        
    return torch.mean(torch.nn.functional.softplus(neg_score - pos_score))


def dot_product(src_emb, dst_emb):
    if src_emb.shape != dst_emb.shape:
        return (src_emb.unsqueeze(-2) * dst_emb).sum(dim=-1)
    else:
        return (src_emb * dst_emb).sum(dim=-1)


############## metric ##############
def evaluate(y_pred_pos, y_pred_neg):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''
    Ks = [50, 100]
    hitsKs = []

    for K in Ks:
        if len(y_pred_neg) < K:
            return {'hits@{}'.format(K): 1.}

        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
        hitsKs.append(hitsK)

    return hitsKs[0], hitsKs[1]



class EdgeDataloader:

    def __init__(self, edge_set, confidence, batch_size, ratio):
        self.edge_set = edge_set
        self.batch_size = batch_size
        self.confidence = confidence
        self.index = 0

        self.num_edges = edge_set.size()[0]
        
        self.batch_size = batch_size
        
        self.batch_per_epoch = int(self.num_edges * ratio / self.batch_size)
        self.batch_remain = None
        
    
    def __len__(self):
        return self.batch_per_epoch
    
    def __iter__(self):
        self.batch_remain = self.batch_per_epoch
        return self
    
    def __next__(self):
        if self.batch_remain == 0:
            raise StopIteration
        else:
            self.batch_remain -= 1
        
        eid = torch.randperm(self.num_edges)[:self.batch_size]
        # print(eid[:10])
        
        return (self.edge_set[eid], self.confidence[eid])
        

