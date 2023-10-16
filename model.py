import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import dgl

from collections import defaultdict
from utils import *
from dgl.nn import SAGEConv, GINConv, GATConv





class merger(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        super(merger, self).__init__()

        # common MLP layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_dim, out_dim))
        # for _ in range(num_layers - 2):
        #     self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        # self.layers.append(torch.nn.Linear(hidden_dim, out_dim))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.layers:
            torch.nn.init.xavier_uniform_(lin.weight)
            torch.nn.init.constant_(lin.bias, 0)
    

    def forward(self, emb_ori, emb_aug):
        x = torch.cat((emb_ori, emb_aug), dim=1)
        # common layers
        for lin in self.layers[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.layers[-1](x)

        return x


class PPRGo(torch.nn.Module):
    def __init__(self, graph, emb_dim, dataset, topk, root, device):
        super().__init__()

        raw_nei = load_pickle(osp.join(root, dataset+"/"+dataset+"_nei.pkl"))
        raw_wei = load_pickle(osp.join(root, dataset+"/"+dataset+"_wei.pkl"))

        self.nei = torch.LongTensor(raw_nei[:, : topk]).to(device)
        self.wei = torch.FloatTensor(raw_wei[:, : topk]).to(device)

        _w = torch.ones(self.nei.shape).to(device)
        _w[self.wei == 0] = 0
        self.wei = _w / (_w.sum(dim=-1, keepdim=True) + 1e-12)

        self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim).to(device)  # initialized from N(0, 1)
        self.opt_param_list = []
        self.opt_param_list.extend(self.emb_table.parameters())
        self.device = device

    def forward(self):
        # device = f'cuda:1' if torch.cuda.is_available() else 'cpu'
        # self.device = torch.device(device)
        top_embs = self.emb_table.to(self.device)(self.nei.to(self.device))
        top_weights = self.wei.to(self.device)

        out_emb = torch.matmul(top_weights.unsqueeze(-2), top_embs)
        
        return out_emb
    
    def parameters(self):
        return self.opt_param_list



class SAGE(torch.nn.Module):
    def __init__(self, graph, emb_dim, num_layers, device):
        super().__init__()
        self.graph = graph.to(device)
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim)  # initialized from N(0, 1)

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(emb_dim, emb_dim, aggregator_type='mean', activation=F.relu))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(emb_dim, emb_dim, aggregator_type='mean', activation=F.relu))
        self.convs.append(SAGEConv(emb_dim, emb_dim, aggregator_type='mean'))
        self.convs = self.convs.to(device)

        self.device = device

    def forward(self):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)
        for conv in self.convs[:-1]:
            x = conv(self.graph, x)
            x = F.relu(x)
        emb = self.convs[-1](self.graph, x)
        return emb




class GAT(torch.nn.Module):
    def __init__(self, graph, emb_dim, num_layers, device):
        super().__init__()
        self.graph = dgl.add_self_loop(graph)
        self.graph = self.graph.to(device)
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim)  # initialized from N(0, 1)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(emb_dim, emb_dim, num_heads=4))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(emb_dim, emb_dim, num_heads=4))
        self.convs.append(GATConv(emb_dim, emb_dim, num_heads=4))
        self.convs = self.convs.to(device)

        self.device = device

    def forward(self):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)
        for conv in self.convs[:-1]:
            x = conv(self.graph, x)
            x = x.mean(dim=-2)
            x = F.relu(x)
        emb = self.convs[-1](self.graph, x)
        emb = emb.mean(dim=-2)
        return emb


class GIN(torch.nn.Module):
    def __init__(self, graph, emb_dim, num_layers, device):
        super().__init__()
        self.graph = graph.to(device)
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim)  # initialized from N(0, 1)

        self.convs = torch.nn.ModuleList()
        lin = torch.nn.Linear(emb_dim, 256)
        self.convs.append(GINConv(lin, 'max'))
        for i in range(num_layers - 2):
            lin = torch.nn.Linear(256, 256)
            self.convs.append(GINConv(i, 'max'))
        lin = torch.nn.Linear(256, emb_dim)
        self.convs.append(GINConv(lin, 'max'))
        self.convs = self.convs.to(device)

        self.device = device

    def forward(self):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)
        for conv in self.convs[:-1]:
            x = conv(self.graph, x)
            x = F.relu(x)
        emb = self.convs[-1](self.graph, x)
        return emb
        


class UltraGCN(nn.Module):

    def __init__(self, graph, config, dataset, device):
        super().__init__()
        self.device = device
        self.config = config
        self.graph = graph
        self.emb_table = torch.nn.Embedding(graph.num_nodes(), config['emb_dim'])  # initialized from N(0, 1)

        constrain_mat_file = 'dataset/'+dataset+'/constrain_mat.pkl'
        topk_neighbors_file = 'dataset/'+dataset+'/ii_topk_neighbors.np.pkl'
        topk_similarity_file = 'dataset/'+dataset+'/ii_topk_similarity_scores.np.pkl'

        if self.config['lambda'] > 0:
            constrain_mat = load_pickle(constrain_mat_file)
            self.beta_uD = torch.FloatTensor(constrain_mat['beta_users']).to(self.device)
            self.beta_iD = torch.FloatTensor(constrain_mat['beta_items']).to(self.device)
            
        if self.config['gamma'] > 0:
            self.ii_topk_neighbors = load_pickle(topk_neighbors_file)
            self.ii_topk_similarity_scores = load_pickle(topk_similarity_file)
            
            topk = config['topk']
            self.ii_topk_neighbors = torch.LongTensor(self.ii_topk_neighbors[:, :topk]).to(self.device)
            self.ii_topk_similarity_scores = torch.FloatTensor(self.ii_topk_similarity_scores[:, :topk]).to(self.device)

    def get_embedding(self):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)
        return x

    def forward(self, batch_data):
        all_idx = torch.tensor(np.arange(self.graph.num_nodes())).to(self.device)
        x = self.emb_table(all_idx).to(self.device)

        src, pos, neg = batch_data
        
        src_emb = x[src]
        pos_emb = x[pos]
        neg_emb = x[neg]
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)

        _pos = pos
        _neg = neg
        
        if self.config['lambda'] > 0:
            beta_pos = self.beta_uD[src] * self.beta_iD[_pos]
            beta_neg = self.beta_uD[src].unsqueeze(1) * self.beta_iD[_neg]
            pos_coe = 1 + self.config['lambda'] * beta_pos 
            neg_coe = 1 + self.config['lambda'] * beta_neg
        else:
            pos_coe = None
            neg_coe = None
        
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, 
            torch.ones(pos_score.shape).to(self.device),
            weight=pos_coe
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, 
            torch.zeros(neg_score.shape).to(self.device),
            weight=neg_coe
        ).mean(dim = -1)
        
        loss_C_O = (pos_loss + self.config['neg_weight'] * neg_loss).sum()
        
        loss = loss_C_O
        
        # loss L_I
        if self.config['gamma'] > 0:
            ii_neighbors = self.ii_topk_neighbors[_pos]
            ii_scores = self.ii_topk_similarity_scores[_pos]

            _ii_neighbors = ii_neighbors
            ii_emb = x[_ii_neighbors]
            
            pos_ii_score = dot_product(src_emb, ii_emb)
            loss_I = -(ii_scores * pos_ii_score.sigmoid().log()).sum()
            
            loss += self.config['gamma'] * loss_I
        
        # L2 regularization loss
        if self.config['l2_reg_weight'] > 0:
            L2_reg_loss = 1/2 * ((src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum())
            if self.config['gamma'] > 0:
                L2_reg_loss += 1/2 * (ii_emb**2).sum()
            
            loss += self.config['l2_reg_weight'] * L2_reg_loss
    
        return loss


class LightGCN(nn.Module):

    def __init__(self, graph, emb_dim, num_layers, use_sparse_emb, load_from_feat, device):
        super().__init__()
        E = graph.edges()
        d1 = (graph.out_degrees(E[0]) + graph.in_degrees(E[0])) / 2.0
        d2 = (graph.out_degrees(E[1]) + graph.in_degrees(E[1])) / 2.0
        edge_weights = (1 / (d1 * d2)).sqrt()
        idx = torch.stack(E)
        num_nodes = graph.num_nodes()
        self.full_adj = torch.sparse_coo_tensor(
            idx, edge_weights, (num_nodes, num_nodes)
        ).to(device)

        if load_from_feat:
            self.emb_table = torch.nn.Embedding.from_pretrained(graph.ndata['feat'], freeze=False, sparse=use_sparse_emb)
        else:
            self.emb_table = torch.nn.Embedding(graph.num_nodes(), emb_dim, sparse=use_sparse_emb)  # initialized from N(0, 1)
        self.nodes_embs = self.emb_table.to(device)

        self.num_gcn_layer = num_layers

        self.opt_param_list = []
        self.opt_param_list.extend(self.emb_table.parameters())

    def forward(self):
        X = self.emb_table.weight

        for _ in range(self.num_gcn_layer):
            X = torch.sparse.mm(self.full_adj, X)
        return X


    def parameters(self):
        return self.opt_param_list



def load_model(graph, model, dataset, device):
    if model == 'pprgo':
        config_file = 'config/pprgo-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))

        model = PPRGo(graph, emb_dim=config['emb_dim'], dataset=dataset, topk=config['topk'], root='dataset/', device=device)

    elif model == 'lightgcn':
        config_file = 'config/lightgcn-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        load_from_feat = False
        num_layers = config['num_layers']
        model = LightGCN(graph, config['emb_dim'], num_layers, config['use_sparse_emb'], load_from_feat, device)

    elif model == 'ultragcn':
        config_file = 'config/ultragcn-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))

        model = UltraGCN(graph, config, dataset, device)

    elif model == 'gin':
        config_file = 'config/gin-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = GIN(graph, config['emb_dim'], config['num_layers'], device)

    elif model == 'gat':
        config_file = 'config/gat-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        model = GAT(graph, config['emb_dim'], config['num_layers'], device)

    elif model == 'sage':
        config_file = 'config/sage-config.yaml'
        config = defaultdict(int)
        config.update(load_yaml(config_file))
        if dataset == 'epinions':
            model = SAGE(graph, config['emb_dim'], config['num_layers'], device)
        else:
            model = SAGE(graph, config['emb_dim'], config['num_layers'], device)
    
    return model.to(device)


