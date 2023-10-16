import dgl
import torch
import networkx as nx
import time
from tqdm import tqdm



def random_walk_generation(graph, membership, split_edge, aug_size):
    seed_nodes = torch.randperm(graph.num_nodes())[:int(0.5*graph.num_nodes())]

    walk_length = 10
    walks = dgl.sampling.random_walk(graph, seed_nodes, length=walk_length)[0].tolist()

    src, dst = list(), list()
    half_win_size=15
    
    # rnd = np.random.randint(1,  half_win_size+1, dtype=np.int64, size=l)
    for walk in tqdm(walks, desc='Num walks'):
        l = len(walk)
        for i in range(l):
            real_win_size = half_win_size
            left = i - real_win_size
            if left < 0:
                left = 0
                right = i + real_win_size
                if right >= l:
                    # print(1)
                    right = l - 1
                    for j in range(left, right + 1):
                        if walk[i] == walk[j]:
                            continue
                        elif walk[i] < 0 or walk[j] < 0:
                            continue
                        else:
                            if membership[walk[i]] == membership[walk[j]]: 
                                continue
                            else:
                                src.append(walk[i])
                                dst.append(walk[j])

    virtual_graph = dgl.graph((torch.tensor(src), torch.tensor(dst)))
    print(len(src))
    virtual_graph.edata['confidence'] = torch.ones(virtual_graph.num_edges())
    # for p_src, p_dst in tqdm(zip(src, dst)):
    virtual_graph.edata['confidence'][virtual_graph.edge_ids(src, dst)] += 1

    train_src_dst = split_edge['train']['edge']

    intra_size = len(train_src_dst[membership[train_src_dst.t()[0]] == membership[train_src_dst.t()[1]]])
    inter_size = len(train_src_dst[membership[train_src_dst.t()[0]] != membership[train_src_dst.t()[1]]])

    augment_size = int((intra_size - inter_size) * aug_size)
    print('>> Generate {:d} pseudo edges.'.format(augment_size))

    pseudo_edges = torch.stack(virtual_graph.edges()).t()[virtual_graph.edata['confidence'].topk(augment_size)[1]]

    split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], pseudo_edges), dim=0)

    return graph, split_edge




def pseudo_generation(graph, membership, split_edge, add_edge, aug_size, true_aug=1):
    graph.edata['pseudo_tag'] = torch.zeros(graph.num_edges())

    nx_graph = nx.Graph(dgl.to_networkx(graph))

    k_hop_graph = dgl.khop_graph(graph, 2)  # generate the 2 hop graph
    k_hop_graph = dgl.remove_self_loop(k_hop_graph)  # remove the self loops

    # select the cross-links
    k_hop_edges = k_hop_graph.edges()
    if true_aug:
        print(">> True augmentation.")
        all_pseudo_edges = torch.stack(k_hop_edges)[:, membership[k_hop_edges[0]] != membership[k_hop_edges[1]]]
    else:
        print(">> False augmentation.")  # for ablation study only
        all_pseudo_edges = torch.stack(k_hop_edges)[:10000000]

    train_src_dst = split_edge['train']['edge']

    intra_size = len(train_src_dst[membership[train_src_dst.t()[0]] == membership[train_src_dst.t()[1]]])
    inter_size = len(train_src_dst[membership[train_src_dst.t()[0]] != membership[train_src_dst.t()[1]]])

    augment_size = int((intra_size - inter_size) * aug_size)
    print('>> Generate {:d} pseudo edges.'.format(augment_size))

    torch.manual_seed(0)
    pseudo_edges = all_pseudo_edges[:, torch.randperm(len(all_pseudo_edges[0]))[:augment_size*2]]

    strength = torch.ones(len(pseudo_edges[0]))

    # calculate the jaccard coefficient as edge strength
    print('>> Start preprocessing...')
    start_time = time.time()
    strength = nx.jaccard_coefficient(nx_graph, list(map(lambda x: tuple(x), pseudo_edges.t().tolist())))
    strength = torch.tensor(list(map(lambda x: x[-1], strength)))
    print('>> Finished. Use {:.2f} seconds.'.format(time.time() - start_time))
    
    if true_aug:
        top_pseudo_edges = pseudo_edges[:, strength.topk(augment_size, dim=0)[1]]  # common neighbors

    else:
        top_pseudo_edges = pseudo_edges[:, :augment_size]

    if add_edge:
        new_graph = dgl.add_edges(graph, torch.cat((top_pseudo_edges[0], top_pseudo_edges[1]), dim=0), torch.cat((top_pseudo_edges[1], top_pseudo_edges[0]), dim=0), {'pseudo_tag': torch.ones(2*len(top_pseudo_edges[0]))})
        
    else:
        new_graph = graph

    split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], top_pseudo_edges.t()), dim=0)

    return new_graph, split_edge


def random_generation(graph, membership, split_edge, add_edge, aug_size, true_aug=1):
    graph.edata['pseudo_tag'] = torch.zeros(graph.num_edges())

    nx_graph = nx.Graph(dgl.to_networkx(graph))

    k_hop_graph = dgl.khop_graph(graph, 2)  # generate the 2 hop graph
    k_hop_graph = dgl.remove_self_loop(k_hop_graph)  # remove the self loops

    # select the cross-links
    k_hop_edges = k_hop_graph.edges()
    if true_aug:
        print(">> True augmentation.")
        all_pseudo_edges = torch.stack(k_hop_edges)[:, membership[k_hop_edges[0]] != membership[k_hop_edges[1]]]
    else:
        print(">> False augmentation.")  # for ablation study only
        all_pseudo_edges = torch.stack(k_hop_edges)[:10000000]

    train_src_dst = split_edge['train']['edge']

    intra_size = len(train_src_dst[membership[train_src_dst.t()[0]] == membership[train_src_dst.t()[1]]])
    inter_size = len(train_src_dst[membership[train_src_dst.t()[0]] != membership[train_src_dst.t()[1]]])

    augment_size = int((intra_size - inter_size) * aug_size)
    print('>> Generate {:d} pseudo edges.'.format(augment_size))

    torch.manual_seed(0)
    pseudo_edges = all_pseudo_edges[:, torch.randperm(len(all_pseudo_edges[0]))[:augment_size*2]]

    strength = torch.ones(len(pseudo_edges[0]))

    # calculate the jaccard coefficient as edge strength
    print('>> Start preprocessing...')
    start_time = time.time()
    strength = nx.jaccard_coefficient(nx_graph, list(map(lambda x: tuple(x), pseudo_edges.t().tolist())))
    strength = torch.tensor(list(map(lambda x: x[-1], strength)))
    print('>> Finished. Use {:.2f} seconds.'.format(time.time() - start_time))
    
    top_pseudo_edges = pseudo_edges[:, :augment_size]

    if add_edge:
        new_graph = dgl.add_edges(graph, torch.cat((top_pseudo_edges[0], top_pseudo_edges[1]), dim=0), torch.cat((top_pseudo_edges[1], top_pseudo_edges[0]), dim=0), {'pseudo_tag': torch.ones(2*len(top_pseudo_edges[0]))})
        
    else:
        new_graph = graph

    split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], top_pseudo_edges.t()), dim=0)

    return new_graph, split_edge





