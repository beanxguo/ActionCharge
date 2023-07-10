import numpy as np
from scipy import sparse as sp
import dgl

def preprocess_adj(adj):
    rowsum = np.array((adj.sum(1)))
    d_inv_sqrt=np.power(rowsum,-0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)]=0
    d_mat_inv_spart = np.diag(d_inv_sqrt)
    ans = adj.dot(d_mat_inv_spart).transpose().dot(d_mat_inv_spart)
    return ans


def build_graph(device,vocab_size,vocabe_embeding):
    adj = np.random.rand(vocab_size,vocab_size)
    adj= np.array(preprocess_adj(adj+sp.eye(adj.shape[0])))
    row = []
    col = []
    weight = []

    for i in range(vocab_size):
        for j in range(vocab_size):
            row.append(i)
            col.append(j)
            weight.append(adj[i,j])
    adj = sp.csr_matrix((weight,(row,col)), shape=(vocab_size,vocab_size))
    g = dgl.from_scipy(adj.astype('float32'),eweight_name='edge_weight')
    g.ndata['node_feature']=vocabe_embeding.to('cpu')
    g=g.to(device)
    return g