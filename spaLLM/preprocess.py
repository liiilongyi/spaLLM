import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import sklearn
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from torch.backends import cudnn
from typing import Optional
import anndata as ad

def fix_seed(seed: int):
    """Fix random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3):
    """Construct spatial and feature neighbor graphs."""
    if datatype == 'Spatial-epigenome-transcriptome':
        n_neighbors = 6

    def _construct_spatial_graph(adata):
        cell_position = adata.obsm['spatial']
        return construct_graph_by_coordinate(cell_position, n_neighbors)

    adata_omics1.uns['adj_spatial'] = _construct_spatial_graph(adata_omics1)
    adata_omics2.uns['adj_spatial'] = _construct_spatial_graph(adata_omics2)

    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = construct_graph_by_feature(
        adata_omics1, adata_omics2
    )
    return {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

def pca(adata: ad.AnnData, use_reps=None, n_comps=10):
    """Perform PCA for dimensionality reduction."""
    pca_model = PCA(n_components=n_comps)
    data = adata.obsm[use_reps] if use_reps else adata.X
    data = data.toarray() if sp.issparse(data) else data
    return pca_model.fit_transform(data)

def clr_normalize_each_cell(adata: ad.AnnData, inplace=True):
    """Normalize count vector for each cell using CLR normalization."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    adata.X = np.apply_along_axis(
        seurat_clr, 1, adata.X.A if sp.issparse(adata.X) else np.array(adata.X)
    )
    return adata

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="correlation"):
    """Construct feature neighbor graphs based on expression profiles."""
    graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=False)
    graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=False)
    return graph_omics1, graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    """Construct spatial graph based on spatial coordinates."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    return pd.DataFrame({'x': x, 'y': y, 'value': 1})

def transform_adjacent_matrix(adj_df):
    """Transform adjacency dataframe into sparse matrix."""
    n_spot = adj_df['x'].max() + 1
    return coo_matrix((adj_df['value'], (adj_df['x'], adj_df['y'])), shape=(n_spot, n_spot))

def preprocess_graph(adj):
    """Normalize adjacency matrix for GNN input."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    degree_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    return sparse_mx_to_torch_sparse_tensor(adj.dot(degree_inv_sqrt).T.dot(degree_inv_sqrt))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return torch.sparse.FloatTensor(indices, torch.from_numpy(sparse_mx.data), torch.Size(sparse_mx.shape))

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adj_emb):
    """Preprocess spatial and feature adjacency matrices for GNNs."""
    def _process_adj(adj):
        adj = adj.toarray() + adj.toarray().T
        adj = np.where(adj > 1, 1, adj)
        return preprocess_graph(adj)

    adj_spatial_omics1 = _process_adj(transform_adjacent_matrix(adata_omics1.uns['adj_spatial']))
    adj_spatial_omics2 = _process_adj(transform_adjacent_matrix(adata_omics2.uns['adj_spatial']))
    adj_emb = _process_adj(adj_emb)

    def _process_feature_adj(adj):
        adj = adj + adj.T
        return preprocess_graph(np.where(adj > 1, 1, adj))

    adj_feature_omics1 = _process_feature_adj(torch.FloatTensor(adata_omics1.obsm['adj_feature'].toarray()))
    adj_feature_omics2 = _process_feature_adj(torch.FloatTensor(adata_omics2.obsm['adj_feature'].toarray()))

    return {
        'adj_spatial_omics1': adj_spatial_omics1,
        'adj_spatial_omics2': adj_spatial_omics2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
        'adj_emb': adj_emb
    }

def lsi(adata: ad.AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs):
    """LSI analysis (Seurat v3 approach)."""
    use_highly_variable = use_highly_variable if use_highly_variable is not None else "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(tfidf(adata_use.X))
    X_lsi = sklearn.utils.extmath.randomized_svd(np.log1p(X_norm * 1e4), n_components, **kwargs)[0]
    adata.obsm["X_lsi"] = (X_lsi - X_lsi.mean(axis=1, keepdims=True)) / X_lsi.std(axis=1, ddof=1, keepdims=True)

def tfidf(X):
    """TF-IDF normalization following Seurat v3 approach."""
    idf = X.shape[0] / X.sum(axis=0)
    tf = X.multiply(1 / X.sum(axis=1)) if sp.issparse(X) else X / X.sum(axis=1, keepdims=True)
    return tf.multiply(idf) if sp.issparse(X) else tf * idf
