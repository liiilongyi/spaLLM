import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from .preprocess import pca

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def convert_csv_to_h5ad(input_path, output_path):
    """Convert a CSV file to AnnData h5ad format."""
    data = pd.read_csv(input_path, index_col=0)
    obs_data = data.iloc[:, :2]
    obs_data.columns = ['barcode', 'assigned_cluster']

    expr_data = data.iloc[:, 2:].values.astype('float32')
    gene_names = data.columns[2:]

    adata = sc.AnnData(X=expr_data)
    adata.obs = obs_data
    adata.var['gene_names'] = gene_names
    adata.write(output_path)
    print(f"Data saved to {output_path}")

def convert_tsv_to_csv(tsv_path, csv_path):
    """Convert a TSV file to CSV format."""
    data = pd.read_csv(tsv_path, sep='\t')
    data.to_csv(csv_path, index=False)
    print(f"TSV file converted to CSV and saved at {csv_path}")

def create_h5ad_from_sparse_csv(input_csv, output_h5ad):
    """Create an AnnData h5ad file from a sparse matrix CSV file."""
    import scipy.sparse as sp
    import anndata as ad

    data = pd.read_csv(input_csv, index_col=0)
    expression_matrix = sp.csr_matrix(data.values)

    spatial_coords = data.index.str.split('x').tolist()
    spatial_coords = np.array(spatial_coords, dtype=int)

    adata = ad.AnnData(X=expression_matrix)
    adata.obsm['spatial'] = spatial_coords
    adata.var['gene_names'] = data.columns.values

    adata.write(output_h5ad)
    print(f"Sparse AnnData file saved to {output_h5ad}")

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """Perform clustering using the R `mclust` algorithm."""
    robjects.r.library("mclust")
    robjects.r['set.seed'](random_seed)
    res = robjects.r['Mclust'](rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    adata.obs['mclust'] = np.array(res[-2]).astype('int')
    return adata

def clustering(adata, n_clusters=7, key='emb', add_key='spaLLM', method='mclust', **kwargs):
    """Spatial clustering using `mclust`, `leiden`, or `louvain`."""
    use_pca = kwargs.get('use_pca', False)
    n_comps = kwargs.get('n_comps', 20)
    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
        key = key + '_pca'

    if method == 'mclust':
        mclust_R(adata, num_cluster=n_clusters, used_obsm=key)
        adata.obs[add_key] = adata.obs['mclust']
    elif method in ['leiden', 'louvain']:
        res = search_res(adata, n_clusters, method=method, use_rep=key, **kwargs)
        clustering_func = sc.tl.leiden if method == 'leiden' else sc.tl.louvain
        clustering_func(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs[method]

def add_noise_by_zeroing(matrix, zero_prob):
    """Add noise by zeroing random elements in the matrix."""
    noise_mask = torch.bernoulli((1 - zero_prob) * torch.ones_like(matrix)).to(matrix.device)
    return matrix * noise_mask

def add_noise_by_zeroing_columns(matrix, zero_prob=0.1):
    """Add noise by zeroing entire random columns of the matrix."""
    zero_mask = torch.bernoulli((1 - zero_prob) * torch.ones(matrix.size(1))).to(matrix.device)
    return matrix * zero_mask.unsqueeze(0).expand_as(matrix)

def add_gaussian_noise(matrix, mean=0.0, std=0.001):
    """Add Gaussian noise to the matrix."""
    noise = torch.normal(mean=mean, std=std, size=matrix.size()).to(matrix.device)
    return matrix + noise

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    """Search for resolution to achieve target cluster count using `leiden` or `louvain`."""
    print('Searching resolution...')
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in np.arange(start, end, increment):
        res = round(res, 3)
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            clusters = adata.obs['leiden']
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            clusters = adata.obs['louvain']
        count_unique = clusters.nunique()
        print(f'resolution={res}, cluster number={count_unique}')
        if count_unique == n_clusters:
            return res
    raise ValueError("Resolution not found. Please try a larger range or smaller step size.")
