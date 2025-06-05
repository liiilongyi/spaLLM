import os
import numpy as np
import torch
import scanpy as sc
import spaLLM
from spaLLM.preprocess import fix_seed, clr_normalize_each_cell, pca, lsi, construct_neighbor_graph
from spaLLM.spaLLM_util import Train_spaLLM
from spaLLM.utils import clustering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
    v_measure_score, homogeneity_score


def set_device():
    """Set device to GPU if available."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_r_home():
    """Set R_HOME environment variable."""
    os.environ['R_HOME'] = 'C:/PROGRA~1/R/R-44~1.1'


def calculate_metrics(label, pred):
    """Calculate clustering evaluation metrics."""
    valid_indices = label != -1
    return {
        'ARI': adjusted_rand_score(label[valid_indices], pred[valid_indices]),
        'NMI': normalized_mutual_info_score(label[valid_indices], pred[valid_indices]),
        'AMI': adjusted_mutual_info_score(label[valid_indices], pred[valid_indices]),
        'V-measure': v_measure_score(label[valid_indices], pred[valid_indices]),
        'Homogeneity': homogeneity_score(label[valid_indices], pred[valid_indices]),
    }


def preprocess_rna(adata, n_top_genes=3000, n_comps=200):
    """Preprocess RNA data."""
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    adata_high = adata[:, adata.var['highly_variable']]
    adata.obsm['feat'] = pca(adata_high, n_comps=n_comps)


def preprocess_atac(adata, adata_rna, n_components=201):
    """Preprocess ATAC data."""
    adata = adata[adata_rna.obs_names].copy()
    if 'X_lsi' not in adata.obsm:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
        lsi(adata, use_highly_variable=False, n_components=n_components)
    adata.obsm['feat'] = adata.obsm['X_lsi']
    return adata


def plot_violin(adata, alpha_key, title, save_path):
    """Plot violin plots for embedding weights."""
    plt.rcParams['figure.figsize'] = (8, 5)
    df = pd.DataFrame({
        'embedding': adata.obsm[alpha_key][:, 0],
        'omic': adata.obsm[alpha_key][:, 1],
        'label': adata.obs['spaLLM']
    })
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_spaLLM', 'Modality', 'Weight value']
    ax = sns.violinplot(data=df, x='label_spaLLM', y='Weight value', hue='Modality', split=True, inner='quart',
                        linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('spaLLM label')
    ax.legend(bbox_to_anchor=(1.4, 1.01), loc='upper right')
    plt.tight_layout(w_pad=0.05)
    plt.savefig(save_path, format='jpeg', bbox_inches='tight', dpi=300)
    plt.show()


def main():
    set_r_home()  # Set R_HOME path
    device = set_device()
    fix_seed(2024)

    # Load data
    file_fold = 'D:/lilongyi/spatial dataset/MISAR-seq/MISAR_seq_mouse_E15_brain_data/'
    file_fold = '<YOUR_DATASET_FILE_FOLDER>'
    adata_omics1 = sc.read_h5ad(file_fold + 'mRNA_data.h5ad')
    adata_omics2 = sc.read_h5ad(file_fold + 'ATAC_data.h5ad')
    embedding = np.load(file_fold + 'mRNA_data_embedding.npy')

    label = np.ravel(adata_omics1.obsm['annotation'])

    # Preprocess
    preprocess_rna(adata_omics1)
    adata_omics2 = preprocess_atac(adata_omics2, adata_omics1)

    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype='Spatial-epigenome-transcriptome')
    data['adj_emb'] = kneighbors_graph(embedding, 20, mode="connectivity", metric="correlation", include_self=False)

    # Train spaLLM
    model = Train_spaLLM(data, datatype='Spatial-epigenome-transcriptome', device=device, embedding=embedding)
    output = model.train(epochs=800)

    # Save results
    adata = adata_omics1.copy()
    for key in output.keys():
        adata.obsm[key] = output[key]

    clustering(adata, key='spaLLM', add_key='spaLLM', n_clusters=7, method='mclust', use_pca=True)

    pred = np.ravel(adata.obs['spaLLM'].to_numpy())
    metrics = calculate_metrics(label, pred)
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    # Plot
    save_path = '<YOUR_SAVE_PATH>'
    plot_violin(adata, 'alpha_att1', 'embedding_spa vs omic', save_path + 'alpha_att1.jpeg')
    plot_violin(adata, 'alpha_att2', 'embedding_fea vs omic', save_path + 'alpha_att2.jpeg')

    # UMAP visualization
    sc.pp.neighbors(adata, use_rep='spaLLM', n_neighbors=10)
    sc.tl.umap(adata)
    fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
    sc.pl.umap(adata, color='spaLLM', ax=ax_list[0], title='spaLLM', s=20, show=False)
    sc.pl.embedding(adata, basis='spatial', color='spaLLM', ax=ax_list[1], title='spaLLM', s=25, show=False)
    plt.tight_layout(w_pad=0.3)
    plt.show()


if __name__ == "__main__":
    main()
