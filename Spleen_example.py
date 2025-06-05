import os
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
from spaLLM.preprocess import fix_seed, clr_normalize_each_cell, pca, construct_neighbor_graph
from spaLLM.spaLLM_util import Train_spaLLM
from spaLLM.utils import clustering


def set_device():
    """Set device to GPU if available."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_r_home():
    """Set R_HOME environment variable."""
    os.environ['R_HOME'] = 'C:/PROGRA~1/R/R-44~1.1'


def preprocess_omics_data(adata_rna, adata_protein, top_genes=3000, pca_comps=200):
    """Preprocess RNA and Protein omics data."""
    # RNA Preprocessing
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=top_genes)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]
    adata_rna.obsm['feat'] = pca(adata_rna_high, n_comps=pca_comps)

    # Protein Preprocessing
    adata_protein = clr_normalize_each_cell(adata_protein)
    sc.pp.scale(adata_protein)
    adata_protein.obsm['feat'] = pca(adata_protein, n_comps=adata_protein.n_vars - 1)
    return adata_rna, adata_protein


def train_model(adata_rna, adata_protein, embedding, datatype, device):
    """Train spaLLM model and return outputs."""
    data = construct_neighbor_graph(adata_rna, adata_protein, datatype=datatype)
    data['adj_emb'] = kneighbors_graph(embedding, 20, mode="connectivity", metric="correlation", include_self=False)
    model = Train_spaLLM(data, datatype=datatype, device=device, embedding=embedding)
    return model.train(epochs=600)


def save_umap_visualization(adata, save_path):
    """Save UMAP visualization plots."""
    fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
    sc.pp.neighbors(adata, use_rep='spaLLM', n_neighbors=10)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color='spaLLM', ax=ax_list[0], title='spaLLM', s=20, show=False)
    sc.pl.embedding(adata, basis='spatial', color='spaLLM', ax=ax_list[1], title='spaLLM', s=25, show=False)
    plt.tight_layout(w_pad=0.3)
    plt.savefig(save_path, format='jpeg', bbox_inches='tight', dpi=300)
    plt.show()


def save_violin_plot(adata, alpha_key, title, save_path):
    """Save violin plot for weight visualization."""
    plt.rcParams['figure.figsize'] = (8, 5)
    df = pd.DataFrame({
        'embedding': adata.obsm[alpha_key][:, 0],
        'omic': adata.obsm[alpha_key][:, 1],
        'label': adata.obs['spaLLM'].values
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
    set_r_home()
    device = set_device()
    fix_seed(2024)

    # Load data
    file_fold = '<YOUR_DATASET_FILE_FOLDER>'
    adata_rna = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
    adata_protein = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
    embedding = np.load(file_fold + 'RNA_embedding.npy')

    adata_rna.var_names_make_unique()
    adata_protein.var_names_make_unique()

    # Preprocess data
    adata_rna, adata_protein = preprocess_omics_data(adata_rna, adata_protein)

    # Train spaLLM model
    output = train_model(adata_rna, adata_protein, embedding, datatype='SPOTS', device=device)

    # Save outputs to adata
    adata = adata_rna.copy()
    for key, value in output.items():
        adata.obsm[key] = value
    clustering(adata, key='spaLLM', add_key='spaLLM', n_clusters=7, method='mclust', use_pca=True)

    save_path = '<YOUR_SAVE_PATH>'
    # UMAP visualization
    save_umap_visualization(adata, save_path + 'umap.jpeg')

    # Save violin plots
    save_violin_plot(adata, 'alpha_att1', 'embedding_spa vs omic', save_path + 'alpha_att1.jpeg')
    save_violin_plot(adata, 'alpha_att2', 'embedding_fea vs omic', save_path + 'alpha_att2.jpeg')


if __name__ == "__main__":
    main()
