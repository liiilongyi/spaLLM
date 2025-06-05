import os
import numpy as np
import torch
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score, homogeneity_score
from spaLLM.preprocess import fix_seed, clr_normalize_each_cell, pca, construct_neighbor_graph
from spaLLM.spaLLM_util import Train_spaLLM
from spaLLM.utils import clustering


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


def preprocess_omics(adata_rna, adata_protein, top_genes=3000):
    """Preprocess RNA and Protein data."""
    # RNA preprocessing
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=top_genes)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]
    adata_rna.obsm['feat'] = pca(adata_rna_high, n_comps=adata_protein.n_vars - 1)

    # Protein preprocessing
    adata_protein = clr_normalize_each_cell(adata_protein)
    sc.pp.scale(adata_protein)
    adata_protein.obsm['feat'] = pca(adata_protein, n_comps=adata_protein.n_vars - 1)
    return adata_rna, adata_protein


def plot_violin(adata, alpha_key, title, save_path):
    """Plot violin plot for weights."""
    plt.rcParams['figure.figsize'] = (8, 5)
    df = pd.DataFrame({
        'embedding': adata.obsm[alpha_key][:, 0],
        'omic': adata.obsm[alpha_key][:, 1],
        'label': adata.obs['spaLLM']
    })
    df = df.set_index('label').stack().reset_index()
    df.columns = ['label_spaFDG', 'Modality', 'Weight value']
    ax = sns.violinplot(data=df, x='label_spaFDG', y='Weight value', hue='Modality', split=True, inner='quart',
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
    file_fold = '<YOUR_DATASET_FILE_FOLDER>'
    adata_rna = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
    adata_protein = sc.read_h5ad(file_fold + 'adata_ADT.h5ad')
    embedding = np.load(file_fold + 'adata2_RNA_embedding.npy')
    annotation = pd.read_csv(file_fold + 'annotation.csv')
    label = annotation['manual-anno'].astype('category').cat.codes.to_numpy()

    # Preprocess data
    adata_rna, adata_protein = preprocess_omics(adata_rna, adata_protein)

    # Construct neighbor graph
    data = construct_neighbor_graph(adata_rna, adata_protein, datatype='10x')
    data['adj_emb'] = kneighbors_graph(embedding, 20, mode="connectivity", metric="correlation", include_self=False)

    # Train spaLLM
    model = Train_spaLLM(data, datatype='10x', device=device, embedding=embedding)
    output = model.train(epochs=800)

    # Save results
    adata = adata_rna.copy()
    for key, value in output.items():
        adata.obsm[key] = value

    clustering(adata, key='spaLLM', add_key='spaLLM', n_clusters=6, method='mclust', use_pca=True)

    # Evaluate metrics
    pred = np.ravel(adata.obs['spaLLM'].to_numpy())
    metrics = calculate_metrics(label, pred)
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    # Visualization
    sc.pp.neighbors(adata, use_rep='spaLLM', n_neighbors=10)
    sc.tl.umap(adata)
    fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))
    sc.pl.umap(adata, color='spaLLM', ax=ax_list[0], title='spaLLM', s=30, show=False)
    sc.pl.embedding(adata, basis='spatial', color='spaLLM', ax=ax_list[1], title='spaLLM', s=25, show=False)
    plt.tight_layout(w_pad=0.3)
    plt.show()

    save_path = '<YOUR_SAVE_PATH>'
    plot_violin(adata, 'alpha_att1', 'embedding_spa vs omic', save_path + 'alpha_att1.jpeg')
    plot_violin(adata, 'alpha_att2', 'embedding_fea vs omic', save_path + 'alpha_att2.jpeg')


if __name__ == "__main__":
    main()
