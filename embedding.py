import numpy as np
import torch
import scanpy as sc
from scgpt.tasks.cell_emb import embed_data

file_fold = '<YOUR_DATASET_FILE_FOLDER>'

adata = sc.read_h5ad(file_fold)
print(adata)
print(adata.var['gene_ids'])

adata.var['gene_ids'] = adata.var.index
adata.var['gene_ids'] = adata.var['gene_ids'].str.upper()

model_dir = '<scGPT_human_FILE_FOLDER>'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#adata = embed_data(adata_or_file=adata, model_dir=model_dir, gene_col="gene_ids", max_length=1200, batch_size=64, obs_to_save=None, device=device, use_fast_transformer=False, return_new_adata=False)
adata = embed_data(adata_or_file=adata, model_dir=model_dir, gene_col="gene_names", max_length=1200, batch_size=64, obs_to_save=None, device=device, use_fast_transformer=False, return_new_adata=False)

cell_embeddings = adata.obsm["X_scGPT"]

print(cell_embeddings)
print(cell_embeddings.shape)
print(cell_embeddings.dtype)


np.save('<YOUR_SAVE_PATH>', cell_embeddings)

print("saved") 


