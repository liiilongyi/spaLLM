import torch
from tqdm import tqdm
import torch.nn.functional as F
from .modelTriatt_Flow1 import EncodingNetwork
from .preprocess import adjacent_matrix_preprocessing
from .utils import add_gaussian_noise
import random
import matplotlib.pyplot as plt


class Train_spaLLM:
    def __init__(self, data, embedding, datatype='10x', device=torch.device('cpu'),
                 random_seed=2024, learning_rate=0.0001, weight_decay=0.0, epochs=600,
                 dim_input=3000, dim_output=64, weight_factors=None):
        self.device = device
        self.data = data.copy()
        self.embedding = torch.from_numpy(embedding).to(device)
        self.datatype = datatype
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors or [5, 5, 1, 10, 10, 10]

        self._init_adj_and_features()
        self.loss_history = []
        self._adjust_hyperparameters()

    def _init_adj_and_features(self):
        """Initialize adjacency matrices and input features."""
        adj = adjacent_matrix_preprocessing(self.data['adata_omics1'], self.data['adata_omics2'], self.data['adj_emb'])
        self.adj_spatial_omics1 = adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = adj['adj_feature_omics2'].to(self.device)
        self.adj_emb = adj['adj_emb'].to(self.device)

        self.features_omics1 = torch.FloatTensor(self.data['adata_omics1'].obsm['feat']).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.data['adata_omics2'].obsm['feat']).to(self.device)
        self.dim_input1, self.dim_input2 = self.features_omics1.shape[1], self.features_omics2.shape[1]

    def _adjust_hyperparameters(self):
        """Adjust hyperparameters based on data type."""
        if self.datatype == 'SPOTS':
            self.epochs, self.weight_factors = 600, [1, 5, 1, 1, 5, 5]
        elif self.datatype == '10x':
            self.epochs, self.weight_factors = 200, [5, 5, 1, 10, 10, 10]
        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs, self.weight_factors = 1600, [1, 5, 1, 1, 10, 10]

    def _add_noise(self):
        """Apply Gaussian noise to features and embeddings."""
        features_omics1_noisy = add_gaussian_noise(self.features_omics1, mean=0, std=0.1)
        embedding_noisy = add_gaussian_noise(self.embedding, mean=0, std=0.01)
        return features_omics1_noisy, embedding_noisy

    def _calculate_losses(self, results):
        """Calculate reconstruction and correspondence losses."""
        loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
        loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
        loss_rec_es = F.mse_loss(self.embedding, results['emb_recon_spa'])
        loss_rec_ef = F.mse_loss(self.embedding, results['emb_recon_fea'])
        loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
        loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

        loss = (self.weight_factors[0] * loss_recon_omics1 +
                self.weight_factors[1] * loss_recon_omics2 +
                self.weight_factors[2] * loss_corr_omics1 +
                self.weight_factors[3] * loss_corr_omics2 +
                self.weight_factors[4] * loss_rec_es +
                self.weight_factors[5] * loss_rec_ef)
        return loss

    def train(self, epochs=None):
        epochs = epochs or self.epochs
        self.model = EncodingNetwork(self.dim_input1, self.dim_output, self.dim_input2, self.dim_output).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        print(f"Training for {epochs} epochs...")
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            if random.random() < 0.5:
                features_omics1, embedding = self._add_noise()
            else:
                features_omics1, embedding = self.features_omics1, self.embedding

            results = self.model(features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1,
                                 self.adj_spatial_omics2, self.adj_feature_omics2, embedding, self.adj_emb)
            loss = self._calculate_losses(results)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

        print("Training completed!")
        return self._evaluate_model()

    def _evaluate_model(self):
        """Evaluate the model and return output embeddings."""
        self.model.eval()
        with torch.no_grad():
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2,
                                 self.embedding, self.adj_emb)

        return {
            'emb_latent_omics1': F.normalize(results['emb_latent_omics1'], p=2).cpu().numpy(),
            'emb_latent_omics2': F.normalize(results['emb_latent_omics2'], p=2).cpu().numpy(),
            'spaLLM': F.normalize(results['emb_latent_combined'], p=2).cpu().numpy(),
            'alpha_omics1': results['alpha_omics1'].cpu().numpy(),
            'alpha_omics2': results['alpha_omics2'].cpu().numpy(),
            'alpha': results['alpha'].cpu().numpy(),
            'alpha_att1': results['alpha_att1'].cpu().numpy(),
            'alpha_att2': results['alpha_att2'].cpu().numpy()
        }

    def plot_loss(self):
        """Plot the training loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()
