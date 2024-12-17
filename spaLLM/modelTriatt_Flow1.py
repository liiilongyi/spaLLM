import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def init_weights(*params):
    """Initialize weights with Xavier uniform distribution."""
    for param in params:
        torch.nn.init.xavier_uniform_(param)

class DeepEncoder(Module):
    """Modality-specific GNN encoder."""
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super().__init__()
        self.dropout = dropout
        self.act = act
        self.hidden_dim = out_feat * 2

        self.weights = [
            Parameter(torch.FloatTensor(in_feat, self.hidden_dim)),
            Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim)),
            Parameter(torch.FloatTensor(self.hidden_dim, out_feat))
        ]
        init_weights(*self.weights)

    def forward(self, feat, adj):
        x = self._apply_layer(feat, adj, self.weights[0])
        x = self._apply_layer(x, adj, self.weights[1])
        x = torch.spmm(adj, torch.mm(x, self.weights[2]))
        return x

    def _apply_layer(self, x, adj, weight):
        x = torch.spmm(adj, torch.mm(x, weight))
        x = self.act(x)
        return F.dropout(x, self.dropout, training=self.training)

class CellEmbedding(Module):
    """Modality-specific cell embedding encoder/decoder."""
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        init_weights(self.weight)

    def forward(self, feat, adj):
        return torch.spmm(adj, torch.mm(feat, self.weight))

class AttentionLayer(Module):
    """Generic Attention Layer."""
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        init_weights(self.w_omega, self.u_omega)

    def forward(self, *embeddings):
        emb_stack = torch.cat([torch.unsqueeze(emb, dim=1) for emb in embeddings], dim=1)
        v = F.tanh(torch.matmul(emb_stack, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(vu.squeeze() + 1e-6, dim=1)
        emb_combined = torch.matmul(emb_stack.transpose(1, 2), alpha.unsqueeze(-1)).squeeze()
        return emb_combined, alpha

class EncodingNetwork(Module):
    """Encoding network with modality-specific encoders, decoders, and attention layers."""
    def __init__(self, dim_in_omics1, dim_out_omics1, dim_in_omics2, dim_out_omics2):
        super().__init__()
        self.encoder_embedding = CellEmbedding(512, 64)
        self.decoder_embedding = CellEmbedding(64, 512)

        self.encoder_omics1 = DeepEncoder(dim_in_omics1, dim_out_omics1)
        self.decoder_omics1 = DeepEncoder(dim_out_omics1, dim_in_omics1)
        self.encoder_omics2 = DeepEncoder(dim_in_omics2, dim_out_omics2)
        self.decoder_omics2 = DeepEncoder(dim_out_omics2, dim_in_omics2)

        self.atten_feature1 = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_feature2 = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_feature = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_omics2 = AttentionLayer(dim_out_omics2, dim_out_omics2)
        self.atten_cross = AttentionLayer(dim_out_omics1, dim_out_omics2)

    def forward(self, f_omics1, f_omics2, adj_spa1, adj_fea1, adj_spa2, adj_fea2, cell_emb, adj_emb):
        emb_spa = self.encoder_embedding(cell_emb, adj_spa1)
        emb_fea = self.encoder_embedding(cell_emb, adj_emb)

        emb_latent_spa1 = self.encoder_omics1(f_omics1, adj_spa1)
        emb_latent_spa2 = self.encoder_omics2(f_omics2, adj_spa2)
        emb_latent_fea1 = self.encoder_omics1(f_omics1, adj_fea1)
        emb_latent_fea2 = self.encoder_omics2(f_omics2, adj_fea2)

        emb_att1, alpha_att1 = self.atten_feature1(emb_spa, emb_latent_spa1)
        emb_att2, alpha_att2 = self.atten_feature2(emb_fea, emb_latent_fea1)
        emb_latent_omics1, alpha_att_omics1 = self.atten_feature(emb_att1, emb_att2)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spa2, emb_latent_fea2)

        emb_combined, alpha = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        emb_recon1 = self.decoder_omics1(emb_combined, adj_spa1)
        emb_recon2 = self.decoder_omics2(emb_combined, adj_spa2)
        emb_recon_spa = self.decoder_embedding(emb_spa, adj_spa1)
        emb_recon_fea = self.decoder_embedding(emb_fea, adj_emb)

        emb_cross1 = self.encoder_omics2(self.decoder_omics2(emb_latent_omics1, adj_spa2), adj_spa2)
        emb_cross2 = self.encoder_omics1(self.decoder_omics1(emb_latent_omics2, adj_spa1), adj_spa1)

        return {
            'emb_latent_omics1': emb_latent_omics1, 'emb_latent_omics2': emb_latent_omics2,
            'emb_combined': emb_combined, 'emb_recon_omics1': emb_recon1, 'emb_recon_omics2': emb_recon2,
            'emb_cross1': emb_cross1, 'emb_cross2': emb_cross2,
            'alpha_att1': alpha_att1, 'alpha_att2': alpha_att2, 'alpha_omics1': alpha_att_omics1,
            'alpha_omics2': alpha_omics2, 'alpha': alpha, 'emb_recon_spa': emb_recon_spa,
            'emb_recon_fea': emb_recon_fea
        }
