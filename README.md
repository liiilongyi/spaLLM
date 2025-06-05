# spaLLM

spaLLM: Enhancing Spatial Domain Analysis in Multi-omics Data through Large Language Model Integration

## Abstract

Spatial multi-omics technologies provide valuable data on gene expression from various omics in the same tissue section while preserving spatial information. However, deciphering spatial domains within spatial omics data remains challenging due to the sparse gene expression. We propose spaLLM, the first multi-omics spatial domain analysis method that integrates large language models to enhance data representation. Our method combines a pre-trained single-cell language model (scGPT) with graph neural networks and multi-view attention mechanisms to compensate for limited gene expression information in spatial omics while improving sensitivity and resolution within modalities. spaLLM processes multiple spatial modalities, including RNA, chromatin, and protein data, potentially adapting to emerging technologies and accommodating additional modalities. Benchmarking against eight state-of-the-art methods across four different datasets and platforms demonstrates that our model consistently outperforms other advanced methods across multiple supervised evaluation metrics.


## Installation Requirements

Install the following dependencies via `pip install -r requirements.txt`, or manually using the versions below:

```txt
scanpy==1.9.1
anndata==0.8.0
pandas==1.4.2
numpy==1.22.3
torch==1.12.0+cu113
matplotlib==3.4.2
seaborn==0.11.2
scikit-learn==1.1.2
