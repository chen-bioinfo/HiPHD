# HiPHD
📋HiPHD: Hierarchical Classification for Protein Remote Homology Detection using Graph Neural Networks and Language Models

## 📘 Abstract
&nbsp;&nbsp;&nbsp;&nbsp; Protein remote homology detection is crucial in various biological tasks, such as protein function annotation and structure prediction. Computational methods have been developed to improve efficiency and accuracy of protein homology detection. However, most existing methods are sequence-based and computationally costly due to their dependence on database searching and protein alignment. Proteins with remote homology usually share similar structures and low sequence identity, resulting limited performance of sequence-based methods. 


&nbsp;&nbsp;&nbsp;&nbsp; This study introduces HiPHD, a hierarchical classification framework for protein remote homology detection. HiPHD fuses protein sequential information embedded by protein language models and structural information encoded by graph neural networks, effectively combining spatial and chemical features.
Experimental results demonstrate that HiPHD outperforms existing methods in terms of accuracy at all hierarchical levels in both SCOPe and CATH databases. It's anticipated HiPHD will become a valuable tool for protein homology detection and representation learning.\\



## 🧬 Model Structure
&nbsp;&nbsp;&nbsp;&nbsp;The protein structural information is used to construct a graph neural network, whose nodes are initialized by the embedding of each amino acid obtained from a protein language model. Subsequently, the structural representation and sequential representation are concatenated to form the representation of the target protein. Afterwards, the target protein representation is fed into the hierarchical classifier, which treats classification of each level as different tasks and reuse the results to predict next level.
<div align=center><img src=img/framework.png></div>

## 🧭 Installation
Create conda environment and then switch to the environment.
```
conda create -n hiphd python=3.8
conda activate hiphd
```

Install required package
```
conda install --file requirements.txt
```
## 🚀 Train
```
python downstream.py -c config/scope/hbond_plm.yaml --gpus [0,1]
```
## 🧐 Prediction
```
python eval.py
```
## 💡 Acknowledgement
This codebase is based on [TorchDrug](https://github.com/DeepGraphLearning/torchdrug) and [GearNet](https://github.com/DeepGraphLearning/GearNet). Thanks for their work!
 
## ✏️ Citation
If you use this code or our model for your publication, please cite the original paper:
