# HiPHD
📋HiPHD: Hierarchical Classification for Protein Remote Homology Detection using Graph Neural Networks and Language Models

## 📘 Abstract
&nbsp;&nbsp;&nbsp;&nbsp; Protein homology detection is crucial in various biological tasks, such as protein function annotation and structure prediction. Computational methods have been developed to improve protein homology detection. However, most existing methods are sequence-based and computationally costly due to their dependence on database searching and protein alignment. Proteins with remote homology usually share similar structures and less than 25\% sequence identity; the performance of entirely sequence-based methods is limited. 
&nbsp;&nbsp;&nbsp;&nbsp; This study proposes an alignment-free method based on a sequence-structure hybrid model and a hierarchical multi-task classification framework to detect protein homology. The experimental results show that integrating sequence and structural information can improve protein homology detection, and the proposed method outperforms existing methods in terms of both efficiency and accuracy in SCOPe and CATH databases. It's anticipated the proposed method will become a valuable tool for protein homology detection and representation learning.


## 🧬 Model Structure
<!-- &nbsp;&nbsp;&nbsp;&nbsp; iMFP-LG consists of two modules: peptide representation module and a graph classification module. The peptide sequences are first fed into the pLM to extract high-quality representations, which are then transformed as node features by node feature encoders. The GAT is performed to fine-tune node features by learning the relationship of nodes. Finally, the updated node features are utilized to determine whether the peptides have corresponding function or not through node classifiers. 
<div align=center><img src=img/framework.png></div> -->

## 🚀 Train

## 🧐 Prediction

## ✏️ Citation
If you use this code or our model for your publication, please cite the original paper: