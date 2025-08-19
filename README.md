# Model-Name

Context-Aware BERT-GAT Approach for Cross-Architecture Binary Code Similarity Detection

## Overview

Model-Name is a research project that introduces a **context-based method for binary code similarity detection across CPU architectures**. The goal is to address the challenges caused by instruction set differences and compiler diversity, which often hinder accurate cross-architecture analysis.

Our approach combines **instruction normalization**, **BERT-based semantic modeling**, and **Graph Attention Networks (GATs)** to effectively capture both semantic and structural context. This enables accurate similarity detection of binary functions compiled under heterogeneous architectures such as **x86** and **ARM**.

## Key Features

- **Cross-Architecture Detection**: Supports binary function similarity analysis across different CPU architectures.
- **Instruction Normalization**: Maps semantically equivalent instructions (e.g., MOV in x86 and LDR in ARM) into a unified representation.
- **Multi-Task Training**: Integrates Masked Language Modeling (MLM), Next Sentence Prediction (NSP), and Contrastive Learning for robust semantic embeddings.
- **Structural Context Modeling**: Employs Graph Attention Networks to encode control-flow graphs, reducing the impact of noisy or irrelevant nodes.
- **High Performance**: Demonstrates superior accuracy, precision, recall, and F1-score compared to existing approaches.

## Dataset

Experiments are conducted on the **Binkit dataset**, which includes large-scale binary functions compiled under multiple architectures and optimization settings. For evaluation, we focus on the **NORMAL subset** with x86 and ARM functions compiled at O0 optimization level.

## Results

Model-Name achieves significant improvements in cross-architecture detection tasks:

- Better balance of **accuracy, precision, recall, and F1-score** compared to baseline models.
- Effective detection on both **large-scale** and **normal-scale** binary functions.
- Demonstrated robustness through ablation studies, showing the advantage of combining semantic and structural features.