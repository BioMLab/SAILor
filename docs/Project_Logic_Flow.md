# LncAPNet 项目逻辑与架构大纲

## 1. 项目概述 (Project Overview)
**LncAPNet** 是一个用于预测长链非编码 RNA (LncRNA) 亚细胞定位的深度学习模型。
它采用**多模态 (Multi-modal)** 策略，整合了序列、结构、相互作用等多种生物学特征，以提高预测的准确性和鲁棒性。

## 2. 核心架构 (Core Architecture)

### 2.1 元架构 (MetaArchitect)
- **文件**: `src/models/meta_architect.py`
- **角色**: 整个模型的 "指挥官"。
- **功能**:
    - 初始化并管理所有子通道 (Channels)。
    - 接收原始数据批次 (Batch)。
    - 将数据分发给各个启用的通道。
    - 收集各通道提取的特征 (Features) 和辅助预测结果 (Logits)。
    - 调用融合模块 (Fuser) 进行特征融合。
    - 生成最终的分类预测。

### 2.2 多模态通道 (Multi-modal Channels)
模型包含多个并行的通道，每个通道负责处理一种特定的模态数据：

#### A. 序列通道 (Sequence Channel) - LncMamba
- **文件**: `src/models/lncmamba.py`
- **输入**: LncRNA 核苷酸序列 (Token IDs)。
- **核心技术**: **Mamba (状态空间模型)** + CNN。
- **优势**: Mamba 擅长捕捉超长序列 (LncRNA通常很长) 的长程依赖关系，计算效率优于 Transformer。CNN 用于提取局部 Motif 特征。
- **创新点**: 结合了局部卷积和全局状态空间建模。

#### B. 相互作用通道 (Interaction Channel) - RPI-Net
- **文件**: `src/models/rpi_channel.py`
- **输入**: RNA-蛋白质相互作用 (RPI) 分数向量。
- **核心技术**: **引导式注意力机制 (Guided Attention)**。
- **优势**: 利用已知的 RBP (RNA结合蛋白) 结合信息作为先验知识。
- **逻辑**: 通过注意力机制，让模型自动学习哪些 RBP 对定位最关键。

#### C. 分子内结构通道 (Intra-Graph Channel)
- **文件**: `src/models/intra_graph_channel.py`
- **输入**: RNA 二级结构图 (节点=核苷酸, 边=化学键)。
- **核心技术**: **GIN (图同构网络)**。
- **优势**: 直接对 RNA 的折叠结构进行建模，捕捉结构基序 (如茎环)。

#### D. 分子间图通道 (Inter-Graph Channel)
- **文件**: `src/models/inter_graph_channel.py`
- **输入**: 全局样本相似性图 (节点=RNA样本, 边=相似度)。
- **核心技术**: **GAT (图注意力网络)**。
- **优势**: 利用全局信息，即使某个样本自身特征不明显，也能通过与其相似的样本推断其定位 (直推式学习)。

### 2.3 融合模块 (Fusion Module) - RobustMBT
- **文件**: `src/models/robust_mbt_fuser.py`
- **输入**: 各通道提取的特征序列。
- **核心技术**: **多模态瓶颈变换器 (MBT)** + **模态 Dropout (Modality Dropout)**。
- **逻辑**:
    - 使用一组可学习的 "瓶颈 Token" (Bottleneck Tokens) 强制不同模态之间进行信息交互。
    - **关键创新**: 引入 **Modality Dropout**，在训练时随机丢弃某些模态（如随机屏蔽序列特征），强制模型去学习和利用其他弱模态（如结构或图特征），从而解决 "模态坍塌" (Modality Collapse) 问题。

## 3. 数据流向 (Data Flow)

1.  **输入 (Input)**: 原始数据 (序列, 结构文件, RPI分数, 基因ID)。
2.  **预处理 (Preprocessing)**:
    - 序列 -> Token IDs.
    - 结构 -> 图数据对象 (PyG Data).
    - RPI -> 归一化分数向量.
3.  **特征提取 (Feature Extraction)**:
    - 各通道并行工作，独立提取特征。
    - 每个通道同时输出一个 "辅助 Logits" (Auxiliary Logits)，用于计算辅助损失，帮助通道独立收敛。
4.  **特征融合 (Fusion)**:
    - 所有通道的特征被投影到统一维度。
    - RobustMBT 融合器将它们整合为一个融合特征表示。
5.  **分类 (Classification)**:
    - 融合特征通过最终分类器 (MLP)。
    - 输出最终的亚细胞定位概率。

## 4. 训练策略 (Training Strategy)

- **联合损失函数 (Joint Loss)**:
    - `Total Loss = Loss_Final + α * Σ(Loss_Auxiliary)`
    - 既优化最终融合结果，也监督每个子通道的学习，防止某个通道 "偷懒"。
- **模态 Dropout**: 训练期间随机屏蔽模态，推理期间使用所有模态。

## 5. 总结 (Summary)
LncAPNet 是一个结构严谨、逻辑清晰的多模态深度学习系统。它不仅利用了先进的 Mamba 和 GNN 技术，还通过独特的融合策略解决了多模态学习中的常见痛点，是生物信息学与深度学习结合的典范。
