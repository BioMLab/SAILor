# RPI通道集成指南

## 概述

本指南介绍了如何将GuidedRPI-Net的RPI（RNA-Protein Interaction）模态集成到LncAPNet的多通道融合架构中。

## 背景

### LncAPNet架构
LncAPNet是一个多通道融合模型，目前包含以下通道：
1. **LncMamba** - 基于Mamba的序列建模
2. **RNALoc-LM** - 基于RNA-FM的预训练模型
3. **CFPLncLoc** - 基于CGR特征的CNN通道
4. **iLoc-BERT** - 基于DNABERT的序列建模
5. **Intra-Graph Channel** - 基于GIN的图神经网络
6. **Inter-Graph Channel** - 基于GAT的图神经网络
7. **RPI Channel** - 基于RPI分数和RBP位置信息的引导式注意力模型（新增）

### GuidedRPI-Net
GuidedRPI-Net是一个基于RPI分数和RBP位置信息的引导式注意力模型，具有以下特点：
- 使用多头注意力机制处理RPI特征
- 位置编码增强序列信息
- 特征交互层增强特征间的关系
- 全局平均池化进行特征聚合

## 集成方案

### 1. RPI通道架构

新的RPI通道包含以下组件：

#### 核心组件
- **PositionalEncoding**: 为RPI特征添加位置信息
- **GuidedAttentionLayer**: 多头注意力机制，包含自注意力和前馈网络
- **FeatureInteractionLayer**: 特征交互层，增强特征间的关系
- **RPIChannel**: 主通道类，整合所有组件

#### 数据流程
1. 输入：基因ID列表
2. 特征提取：从RPI数据中获取对应的RPI分数
3. 特征嵌入：将RPI特征投影到高维空间
4. 序列化：将特征重塑为序列形式
5. 位置编码：添加位置信息
6. 注意力处理：通过多层引导式注意力
7. 特征交互：增强特征间的关系
8. 全局池化：聚合特征
9. 分类：生成最终预测

### 2. 数据对齐

为了将RPI数据与主数据集对齐，我们创建了数据预处理脚本：

```bash
python scripts/prepare_rpi_data.py
```

该脚本会：
- 加载主数据集和RPI数据
- 找到基因ID的交集
- 为缺失的基因ID创建零向量
- 保存对齐后的数据

### 3. 配置更新

在`configs/main_config.yaml`中添加了RPI通道配置：

```yaml
rpi_channel:
  enabled: true
  rpi_file: "data/processed/aligned_rpi_scores.csv"
  rbp_file: "data/rpi_data/rbp_locations.csv"
```

### 4. 模型集成

在`MetaArchitect`中集成了RPI通道：

- 初始化RPI通道
- 在前向传播中调用RPI通道
- 将RPI特征与其他通道特征融合

## 使用方法

### 1. 数据准备

首先运行数据预处理脚本：

```bash
python scripts/prepare_rpi_data.py
```

### 2. 配置设置

在`configs/main_config.yaml`中：

1. 启用RPI通道：
```yaml
rpi_channel:
  enabled: true
```

2. 将RPI通道添加到融合列表：
```yaml
active_fusion_channels: ["intra_graph_channel", "inter_graph_channel", "rpi_channel"]
```

### 3. 测试集成

运行测试脚本验证集成：

```bash
python scripts/test_rpi_integration.py
```

### 4. 训练模型

运行主训练脚本：

```bash
python main_run.py
```

## 技术细节

### 特征维度
- RPI特征维度：22（对应22个RBP）
- 嵌入维度：132（22 × 6）
- 序列长度：22
- 特征维度：6

### 注意力机制
- 多头注意力：2个头
- 注意力层数：3层
- Dropout率：0.1

### 融合维度
- 融合维度：768（与其他通道一致）
- 投影层：将6维特征投影到768维

## 优势

### 1. 多模态融合
RPI通道提供了RNA-蛋白质相互作用的信息，与序列和结构信息互补。

### 2. 引导式注意力
使用多头注意力机制处理RPI特征，能够捕获复杂的相互作用模式。

### 3. 位置感知
位置编码确保模型能够理解RPI特征的位置信息。

### 4. 特征交互
特征交互层增强了不同RBP特征之间的关系建模。

### 5. 灵活配置
可以通过配置文件轻松启用/禁用RPI通道，调整融合策略。

## 性能优化

### 1. 内存优化
- 使用批处理处理RPI特征
- 零向量处理缺失数据

### 2. 计算优化
- 并行处理多个基因ID
- 使用torch.no_grad()进行推理

### 3. 数据优化
- 预对齐RPI数据
- 缓存对齐结果

## 故障排除

### 常见问题

1. **RPI数据加载失败**
   - 检查文件路径是否正确
   - 确保RPI数据文件存在

2. **基因ID不匹配**
   - 运行数据预处理脚本
   - 检查基因ID格式

3. **内存不足**
   - 减少批处理大小
   - 使用CPU进行推理

4. **维度不匹配**
   - 检查配置文件中的维度设置
   - 确保融合维度一致

### 调试技巧

1. 使用测试脚本验证集成
2. 检查日志输出
3. 验证数据对齐结果
4. 监控内存使用情况

## 未来改进

### 1. 特征增强
- 添加更多RBP信息
- 集成蛋白质序列信息

### 2. 架构优化
- 尝试不同的注意力机制
- 优化特征交互层

### 3. 训练策略
- 使用预训练的RPI模型
- 实现渐进式训练

### 4. 评估方法
- 添加RPI特定的评估指标
- 进行消融实验

## 总结

RPI通道的成功集成为LncAPNet提供了RNA-蛋白质相互作用信息，增强了模型的多模态融合能力。通过引导式注意力机制和特征交互层，RPI通道能够有效捕获复杂的相互作用模式，为RNA亚细胞定位预测提供更丰富的信息。
