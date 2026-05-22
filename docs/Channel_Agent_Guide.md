# 通道选择智能体使用指南

## 概述

通道选择智能体是一个基于PPO（Proximal Policy Optimization）的强化学习智能体，用于自动选择LncAPNet多通道融合架构中的最优通道组合。该智能体能够从6种可能的通道组合中选择性能最佳的配置，替代手工调优过程。

## 功能特点

- **智能通道选择**：从6种预定义的通道组合中选择最优配置
- **PPO强化学习**：使用先进的策略优化算法
- **一次性训练**：训练完成后固定使用，无需重复训练
- **易于集成**：与现有LncAPNet架构无缝集成

## 通道组合

智能体支持以下6种通道组合（rpi_channel为必选）：

1. `[rpi_channel, lncmamba]`
2. `[rpi_channel, cfploc]`
3. `[rpi_channel, rnaloclm]`
4. `[rpi_channel, lncmamba, cfploc]`
5. `[rpi_channel, lncmamba, rnaloclm]`
6. `[rpi_channel, cfploc, rnaloclm]`

## 使用方法

### 1. 配置智能体

在 `configs/main_config.yaml` 中配置智能体参数：

```yaml
channel_agent:
  enabled: true  # 启用智能体
  model_path: "pretrained/channel_agent.pth"  # 训练好的模型路径
  
  # PPO参数
  state_dim: 64
  hidden_dim: 128
  learning_rate: 3e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  
  # 训练参数
  max_episodes: 100
  episode_length: 10
  batch_size: 32
  update_epochs: 10
```

### 2. 训练智能体

```bash
# 训练智能体
python train_agent.py --config configs/main_config.yaml --output_dir outputs/agent_training

# 仅评估已训练的智能体
python train_agent.py --config configs/main_config.yaml --eval_only
```

### 3. 在主训练中使用智能体

启用智能体后，主训练脚本会自动：

1. 加载训练好的智能体模型
2. 基于数据集特征生成状态
3. 选择最优通道组合
4. 应用选择的配置到模型训练

```bash
# 使用智能体进行主训练
python main_run.py
```

### 4. 测试智能体功能

```bash
# 运行测试脚本
python test_agent.py
```

## 文件结构

```
src/agent/
├── __init__.py              # 模块初始化
├── channel_selector.py      # 通道选择器
├── channel_agent.py         # PPO智能体
└── agent_trainer.py         # 智能体训练器

train_agent.py               # 智能体训练脚本
test_agent.py                # 智能体测试脚本
```

## 技术细节

### 状态空间

智能体的状态包含：
- 序列长度（归一化）
- 标签分布（10个类别）
- 特征统计（53维）

### 动作空间

6个离散动作，对应6种通道组合。

### 奖励函数

基于验证集准确率的性能评估：
```
reward = validation_accuracy
```

### 网络架构

- **Actor网络**：3层全连接网络，输出动作概率
- **Critic网络**：3层全连接网络，估计状态价值

## 训练流程

1. **初始化**：创建智能体和通道选择器
2. **经验收集**：智能体选择动作并评估性能
3. **策略更新**：使用PPO算法更新网络参数
4. **模型保存**：保存性能最佳的智能体

## 集成说明

智能体集成到主训练流程中：

1. 在模型构建前检查智能体配置
2. 如果启用，加载训练好的智能体
3. 生成状态并选择最优通道组合
4. 应用选择到配置并继续训练

## 注意事项

1. **训练时间**：智能体训练可能需要较长时间，建议在GPU上运行
2. **模型路径**：确保训练好的智能体模型路径正确
3. **配置一致性**：智能体选择的通道必须在配置中定义
4. **性能评估**：当前使用简化的性能评估，实际应用中需要真实模型训练

## 故障排除

### 常见问题

1. **模型文件不存在**
   - 确保先运行智能体训练脚本
   - 检查模型文件路径是否正确

2. **配置错误**
   - 检查 `channel_agent.enabled` 设置
   - 确保所有通道配置正确

3. **导入错误**
   - 确保 `src/agent` 目录存在
   - 检查Python路径设置

### 调试建议

1. 运行测试脚本验证基本功能
2. 检查日志输出了解执行流程
3. 使用较小的训练参数进行快速测试

## 扩展功能

未来可以考虑的扩展：

1. **动态权重分配**：不仅选择通道，还分配权重
2. **样本级选择**：为不同样本选择不同通道组合
3. **在线学习**：在训练过程中持续更新智能体
4. **多目标优化**：同时优化准确率和计算效率

