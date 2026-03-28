# Transformer残差连接在代码中的体现

## 残差连接的位置

### 1. 注意力模块之后的残差连接

在 `LlamaDecoderLayer.forward()` 方法中（1101-1117行）：

```python
# 第1101行：保存输入作为残差
residual = hidden_states

# 第1103行：Layer Normalization（Pre-Norm架构）
hidden_states = self.input_layernorm(hidden_states)

# 第1105-1116行：Self Attention计算
hidden_states, self_attn_weights, ... = self.self_attn(
    hidden_states=hidden_states,
    ...
)

# 第1117行：残差连接 - 将注意力输出与原始输入相加
hidden_states = residual + hidden_states
```

**关键点：**
- `residual` 保存了**注意力计算之前的原始输入**
- `self.self_attn()` 返回的 `hidden_states` 是**注意力模块的输出**
- `residual + hidden_states` 实现了残差连接，将原始输入和注意力输出相加

### 2. MLP模块之后的残差连接

在同一个方法中（1119-1123行）：

```python
# 第1120行：保存当前状态作为残差
residual = hidden_states

# 第1121行：Layer Normalization
hidden_states = self.post_attention_layernorm(hidden_states)

# 第1122行：MLP（全连接层）计算
hidden_states = self.mlp(hidden_states)

# 第1123行：残差连接 - 将MLP输出与输入相加
hidden_states = residual + hidden_states
```

## 注意力输出到残差连接的完整流程

### 步骤1：注意力计算产生输出

在 `LlamaAttention.forward()` 方法中（1036-1051行）：

```python
# 第1036行：注意力权重与Value相乘，得到注意力输出
attn_output = torch.matmul(attn_weights, value_states)

# 第1043-1044行：调整维度
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

# 第1051行：通过输出投影层
attn_output = self.o_proj(attn_output)

# 第1059行：返回注意力输出
return attn_output, attn_weights, ...
```

### 步骤2：残差连接接收注意力输出

在 `LlamaDecoderLayer.forward()` 方法中：

```python
# 注意力模块返回的输出被赋值给 hidden_states
hidden_states, ... = self.self_attn(...)  # 这里接收 attn_output

# 残差连接：原始输入 +  /= 注意力输出
hidden_states = residual + hidden_states
```

## 数据流图示

```
输入: hidden_states (第1101行)
    ↓
[保存为 residual] ← residual = hidden_states (第1101行)
    ↓
[Layer Norm] ← self.input_layernorm(hidden_states) (第1103行)
    ↓
[Self Attention计算]
    ├─ Q, K, V投影
    ├─ 注意力权重计算: attn_weights = softmax(QK^T / √d)
    ├─ 注意力输出计算: attn_output = attn_weights @ V (第1036行)
    └─ 输出投影: attn_output = o_proj(attn_output) (第1051行)
    ↓
[返回 attn_output 作为 hidden_states] (第1105行)
    ↓
[残差连接] ← hidden_states = residual + hidden_states (第1117行)
    │         │                    │
    │         │                    └─ 注意力输出 (attn_output)
    │         └─ 原始输入 (residual)
    │
    ↓
输出: hidden_states (传递给下一层或MLP)
```

## 为什么残差连接会导致system注意力恢复？

### 问题分析

在第14层调整了注意力权重，增强了对图像token的关注，减弱了对system token的关注。但是：

1. **注意力输出（attn_output）** 是基于调整后的注意力权重计算的
   - 理论上，这应该减少对system token的依赖

2. **残差连接保留了原始信息**
   - `residual` 保存的是调整**之前**的 `hidden_states`
   - 这个 `residual` 包含了原始的system token信息
   - `hidden_states = residual + attn_output` 会将原始信息加回来

3. **结果**
   - 即使注意力输出减少了system token的影响
   - 但残差连接把原始system token信息又加回来了
   - 这使得调整的效果被部分抵消

### 数学表达

假设：
- 原始输入（residual）：包含system token信息 `x_system`
- 调整后的注意力输出：`attn_output_adjusted`（减少了对system的关注）
- 最终输出：`residual + attn_output_adjusted`

即使 `attn_output_adjusted` 中system token信息减少了，但 `x_system` 通过 `residual` 被保留了下来。

## 代码位置总结

| 位置 | 行号 | 代码 | 说明 |
|------|------|------|------|
| **保存残差1** | 1101 | `residual = hidden_states` | 保存注意力计算前的输入 |
| **注意力计算** | 1105-1116 | `self.self_attn(...)` | 计算注意力，返回attn_output |
| **残差连接1** | 1117 | `hidden_states = residual + hidden_states` | 注意力输出 + 原始输入 |
| **保存残差2** | 1120 | `residual = hidden_states` | 保存MLP计算前的输入 |
| **残差连接2** | 1123 | `hidden_states = residual + hidden_states` | MLP输出 + 输入 |

## 与注意力调整的关系

在第14层进行注意力调整时（485-521行），调整发生在：

1. **注意力权重层面**：修改 `attn_weights`
2. **注意力输出层面**：`attn_output = attn_weights @ value_states` 受到影响
3. **但是残差连接层面**：`residual` 保存的是调整**之前**的输入，这部分不受影响

因此，残差连接起到了**信息保留**的作用，使得后续层仍然可以访问到原始的system token信息，这可能就是为什么system注意力会恢复的原因之一。

