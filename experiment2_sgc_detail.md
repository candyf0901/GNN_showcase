# 实验二详解：SGC 图传播实验

> 对应代码：`GNN4TaskPlan/trainfree/sgc.py`
> 论文章节：4.2 A Training-free GNN-based Approach + 5.2 Performance of the Training-free Approach

---

## 一、这个实验是干什么的？

**一句话：** 在 LLM 预测完工具链之后，用 SGC（简单图卷积）在工具关系图上做一次"邻居信息传播"，然后重新选一遍工具，看看效果能不能变好。

### 论文中的定位

```
论文的三步推理：
  ① LLM 做任务规划效果差（实验一验证了） 
  ② 原因是 LLM 不擅长处理图结构（理论分析）
  ③ 那加一个擅长处理图的东西（GNN）来帮忙 → 这就是实验二
```

---

## 二、实验流程（在你的电脑上实际发生了什么）

### 整体流水线

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SGC 实验完整流程                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ① 加载 e5-large 模型（335M 参数）                                    │
│     ↓                                                               │
│  ② 读取工具描述 → 用 e5 转成向量（23个工具，每个1024维）                  │
│     ↓                                                               │
│  ③ 读取工具关系图（谁依赖谁）→ 构建邻接矩阵                              │
│     ↓                                                               │
│  ④ SGC 图传播：h' = α·h + (1-α)·A·h（对不同的 α 都做一次）            │
│     ↓                                                               │
│  ⑤ 读取 LLM 的预测结果（preduction/.../direct.json）                    │
│     ↓                                                               │
│  ⑥ 对每个测试样本的每个步骤：从邻居中贪心选择最匹配的工具                    │
│     ↓                                                               │
│  ⑦ 与真实答案对比 → 输出 Node-F1、Link-F1、Accuracy 表格               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、逐个步骤详解

### 步骤①：加载 e5 模型

**你的电脑上发生的真实事情：**

```python
# sgc.py 第 112-114 行
tokenizer = AutoTokenizer.from_pretrained(args.lm_name)    # 加载分词器
lm_model = AutoModel.from_pretrained(args.lm_name).to(device)  # 加载模型到 GPU
```

**实际操作：**

1. 读取你电脑上 `hf_models/intfloat/e5-large/` 目录里的 **1.3GB 模型文件**
2. 把模型加载到 **RTX 5060 显存**中
3. 打印：
   ```
   Loading weights: 100% |████████████████████| 391/391
   [Model] Number of parameters: 335141888
   ```
   - 391 个权重文件块全部加载完毕
   - 模型有 3.35 亿个参数

---

### 步骤②：工具描述 → 向量

**代码做的事情：**

```python
# data/huggingface/tool_desc.json 里的工具
工具列表 = [
  {"id": "Automatic Speech Recognition", "desc": "Transcribes audio into text"},
  {"id": "Audio-to-Audio", "desc": "Enhances audio quality"},
  {"id": "Question Answering", "desc": "Answers questions based on context"},
  ...
]

# 每个工具组装成一段完整的文字描述：
"{{tool_name}} # Description # {{tool_desc}}"
→ "Automatic Speech Recognition # Description # Transcribes audio into text"

# e5 模型把这行文字转成一个 1024 维的向量
→ [0.123, -0.456, 0.789, ..., 0.321]  # 1024 个浮点数
```

**通俗理解：** 给每个工具做一个"数字身份证"，意思相近的工具身份证也长得像。

**为什么在这里就要转向量？** 因为后面在 GPU 上算向量相似度比读文字快几万倍。

---

### 步骤③：读取工具关系图

**代码做的事情：**

```python
# data/huggingface/graph_desc.json 里的边
边列表 = [
  {"source": "Audio-to-Audio", "target": "Automatic Speech Recognition"},
  {"source": "Automatic Speech Recognition", "target": "Question Answering"},
  {"source": "Automatic Speech Recognition", "target": "Text-to-Speech"},
  ...
]
```

#### 什么是邻接矩阵？

把"谁连谁"转成一个 23×23 的矩阵：

```
               Audio-to-Audio  ASR  QA  TTS  ...
Audio-to-Audio       0         1    0   0   ...   ← Audio-to-Audio → ASR
ASR                  0         0    1   1   ...   ← ASR → QA, ASR → TTS
QA                   0         0    0   0   ...
...
```

**通俗理解：** 一张表格，行是"从哪出发"，列是"到哪去"，有连接就填 1，没有就填 0。

```python
# 然后对邻接矩阵做归一化：
norm_adj = D^(-0.5) × adj × D^(-0.5)

# 这步的目的是：不让连接多的工具（"社交达人"）的信息淹没连接少的工具
```

---

### 步骤④：SGC 图传播（核心！！）

#### 代码中对应的部分

```python
# sgc.py 第 125-130 行
tool_emb_neigh = torch.sparse.mm(link_g, tool_emb)  # A · h（获取邻居信息）

for alpha in alpha_list:  # 遍历不同的 α 值
    cur_tool_emb = alpha * tool_emb + (1 - alpha) * tool_emb_neigh
    # h' = α · h + (1-α) · A · h
```

#### 图示

```
传播前：
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Audio-to-    │────→│ Automatic Speech │────→│ Question     │
│ Audio        │     │ Recognition      │     │ Answering    │
│ [0.1, 0.2]   │     │ [0.4, 0.5]       │     │ [0.7, 0.8]   │
└──────────────┘     └──────────────────┘     └──────────────┘

SGC 传播（α=0.8）：
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Audio-to-    │     │ ASR              │     │ QA           │
│ Audio        │     │                  │     │              │
│ 新向量 =     │────→│ 新向量 =         │────→│ 新向量 =     │
│ 0.8×自己     │     │ 0.8×自己        │     │ 0.8×自己    │
│ +0.2×ASR    │     │ +0.2×(QA+TTS)   │     │ +0.2×...    │
│ = [0.16,...] │     │ = [0.44,...]     │     │ = [0.69,...] │
└──────────────┘     └──────────────────┘     └──────────────┘
```

#### 为什么 α 取不同值？

论文遍历了 11 个 α 值：0.5, 0.55, 0.6, 0.65, ..., 1.0

| α | 自己占比 | 邻居占比 | 含义 |
|---|---------|---------|------|
| 0.5 | 50% | 50% | 完全信任邻居 |
| 0.8 | 80% | 20% | 主要信自己，参考一下邻居 |
| 1.0 | 100% | 0% | 完全不信邻居（等于没做图传播） |

**最佳 α=0.80：** 加 20% 的邻居信息就够了，加多了反而引入噪声。

---

### 步骤⑤：读取 LLM 预测结果

```python
# sgc.py 第 135-136 行
alignment_ids = json.load(open(f"../data/{args.dataset}/split_ids.json"))["test_ids"]["chain"]
new_alignment_ids, pred_dict = load_test_data(args.dataset, args.llm_name, ...)
```

**读取的文件：**
- `data/huggingface/split_ids.json` → 哪些样本是测试集（497 个）
- `prediction/huggingface/CodeLlama-13b/direct.json` → LLM 的预测结果

**预测结果的内容（每条样本）：**
```json
{
  "id": "23046980",
  "user_request": "I have a noisy audio recording...",
  "task_steps": [
    "Step 1: Use Automatic Speech Recognition to transcribe...",
    "Step 2: Enhance the audio quality...",
    "Step 3: Answer questions about the content..."
  ],
  "task_nodes": [
    {"task": "Automatic Speech Recognition", "arguments": ["example.wav"]},
    {"task": "Text-to-Speech", "arguments": ["enhanced_audio.wav"]},
    {"task": "Question Answering", "arguments": ["transcript.txt"]}
  ],
  "task_links": [
    {"source": "Automatic Speech Recognition", "target": "Text-to-Speech"},
    {"source": "Text-to-Speech", "target": "Question Answering"}
  ]
}
```

---

### 步骤⑥：贪心工具选择（Greedy Selection）

这是 SGC 做"决策"的关键步骤。

```python
# 对每个测试样本的每个步骤：
for data_id in new_alignment_ids:
    steps = pred_dict[data_id]["steps"]
    
    steps_emb = text_lm_forward(steps, ...)  # 步骤文字 → 向量
    
    ans = sequence_greedy_tool_selection(
        steps_emb,        # 步骤向量
        tool_emb_list[i], # 传播后的工具向量
        index2tool,       # 索引→工具名称的映射
        adj_g,            # 工具关系图（邻居关系）
        measure="distance"
    )
```

#### 贪心选择算法（文字描述）

```
第1步（从全局选）：
  步骤1的向量 vs 所有23个工具 → 算相似度 → 最高分的当选

第2步（从邻居选）：
  查看第1步选的工具的邻居 → 步骤2的向量 vs 这些邻居 → 最高分当选

第3步（从邻居选）：
  查看第2步选的工具的邻居 → 步骤3的向量 vs 这些邻居 → 最高分当选
  ...
```

**为什么叫"贪心"？** 每一步都只选当前最好的，不考虑未来。简单但有效。

#### 对比：没有 SGC 时怎么选的

LLM 直接推理（Direct）是让 LLM **一次性生成**整个工具链，等价于让 LLM 同时预测所有工具和连接。

SGC 是**一步一步选**，并且限制了只能从上一个工具的邻居里选——这就天然保证了连接的正确性。

---

### 步骤⑦：输出对比结果

```python
# 调用 bi_evaluate()，计算 F1 分数
score_dict = bi_evaluate(new_alignment_ids, final_pred_dict)
```

计算方式和实验一的 evaluate.py 完全一样：
- **Node-F1**：预测的工具集合 vs 真实工具集合
- **Link-F1**：预测的连接集合 vs 真实连接集合
- **Accuracy**：整条链完全正确的比例

---

## 四、你电脑上的完整执行记录

### 你跑的命令

```bash
cd GNN4TaskPlan\trainfree
.venv\Scripts\python sgc.py --dataset=huggingface --llm_name=CodeLlama-13b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0
```

### 命令行参数说明

| 参数 | 值 | 含义 |
|------|-----|------|
| `--dataset` | `huggingface` | 使用 HuggingFace 数据集（23个工具） |
| `--llm_name` | `CodeLlama-13b` | 使用 CodeLlama-13B 的预计算结果 |
| `--lm_name` | `../hf_models/intfloat/e5-large` | 使用本地下载好的 e5 模型 |
| `--use_graph` | `1` | 开启图传播（=1 做 SGC，=0 不做） |
| `--device` | `cuda:0` | 使用第一块 GPU（你的 RTX 5060） |
| `--alpha_list` | `[0.5..1.0]` | 默认，遍历 11 个 α 值 |
| `--measure` | `distance` | 使用欧氏距离算相似度 |

### 在你的 GPU 上跑了多久

| 阶段 | 耗时 |
|------|------|
| 加载 e5 模型 | ~2 秒 |
| 工具描述转向量 | ~1 秒 |
| SGC 图传播 | ~0.1 秒 |
| 497 个测试样本的步骤编码 + 匹配 | ~20 秒 |
| **总计** | **~25 秒** |

---

## 五、结果解读

### 你看到的结果表格

```
+-------------+---------------+---------------+----------+---------+---------+----------+
|   Dataset   |      LLM      |      LM      |  Alpha   | Node-F1 | Link-F1 | Accuracy |
+-------------+---------------+---------------+----------+---------+---------+----------+
| huggingface | CodeLlama-13b |   e5-large   |  Direct  |  0.5755 |  0.2888 |  0.1429  |
| huggingface | CodeLlama-13b |   e5-large   |   0.5    |  0.6096 |  0.341  |  0.159   |
| huggingface | CodeLlama-13b |   e5-large   |   0.55   |  0.6244 |  0.3618 |  0.1771  |
| huggingface | CodeLlama-13b |   e5-large   |   0.6    |  0.6288 |  0.3666 |  0.1811  |
| huggingface | CodeLlama-13b |   e5-large   |   0.65   |  0.6483 |  0.3869 |  0.2133  |
| huggingface | CodeLlama-13b |   e5-large   |   0.7    |  0.6551 |  0.3944 |  0.2254  |
| huggingface | CodeLlama-13b |   e5-large   |   0.75   |  0.6591 |  0.3986 |  0.2334  |
| huggingface | CodeLlama-13b |   e5-large   |   0.8    |  0.658  |  0.4021 |  0.2354  | ← 最佳
| huggingface | CodeLlama-13b |   e5-large   |   0.85   |  0.6546 |  0.3988 |  0.2314  |
| huggingface | CodeLlama-13b |   e5-large   |   0.9    |  0.6435 |  0.3845 |  0.2173  |
| huggingface | CodeLlama-13b |   e5-large   |   0.95   |  0.6377 |  0.3802 |  0.2133  |
| huggingface | CodeLlama-13b |   e5-large   | No Graph |  0.6338 |  0.3744 |  0.2093  |
+-------------+---------------+---------------+----------+---------+---------+----------+
```

### 逐行解读

| 行 | 解读 |
|----|------|
| **Direct** | LLM 自己推理的结果。只看这行你会觉得"LLM 确实不行" |
| **α=0.5** | 邻居加太多了（50%），效果比 Direct 好但不如后面 |
| **α=0.65-0.85** | 最佳区间，Link-F1 都在 0.39 以上 |
| **α=0.80 🏆** | 综合最佳：Node-F1=0.658, Link-F1=0.402, Acc=0.235 |
| **α=1.0 (No Graph)** | 完全不用图，但用 e5 匹配。比 Direct 好（说明 e5 编码质量比 LLM 直接输出高），但不如 α=0.80（说明加图传播确实有帮助） |

### 关键结论

```
SGC 带来的提升（最佳 α vs Direct）：
  Node-F1:  0.5755 → 0.6580  (+14%)
  Link-F1:  0.2888 → 0.4021  (+39%)  ← 提升最大！
  Accuracy: 0.1429 → 0.2354  (+65%)

为什么 Link-F1 提升最大？
  因为 SGC 强制"只能从邻居里选下一个工具"
  这天然保证了连接的正确性
  而 LLM 直接输出时经常选"不存在连接的路径"
```

---

## 六、如果要自己手动复现完整的步骤

### 在你电脑上操作

```bash
# 1. 进入项目目录
cd E:\工作\天目启航结题\fjw负责的论文\GNN4TaskPlan

# 2. 激活虚拟环境（每次新开终端都要做）
.venv\Scripts\activate

# 3. 跑实验二（SGC）
cd trainfree
python sgc.py --dataset=huggingface --llm_name=CodeLlama-13b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0

# 4. 换数据集跑
python sgc.py --dataset=multimedia --llm_name=CodeLlama-13b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0

# 5. 换 LLM 跑
python sgc.py --dataset=huggingface --llm_name=Mistral-7b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0
```

### 查看结果

结果会直接打印在终端里，就是一个表格。

如果你想把结果保存到文件：

```bash
python sgc.py ... > result_sgc.txt 2>&1
```

然后打开 `result_sgc.txt` 就能看到完整的输出。

---

## 七、常见问题

### Q1：SGC 和普通的 GNN 有什么区别？

SGC 是 GNN 中最简单的一种。普通 GNN（如 GCN）每层都要学参数（权重矩阵），而 SGC 把参数去掉，只做一步：**邻居求和 → 取平均**。

```
普通 GCN：h' = ReLU(W · A · h)    （W 是要训练的参数矩阵）
SGC：      h' =           A · h    （没有参数，直接传播）
```

所以论文说 SGC 是"parameter-free"（无参数），不需要训练。

### Q2：为什么 No Graph 也比 Direct 好？

```
Direct：    LLM 直接输出工具名（容易出错）
No Graph：  e5 算步骤和工具的相似度 → 选最像的

e5 是专门做"文本匹配"的模型，比 LLM 的"阅读理解"更擅长这个任务。
所以哪怕是 No Graph（没加图传播），也比 LLM 直接推理好。
```

### Q3：α=0.80 是什么意思？

```
α 控制"听自己的"和"听邻居的"的比例：
  - α=0.80 → 80% 自己的信息 + 20% 邻居的平均信息
  - 这 20% 的邻居信息，让工具向量"融合"了周围工具的特征
  - 结果就是：经常一起出现的工具，向量变得更像
  - 选工具时就更容易选到"正确配对"的工具对
```

### Q4：如果图很大（260 个工具），效果会变好吗？

论文说会更好，因为：

```
图越小（23个工具）→ LLM 还能勉强记住 → SGC 提升有限（~14%）
图越大（260个工具）→ LLM 完全懵了 → SGC 提升明显（~30%+）
```

这也是直觉上合理的：**工具越多，越需要结构化的图信息来导航。**
