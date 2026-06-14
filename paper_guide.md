# 论文精读与中英对照翻译

## Can Graph Learning Improve Planning in LLM-based Agents?
## 图学习能否提升基于 LLM 的 Agent 的规划能力？

> 论文来源：NeurIPS 2024 · 作者：Wu et al. (微软亚洲研究院 + 香港中文大学 + 复旦大学)
> 开源代码：https://github.com/WxxShirley/GNN4TaskPlan

---

## 一、全文逻辑 · 一句话总结

> **LLM 在做任务规划时表现很差（准确率仅 14-25%），因为它天生不擅长处理"图结构"决策问题。
> 论文证明了这个问题的理论根源（注意力机制偏差 + 自回归损失缺陷），
> 然后提出在 LLM 旁边加一个 GNN（图神经网络）来做工具选择，
> GNN 天然适合处理图结构数据，不需要训练就已经能大幅提升效果（最高提升 65%）。**

---

## 二、论文结构总览

```
1. Abstract（摘要）→ 概述全文
2. Introduction（引言）→ 背景 + 问题 + 贡献
3. Preliminaries（预备知识）→ 任务规划的定义
4. Graph Formulation and Insights（图建模与分析）
   ├─ 3.1 图形式化定义
   ├─ 3.2 经验发现：LLM 规划失败的现象
   └─ 3.3 理论分析：为什么 LLM 会失败
5. Integrating GNNs and LLMs for Planning（方法）
   ├─ 4.1 动机
   ├─ 4.2 无需训练的 GNN 方法 (SGC)
   └─ 4.3 需要训练的 GNN 方法 (GraphSAGE等)
6. Experiments and Analysis（实验分析）
   ├─ 5.1 实验设置
   ├─ 5.2 无需训练方法的结果
   ├─ 5.3 需要训练方法的结果
   └─ 5.4 在大图上的扩展性
```

---

## 三、逐段落中英对照翻译

---

### Abstract / 摘要

**English:**
> Task planning in language agents is emerging as an important research topic alongside the development of large language models (LLMs). It aims to break down complex user requests in natural language into solvable sub-tasks, thereby fulfilling the original requests. In this context, the sub-tasks can be naturally viewed as a graph, where the nodes represent the sub-tasks, and the edges denote the dependencies among them. Consequently, task planning is a decision-making problem that involves selecting a connected path or subgraph within the corresponding graph and invoking it.

**中文翻译：**
> 语言 Agent 中的任务规划正在成为大语言模型（LLM）发展过程中的一个重要研究课题。它的目标是将用户用自然语言表达的复杂请求分解为可解决的子任务，从而完成原始请求。在这里，子任务可以自然地看作一个图（Graph），其中节点（Node）代表子任务，边（Edge）代表子任务之间的依赖关系。因此，任务规划本质上是一个决策问题——在图中选择一个连通的路径或子图，并执行它。

**English:**
> In this paper, we explore graph learning-based methods for task planning, a direction that is orthogonal to the prevalent focus on prompt design. Our interest in graph learning stems from a theoretical discovery: the biases of attention and auto-regressive loss impede LLMs' ability to effectively navigate decision-making on graphs, which is adeptly addressed by graph neural networks (GNNs). This theoretical insight led us to integrate GNNs with LLMs to enhance overall performance. Extensive experiments demonstrate that GNN-based methods surpass existing solutions even without training, and minimal training can further enhance their performance. The performance gain increases with a larger task graph size.

**中文翻译：**
> 本文探索了基于图学习（Graph Learning）的任务规划方法——这是一个与当前主流的提示设计（Prompt Design）正交的方向。我们对图学习的兴趣源于一个**理论发现**：注意力机制（Attention）的偏差和自回归损失（Auto-regressive Loss）阻碍了 LLM 在图上进行有效决策的能力，而这正是图神经网络（GNN）擅长的。这个理论洞见促使我们将 GNN 与 LLM 整合以提升整体性能。大量实验表明，基于 GNN 的方法即使**无需训练**也超越了现有方案，而少量训练可以进一步提升性能。而且**任务图越大，性能提升越多**。

---

### 1. Introduction / 引言

**English:**
> LLM-based agents have recently emerged as a rapidly growing field of research and are considered a significant step towards artificial general intelligence (AGI). These agents have achieved remarkable successes across a variety of domains, as evidenced by their ability to address complex AI challenges (e.g., HuggingGPT), excel in gaming environments (e.g., Voyager), and drive innovation in chemical research.

**中文翻译：**
> 基于 LLM 的 Agent 最近成为一个快速发展的研究领域，被认为是迈向通用人工智能（AGI）的重要一步。这些 Agent 在各个领域都取得了显著成功，例如解决复杂 AI 挑战（HuggingGPT）、在游戏环境中表现出色（Voyager）、推动化学研究创新等。

**English:**
> Within this burgeoning field, task planning in language agents emerges as a critical area of study. It involves LLMs autonomously interpreting user instructions, breaking user's instructions in natural language into concrete and solvable sub-tasks, and then fulfilling the user's request by executing each sub-task. For instance, in the case of HuggingGPT, task planning involves invoking expert AI models from the HuggingFace website to solve complex AI tasks beyond the capabilities of GPT alone.

**中文翻译：**
> 在这个蓬勃发展的领域中，语言 Agent 的任务规划成为一个关键研究领域。它涉及 LLM 自主解释用户指令，将用户的自然语言指令分解为具体可解决的子任务，然后通过执行每个子任务来满足用户请求。例如在 HuggingGPT 中，任务规划涉及从 HuggingFace 网站调用专家 AI 模型来解决 GPT 单独无法完成的复杂 AI 任务。

**English:**
> Given its practical significance, numerous algorithms have been proposed, with a major focus on prompt design. This paper proposes to explore an orthogonal direction, i.e., graph-learning-based approaches. In task planning, solvable sub-tasks can be naturally represented as a task graph, wherein each node corresponds to a distinct sub-task, and each edge signifies the dependencies between these sub-tasks. The crux of task planning, therefore, involves selecting a connected path or subgraph to satisfy the user's request, which is a decision-making problem on graphs.

**中文翻译：**
> 由于其实际重要性，已经有很多算法被提出，主要集中在提示设计（Prompt Design）上。本文提出探索一个正交的方向——基于图学习的方法。在任务规划中，可解决的子任务可以自然地表示为一个任务图（Task Graph），其中每个节点对应一个不同的子任务，每条边表示子任务之间的依赖关系。因此任务规划的关键就是选择一个连通的路径或子图来满足用户请求，这是一个图上的决策问题。

**English:**
> Adopting this framework, we analyze the task planning capabilities of LLMs, specifically within the context of HuggingGPT. Our empirical investigation uncovers that a considerable portion of planning failures can be ascribed to the LLMs' inefficacy in accurately discerning the structure of the task graph. This finding presents intriguing questions from both theoretical and empirical perspectives.

**中文翻译：**
> 采用这个框架，我们分析了 LLM 的任务规划能力，特别是在 HuggingGPT 的背景下。我们的实证调查发现，相当一部分规划失败可以归因于 LLM 无法准确理解任务图的结构。这一发现从理论和实证角度都提出了有趣的问题。

**English:**
> For the theoretical question, we first investigate the expressiveness of Transformer architectures when applied to graph tasks with sequential graph input, such as edge list representations, which is the graph input format for task planning. Our initial hypothesis is that the format of sequential graph input might not align with the inductive bias inherent to graph structures, potentially reducing expressiveness. Contrary to this hypothesis, it is proved that by taking edge lists as the input, a constant-width Transformer can solve graph decision-making problems by simulating dynamic programming algorithms on edge lists. Nevertheless, we find that LLMs' solutions lack invariance under graph isomorphism, an important property for graph decision-making problems. In addition, the expressiveness is weakened if the attention is sparse, which is typically observed in LLMs.

**中文翻译：**
> 对于理论问题，我们首先研究 Transformer 架构在处理顺序图输入（如边列表表示，即任务规划中图的输入格式）时的表达能力。我们最初的假设是：顺序图输入的格式可能与图结构的归纳偏差不一致，可能降低表达能力。与这个假设相反，我们证明：通过以边列表作为输入，一个恒定宽度的 Transformer 可以通过模拟边列表上的动态规划算法来解决图决策问题。然而，我们发现 LLM 的解决方案在图同构（Graph Isomorphism）下缺乏不变性，而这是图决策问题的一个重要属性。此外，如果注意力是稀疏的（这在 LLM 中很常见），表达能力会减弱。

> **💡 大白话解读：** 论文先假设"因为图是结构化的，而 Transformer 处理的是序列，所以 Transformer 不行"，结果发现理论上 Transformer 是足够的——给一个足够好的 Transformer，它能模拟动态规划算法解决图问题。但问题在于：真实的 LLM 有注意力稀疏性（不会关注图上所有节点），而且同样的图换个顺序输入 LLM，LLM 给出的答案可能不同（缺乏图同构不变性）。

**English:**
> Beyond expressiveness, we also examine the influence of auto-regressive loss, demonstrating that it introduces spurious correlations that can be harmful to graph decision-making tasks. These insights expose the inherent limitations of LLMs in task planning and, more broadly, in graph-related problems.

**中文翻译：**
> 除了表达能力，我们还考察了自回归损失（Auto-regressive Loss）的影响，证明它引入了虚假相关性（Spurious Correlations），这对图决策任务是有害的。这些发现揭示了 LLM 在任务规划中——更广泛地说，在与图相关的问题中——的固有限制。

> **💡 大白话解读：** 自回归损失（下一个词预测）让 LLM 去学"频率"而不是"逻辑"。比如路径 A→B→C→D 在训练数据中出现得多，模型就学这条路。但当需要 A→D 时，正确答案是 A→B→C→D（拼接两条路径），但自回归损失可能因为没见过而完全答不出来。

**English:**
> To tackle the limitations, we take the use of GNNs, which have been shown to adeptly handle graph decision-making problems, both in theory and in practice. Initially, we deploy LLMs to interpret an ambiguous user request, breaking it down into more detailed steps. Subsequently, we utilize a GNN to retrieve the relevant sub-tasks based on these detailed steps and the corresponding sub-task descriptions. Notably, this approach can be implemented without training if we adopt parameter-free GNN models such as SGC. In the case of training-based methods, we apply the Bayesian Personalized Ranking (BPR) loss to facilitate learning from the implicit sub-task rankings.

**中文翻译：**
> 为了解决这些限制，我们采用了 GNN —— 理论上和实践上都已被证明能熟练处理图决策问题。首先，我们用 LLM 来解释模糊的用户请求，将其分解为更详细的步骤。然后，我们利用 GNN 来根据这些详细步骤和对应的子任务描述来检索相关的子任务。值得注意的是，如果我们采用无参数 GNN 模型（如 SGC），这种方法可以**无需训练**直接实施。对于需要训练的方法，我们应用贝叶斯个性化排序（BPR）损失来从隐式子任务排序中学习。

**English:**
> Our main contributions are summarized as follows:
> 1. **Task Planning Formulation**: This study presents a formulation of task planning as a graph decision-making problem.
> 2. **Theoretical Insights**: We prove that Transformers have expressiveness to solve graph decision-making problems based on edge list input, but inductive biases of attention and the auto-regressive loss function may serve as obstacles to their full potential.
> 3. **Novel Algorithms**: Based on the theoretical analysis, we introduce an additional GNN for sub-task retrieval, available in both training-free and training-based variants.

**中文翻译：**
> 我们的主要贡献总结如下：
> 1. **任务规划形式化**：将任务规划定义为图决策问题
> 2. **理论洞见**：证明 Transformer 理论上可以解决图决策问题，但注意力的归纳偏差和自回归损失函数阻碍了其充分发挥潜力
> 3. **新算法**：基于理论分析，引入了一个额外的 GNN 进行子任务检索，有无需训练和需要训练两种版本

---

### 2. Preliminaries / 预备知识

**English:**
> In task planning, there is a pool of pre-defined tasks. Task planning inputs include this task pool and a user request. The user request is expressed in natural language, which is ambiguous and could encompass multiple complex tasks. The output is a sequence of tasks and the order of their invocation to address the user's request.

**中文翻译：**
> 在任务规划中，有一个预定义的任务池。任务规划的输入包括这个任务池和一个用户请求。用户请求以自然语言表达，可能含糊不清且包含多个复杂任务。输出是一系列任务及其调用顺序，用于满足用户请求。

> **💡 大白话举例：**
> - **任务池**：HuggingFace 上的各种 AI 模型，如图像分割、目标检测、翻译、语音合成等
> - **用户请求**："生成一张女孩在看书的图片，她的姿势和 example.jpg 中的男孩一样，然后用语音描述这张新图片。"
> - **输出**：姿态检测 → 姿态转图像 → 图像描述 → 语音合成（4个任务的调用序列）

**English:**
> The current solution of task planning is purely based on LLMs and involves two stages. The first stage involves request decomposition, where a user's ambiguous request is broken down into concrete steps via LLMs. The second stage is task retrieval. For each decomposed step, LLMs are employed to retrieve an appropriate task from the task pool and execute them in sequence.

**中文翻译：**
> 当前的任务规划方案完全基于 LLM，包含两个阶段。第一阶段是**请求分解（Request Decomposition）**，通过 LLM 将用户模糊的请求分解为具体的步骤。第二阶段是**任务检索（Task Retrieval）**，对于每个分解出的步骤，使用 LLM 从任务池中检索合适的任务并按顺序执行。

---

### 3. Graph Formulation and Insights / 图建模与分析

#### 3.1 图形式化定义

**English:**
> In this subsection, we formulate the task planning as a decision-making problem on the task graph. The task graph is a special kind of text-attributed graphs and we define it as G = (V, E, T). Each node v ∈ V represents a pre-defined task in the task pool, associated with a text t_v ∈ T describing its function. Each edge (u,v) ∈ E indicates a dependency between tasks (e.g., the output format of task u matches the input format of task v). Task planning is to select a path or connected sub-graph on the task graph.

**中文翻译：**
> 在本小节中，我们将任务规划形式化为任务图上的决策问题。任务图是一种特殊的文本属性图，定义为 G = (V, E, T)。每个节点 v ∈ V 代表任务池中的一个预定义任务，关联一个描述其功能的文本 t_v ∈ T。每条边 (u,v) ∈ E 表示任务之间的依赖关系（例如，任务 u 的输出格式匹配任务 v 的输入格式）。任务规划就是在任务图上选择一个路径或连通子图。

> **💡 大白话：** 任务图就像一张地铁图。每个站（节点）是一个 AI 工具，站之间的连线（边）表示可以从一个工具的输出接到另一个工具的输入。规划就是找到一条从起点到终点的正确路线。

#### 3.2 经验发现：LLM 规划失败的现象

**English:**
> With the task graph at hand, we diagnose LLMs in task planning. We adopt the experimental settings as outlined in the work of HuggingGPT. The evaluation metric calculates the F1 score to assess the accuracy of the tasks identified by LLMs against the ground-truth tasks. Additionally, we report two task-graph-related metrics: the node hallucination ratio and the edge hallucination ratio. These metrics measure the frequency of non-existent nodes (i.e., tasks) and edges (i.e., dependencies) outputted by LLMs, respectively, indicative of the models' misinterpretation of the graph input.

**中文翻译：**
> 有了任务图后，我们对 LLM 的任务规划能力进行诊断。我们采用 HuggingGPT 中的实验设置。评估指标计算 F1 分数来衡量 LLM 识别的任务与真实任务之间的准确度。此外，我们报告了两个与任务图相关的指标：**节点幻觉率（Node Hallucination Ratio）**和**边幻觉率（Edge Hallucination Ratio）**。这些指标分别衡量 LLM 输出的不存在的节点（任务）和边（依赖关系）的频率，表明模型对图输入的误解程度。

**English:**
> Our empirical findings reveal that (1) LLMs exhibit a certain hallucination ratio, and (2) there is a strong correlation between the hallucination ratio and planning performance. This suggests that LLMs struggle to accurately interpret the task graph while the task graph is the key to the performance. We further explore whether the incidence of hallucinations correlates with the number of sub-tasks. Figure 2b illustrates that the hallucination ratio increases with a larger task graph size.

**中文翻译：**
> 我们的实证发现表明：(1) LLM 确实有一定比例的幻觉；(2) 幻觉率与规划性能之间存在强相关性。这表明 LLM 难以准确理解任务图，而任务图恰恰是性能的关键。我们进一步探索了幻觉的发生是否与子任务数量相关。图 2b 显示，**任务图越大，幻觉率越高**。

> **💡 大白话：** LLM 在规划时经常"编造"不存在的工具或连接。而且工具数量越多（图越大），它编造得就越离谱。

#### 3.3 理论分析：为什么 LLM 会失败

**English:**
> In contrast to previous graph learning approaches for graph decision-making problems, LLMs process the graph input by flattening it into a sequence and are trained using an auto-regressive loss. We will then examine the impact of these two factors.

**中文翻译：**
> 与以往图决策问题的图学习方法不同，LLM 通过将图输入展平（Flatten）为序列来处理，并使用自回归损失进行训练。我们将研究这两个因素的影响。

> **English (Theorem 1):**
> Theorem 1. (LLMs have enough expressiveness) Assume the input format is given in (1) and f, g, □ in DP update (2) satisfy the assumptions 1 and 2 in Appendix. There exists a log-precision constant-depth and constant-width Transformer that simulates one step of DP update in (2). As a consequence, there exists a log-precision O(k)-depth and constant-width Transformer that simulates k steps of DP update in (2).

> **中文翻译（定理 1）：**
> 定理 1：（LLM 有足够的表达能力）假设输入格式如公式(1)所示，且动态规划更新中的 f, g, □ 满足附录中的假设1和2。存在一个对数精度恒定深度和恒定宽度的 Transformer，可以模拟(2)中的一步动态规划更新。因此，存在一个对数精度 O(k) 深度恒定宽度 Transformer，可以模拟(2)中的 k 步动态规划更新。

> **💡 大白话定理1：** 理论上，Transformer 在数学上只要有足够多的层和宽度，是可以模拟动态规划算法来解决图问题的。所以从纯理论角度，LLM "有能力"做好规划。

**English:**
> However, certain aspects of the proof's constructions are challenging to be realized in Transformers that have been pretrained on natural language. First, the embedding process must be carefully filtered to ensure invariance under graph isomorphism. This invariance property does not align with the inductive biases inherent in natural language, making it difficult to achieve. Consequently, if an LLM can accurately produce the correct answer for a specific ordering of nodes, it might not maintain this accuracy after the nodes have been reordered. Second, each token needs to synchronize its hidden states with all other tokens sharing the same token ID, which is of order O(|V|). In practice, the attention trained from natural language is typically sparse, leading to intractability issues.

**中文翻译：**
> 然而，证明中的某些构建在预训练于自然语言的 Transformer 中很难实现。首先，嵌入过程必须仔细筛选以确保图同构下的不变性。这种不变性与自然语言的归纳偏差不一致，因此很难实现。结果就是：如果 LLM 能准确给出某个节点顺序下的正确答案，但节点重新排序后可能就没那么准确了。其次，每个 token 需要与共享相同 token ID 的所有其他 token 同步其隐藏状态，这是 O(|V|) 的量级。实际上，从自然语言中训练的注意力通常是稀疏的，导致这个问题难以处理。

> **💡 大白话：** 虽然理论上 Transformer 能解决图问题，但实际上有两个障碍：
> 1. **图同构不变性**：图换个节点顺序应该还是同一个图，但 LLM 看到不同的输入顺序会给出不同答案
> 2. **注意力稀疏性**：LLM 的注意力是稀疏的（不可能关注图上所有几百个节点），所以实际上算不了图上所有节点的关系

**English:**
> How does auto-regressive Loss impact the generalization? Our investigation next focuses on the auto-regressive loss. Our findings indicate that auto-regressive loss can lead to the emergence of a frequency-based spurious correlation.

**中文翻译：**
> 自回归损失如何影响泛化？我们的研究接下来关注自回归损失。我们的发现表明，自回归损失可能导致基于频率的虚假相关性的出现。

**English (Example 1):**
> Consider a training dataset consisting of a sufficient number of valid paths. Suppose the dataset contains two paths a b c and b c d and there are no other paths such that t = d and the current node v_i = a for all i. Then we have N ≡ 0 for all u and the logits for the next node can be arbitrary. This results in the model's inability to predict the next node of a when given a as the source node and d as the target node. To a human, finding a path from a to d simply involves concatenating the paths a b c and b c d. However, auto-regressive loss fails under such circumstances.

**中文翻译（例子 1）：**
> 考虑一个包含足够多有效路径的训练数据集。假设数据集中包含两条路径：a→b→c 和 b→c→d，但没有其他从 a 到 d 的路径。那么对于所有 u，N ≡ 0，下一个节点的 logits 可以是任意的。这就导致模型在给定源节点 a 和目标节点 d 时，无法预测 a 的下一个节点。对人类来说，找到从 a 到 d 的路径只需要拼接 a→b→c 和 b→c→d 即可。但自回归损失在这种情况下会失败。

> **💡 大白话：** 自回归损失学的是"下一个词是什么"——它看训练数据中 A 后面经常出现 B，就学 A→B。但它不会像人一样"理解"图的结构然后把两条路径拼起来。所以遇到 A→?→D 这种需要推理的情况，它就傻了。

---

### 4. Integrating GNNs and LLMs for Planning / 方法

#### 4.1 动机

**English:**
> In the last section, we find that a considerable portion of planning failures can be ascribed to the LLMs' inefficacy in accurately discerning the structure of the task graph, due to the hallucination, the inductive bias of the attention, and next-token prediction loss. In contrast to LLMs, GNNs can strictly operate on the task graph, thereby avoiding hallucinations. Additionally, they leverage the graph structure as input, rather than flattening the graph into a sequence, thus overcoming the theoretical limitations discussed previously. Furthermore, GNNs have demonstrated proficiency in handling graph decision-making problems, both theoretically and empirically. As a result, the simplest fix is to integrate GNNs into the task-planning algorithm.

**中文翻译：**
> 在上一节中，我们发现相当一部分规划失败可以归因于 LLM 无法准确理解任务图的结构——原因包括幻觉、注意力的归纳偏差和自回归损失。与 LLM 不同，GNN 可以直接在任务图上操作，从而避免幻觉。此外，GNN 将图结构作为输入，而不是将图展平成序列，从而克服了前面讨论的理论限制。因此，最简单的修复方法就是将 GNN 集成到任务规划算法中。

> **💡 大白话：** LLM 的问题在于它"看不懂"图这种结构。GNN 天生就是处理图的，所以最自然的修正是：让 LLM 做它擅长的（理解自然语言，分解请求），让 GNN 做它擅长的（在图里选路径）。

#### 4.2 无需训练的 GNN 方法 (Training-free SGC)

**English:**
> As we discussed, the current solution to task planning involves two stages. The first stage requires the ability to understand users' requests in natural language and break them down into concrete instructions, which is the unique ability of LLMs. The second stage is to select a path on the task graph, where each node corresponds to a decomposed step. Thus, we can integrate GNNs in this stage.

**中文翻译：**
> 正如我们讨论的，当前的任务规划方案包含两个阶段。第一阶段需要理解用户的自然语言请求并将其分解为具体指令，这是 LLM 的独特能力。第二阶段是在任务图上选择一条路径，其中每个节点对应一个分解后的步骤。因此，我们可以在这个阶段集成 GNN。

**English:**
> For each decomposed step outputted by the first stage, we use a GNN to select a corresponding node within the task graph. Suppose we are selecting the node for the i-th decomposed step. First, we utilize a small pre-trained language model, e5-335M, to embed the i-th decomposed step. The resulting embedding is denoted as x_step_i. Second, for the task graph, we first use the same e5-335M to convert each node's description into embeddings, denoted as the node feature h_v^0. Then we adopt a K-layer SGC to compute the final node embeddings, resulting in h_v = h_v^(K). Given a sequence of previously selected task nodes {v_1, ..., v_{i-1}}, the next node v_i is chosen according to: v_i = argmax ⟨h_v, x_step_i⟩, where h_v is the final node embedding, and v is chosen from the neighbors of v_{i-1} in the task graph. Particularly, v_1 can be selected from the whole graph.

**中文翻译：**
> 对于第一阶段输出的每个分解步骤，我们使用 GNN 在任务图中选择对应的节点。假设我们要为第 i 个分解步骤选择节点：首先，我们使用一个小型预训练语言模型 e5-335M 来嵌入第 i 个分解步骤，得到嵌入向量 x_step_i。其次，对于任务图，我们使用同样的 e5-335M 将每个节点的描述转换为嵌入，作为节点特征 h_v^0。然后我们采用 K 层 SGC 来计算最终节点嵌入，得到 h_v = h_v^(K)。给定先前已选择的任务节点序列 {v_1, ..., v_{i-1}}，下一个节点 v_i 按照以下方式选择：v_i = argmax ⟨h_v, x_step_i⟩，其中 h_v 是最终节点嵌入，v 从 v_{i-1} 在任务图中的邻居中选择。特别地，第一个节点 v_1 可以从整个图中选择。

> **💡 大白话 SGC 方法：**
> 1. LLM 把用户请求分解为步骤（如"第一步：检测姿态；第二步：生成图片..."）
> 2. 用 e5 模型把每个步骤转成向量
> 3. 用 SGC 在任务图上做"图传播"——把邻居工具的信息融合到当前工具
> 4. 选择跟步骤向量最匹配的工具（从上一个工具的邻居里选）
>
> SGC 的公式：**h' = α · h + (1−α) · Adj · h**（自己的信息 + 邻居的信息加权融合）
> 因为 e5 是预训练的，SGC 没有参数要学，所以整个过程**不需要任何训练**！

#### 4.3 需要训练的 GNN 方法 (Training-based)

**English:**
> The inference process in training-required methods mirrors that of the training-free approach, with the difference being the substitution of parameter-free GNNs with parametric counterparts, such as GAT or GraphSAGE. Here we specify the training process of GNNs.

**中文翻译：**
> 需要训练的 GNN 方法的推理过程与无需训练方法类似，区别在于将无参数 GNN 替换为有参数的 GNN（如 GAT、GraphSAGE）。下面说明 GNN 的训练过程。

**English:**
> Data Preparation: We assume that each entry in the task planning dataset comprises a user request, a sequence of decomposed steps, and the corresponding ground-truth tasks. There is a one-to-one correspondence between the steps and tasks in the dataset. Therefore, the training dataset can be represented as {(s_i, v_i)}_i^n, where s_i is a step described in natural language, and v_i is its corresponding invoked task.

**中文翻译：**
> 数据准备：假设任务规划数据集中的每条记录包含用户请求、分解步骤序列和对应的真实任务。数据中的步骤和任务是一一对应的。因此，训练数据集可以表示为 {(s_i, v_i)}_i^n，其中 s_i 是自然语言描述的步骤，v_i 是其对应的工具任务。

**English:**
> Training Loss: The problem in the dataset can be viewed as a binary ranking problem, where the labeled node is 1 and the other nodes are 0. Therefore, we adopt the Bayesian Personalized Ranking (BPR) loss designed for recommendation with binary rankings. We select negative tasks that are textually similar to the positive task, and for computational efficiency, we limit our selection to 2 negative tasks per positive task.

**中文翻译：**
> 训练损失：数据集中的问题可以看作一个二值排序问题，其中标记节点为 1，其他节点为 0。因此，我们采用为推荐系统设计的贝叶斯个性化排序（BPR）损失。我们选择与正样本文本相似的负样本任务，为了计算效率，每个正样本只选 2 个负样本。

---

### 5. Experiments and Analysis / 实验分析

#### 5.1 实验设置

**English:**
> **Datasets**: We utilize four datasets across two task planning benchmarks: HuggingFace tasks, Multimedia tasks, and DailyLife API tasks from TaskBench, as well as TMDB API tasks from RestBench.

**中文翻译：**
> **数据集**：我们使用了两个任务规划基准中的四个数据集：
> - **HuggingFace**：HuggingFace 上的 AI 模型（如图像分割、翻译等），23 个工具
> - **Multimedia**：多媒体相关的用户任务（文件下载、视频编辑等）
> - **DailyLife**：日常生活 API（网络搜索、购物等）
> - **TMDB**：电影相关的搜索和检索任务
>
> 另外还有一个大图数据集 **UltraTool**（260 个工具节点）。

**English:**
> **Evaluation**: We adopt the evaluation metric in TaskBench and HuggingGPT, i.e., Node F1-Score (n-F1) and Link F1-Score (l-F1), which measure the accuracy of invoked tasks and invoked dependencies, respectively. Besides, the Accuracy (Acc) can measure the success rate from task level. We also measure the token consumption (#tok) as the efficiency metric.

**中文翻译：**
> **评估指标**：
> - **Node-F1 (n-F1)**：预测的工具节点与真实节点的匹配度
> - **Link-F1 (l-F1)**：预测的工具间依赖关系与真实依赖的匹配度
> - **Accuracy (Acc)**：整个任务链完全正确的比例
> - **#Tok**：消耗的 token 数量（效率指标）

**English:**
> **Choices of LLMs**: We consider close-sourced LLMs, i.e., GPT-3.5-turbo and GPT-4-turbo, as well as open-sourced LLMs with different parameter scales, including CodeLlama-13B, Mistral-7B, Vicuna-13B, and Baichuan2-13B.

**中文翻译：**
> **LLM 选择**：闭源 LLM（GPT-3.5-turbo、GPT-4-turbo）和开源 LLM（CodeLlama-13B、Mistral-7B、Vicuna-13B、Baichuan2-13B）

**English:**
> **Choices of GNNs**: To comprehensively investigate the effectiveness of different graph learning methods for task planning, we consider a wide range of graph neural networks, including SGC, GCN, GAT, GraphSAGE, GIN, and Graph Transformers.

**中文翻译：**
> **GNN 选择**：SGC、GCN、GAT、GraphSAGE、GIN、Graph Transformer 等多种图神经网络

---

#### 5.2 无需训练方法的实验结果

**English:**
> Table 1 shows both the overall performance and token consumption costs. Compared with direct inference, integrating an SGC consistently improves performance, underscoring the effectiveness of the proposed method. GraphSearch-type methods rely on beam search to identify paths and employ LLMs for evaluation, where longer processing times generally lead to better outcomes. Notably, our proposed method achieves comparable or superior performance to Beam Search while requiring 5-10 times fewer tokens and inference time.

**中文翻译：**
> 表 1 显示了整体性能和 token 消耗。与直接推理相比，**集成 SGC 一致地提升了性能**，证明了所提出方法的有效性。GraphSearch 类方法依赖波束搜索（Beam Search）来识别路径并使用 LLM 进行评估，处理时间越长效果越好。值得注意的是，**我们提出的方法达到了与 Beam Search 相当或更好的性能，同时只需要 5-10 分之一的 token 和推理时间**。

> **💡 核心发现：** SGC 在所有 LLM 和所有数据集上都比 Direct 好。而且消耗的 token 几乎没有增加（SGC 只需要把步骤转为向量，不需要 LLM 反复调用）。

---

#### 5.3 需要训练方法的实验结果

**English:**
> From Table 2, we observe a significant improvement in performance when employing a training-based GraphSAGE approach over the training-free method. However, the co-training of GNNs with e5-335M does not yield a marked improvement, suggesting that message passing is the crucial element for enhancing performance.

**中文翻译：**
> 从表 2 可以看出，使用基于训练的 GraphSAGE 方法相比无训练方法有显著的性能提升。然而，GNN 与 e5-335M 的联合训练并没有带来明显的改善，这表明**消息传递（Message Passing）才是提升性能的关键因素**。

**English:**
> Further analysis across a broad spectrum of GNNs reveals that powerful GNNs, such as GINs, perform similarly to networks perceived as less complex, like GCNs, and even underperform compared to GraphSAGE. This pattern indicates that the task's challenge may not lie in the expressiveness of the models but rather in their ability to generalize.

**中文翻译：**
> 在多种 GNN 上的进一步分析表明，强大的 GNN（如 GIN）与被认为较简单的网络（如 GCN）表现相似，甚至不如 GraphSAGE。这表明**任务的挑战不在于模型的表达能力，而在于它们的泛化能力**。

---

#### 5.4 大规模任务图上的扩展性

**English:**
> To demonstrate the scalability of our method to larger task graphs, we conducted a supplementary experiment on the newly released planning benchmark UltraTool, which features a relatively large task graph with 260 nodes. We present a performance comparison of GNN models (training-free SGC and training-required GraphSAGE) against strong baselines like Beam Search in Table 3. Among the metrics, accuracy (Acc) is calculated based on whether the predicted tasks match the ground-truth tasks, measuring the success rate at each case level. In such conditions, integrating a GNN significantly enhances performance and mitigates planning failures, e.g., GPT-4-turbo undergoes a 9.05% accuracy improvement with the introduction of GraphSAGE.

**中文翻译：**
> 为了展示我们的方法在更大任务图上的可扩展性，我们在新发布的大型规划基准 UltraTool（260 个节点）上进行了补充实验。在这种条件下，集成 GNN 显著提升了性能并减少了规划失败。例如，GPT-4-turbo 引入 GraphSAGE 后准确率提升了 9.05%。

> **💡 关键发现：图越大，GNN 带来的提升越明显！** 对于只有 23 个工具的 HuggingFace，SGC 提升有限；对于 260 个工具的 UltraTool，提升巨大。这是因为图越大，LLM 越容易"迷路"，GNN 的结构化优势越明显。

---

## 四、通俗解释全文逻辑

### 📖 这篇论文到底在讲什么？

想象你是一个**调度员**，你的工作是把用户的指令拆解成步骤，然后从几百个工具中选出合适的工具并排序。

**以前的做法（Baseline）：**
直接用 LLM（GPT、CodeLlama 等）来做这件事。给 LLM 一段用户指令 + 工具列表，让它直接输出工具序列。

**结果发现：** LLM 做得不好（准确率只有 14-25%），而且经常：
- 选不存在的工具（幻觉）
- 工具间的连接不对
- 工具多了就完全懵了

### 🔬 论文的核心贡献

**第一：找到了 LLM 失败的原因（理论分析）**
1. LLM 处理图的方式是把图"拍扁"成一段文字，但图换个顺序输入，LLM 的输出就不一样了（缺乏图同构不变性）
2. LLM 的注意力是稀疏的，不能同时关注图上所有节点
3. LLM 的训练目标（预测下一个词）让它学的是"频率"而不是"逻辑关系"

**第二：提出了修复方案（GNN + LLM 混合框架）**
- **LLM 做它擅长的**：理解自然语言，把用户请求分解成步骤
- **GNN 做它擅长的**：在工具关系图上做选择，找出与每个步骤最匹配的工具
- **最简单的版本（SGC）甚至不需要训练**，拿过来直接用

**第三：实验证明有效**
- 在所有 LLM（GPT-4、CodeLlama、Mistral 等）和所有数据集上都有效
- Link-F1 最高提升 39%，Accuracy 最高提升 65%
- 消耗的 token 极少（几乎没有增加推理成本）
- 图越大，提升越明显

### 🎯 为什么要做这个（答辩要点）

```
问题 → LLM 在任务规划上表现差
原因 → LLM 天生不擅长处理图结构数据（理论证明）
方案 → 加入 GNN 做图检索，LLM 只做自然语言理解
效果 → 所有指标一致提升，无需训练，几乎零成本
意义 → 简单有效的混合架构，图越大效果越好
```

---

## 五、术语表

| 英文 | 中文 | 简单解释 |
|------|------|---------|
| LLM-based Agent | 基于 LLM 的智能体 | 用大模型驱动的 AI 助手，能调用工具完成任务 |
| Task Planning | 任务规划 | 把用户请求拆解成子任务并排序执行 |
| Task Graph | 任务图 | 工具之间的依赖关系图，节点=工具，边=依赖 |
| Node Hallucination | 节点幻觉 | LLM 选择了不存在的工具 |
| Edge Hallucination | 边幻觉 | LLM 创建了不存在的工具间连接 |
| GNN (Graph Neural Network) | 图神经网络 | 专门处理图结构数据的神经网络 |
| SGC (Simple Graph Convolution) | 简单图卷积 | 一种无参数的 GNN，通过邻居信息传播增强节点特征 |
| GraphSAGE | 图采样聚合网络 | 一种有参数的 GNN，通过采样邻居聚合信息 |
| Graph Isomorphism | 图同构 | 图的"形状"不变性——换个节点顺序还是同一个图 |
| Auto-regressive Loss | 自回归损失 | LLM 的训练目标：预测下一个词 |
| Spurious Correlation | 虚假相关性 | 模型学到的不是真正的规律，而是表面的频率统计 |
| BPR Loss | 贝叶斯个性化排序损失 | 一种排序损失函数，常用于推荐系统 |
| Message Passing | 消息传递 | GNN 的核心机制：节点之间互相传递信息 |
| Zero-shot | 零样本 | 不经过任何训练，直接应用于新任务 |
