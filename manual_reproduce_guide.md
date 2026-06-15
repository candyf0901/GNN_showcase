# 手动复现操作指南

> 本文档教你如何**亲手**在电脑上复现论文实验，不依赖任何 AI 辅助。

---

## 第一步：找到项目目录

打开文件管理器（Win+E），找到：

```
E:\工作\天目启航结题\fjw负责的论文\GNN4TaskPlan\
```

确认能看到这些内容：

```
GNN4TaskPlan/
├── .venv/            ← 虚拟环境（已配好）
├── data/             ← 数据集
├── trainfree/        ← SGC 代码
├── evaluate.py       ← 评估脚本
├── demo.bat          ← 现场演示脚本
├── hf_models/        ← e5 模型文件
└── showcase/         ← 展示网页
```

---

## 第二步：打开终端（cmd）

方法一：在文件夹地址栏输入 `cmd` 回车

> 打开 `GNN4TaskPlan` 文件夹 → 点击地址栏（显示路径的地方）→ 输入 `cmd` → 回车

方法二：Win+R 输入 cmd，然后 cd 到目录

```cmd
cd /d E:\工作\天目启航结题\fjw负责的论文\GNN4TaskPlan
```

---

## 第三步：跑实验一（评估 LLM 直接推理）

在 cmd 中输入以下命令，回车：

```cmd
.venv\Scripts\python evaluate.py --dataset=huggingface --llm=CodeLlama-13b --method=direct
```

**你会在屏幕上看到：**

```
prediction/huggingface/CodeLlama-13b/direct.json # Valid Predictions 497
+-------------+---------------+-------+--------+--------+--------+--------+...
| huggingface | CodeLlama-13b | chain | 0.5755 | 0.2888 | 0.1429 | ...
+-------------+---------------+-------+--------+--------+--------+--------+...
```

- 这个过程不到 1 秒
- 它是读取作者提供的预测结果跟真实答案做对比
- 结果中的 **NF=0.5755, LF=0.2888, Acc=0.1429** 就是论文中的 Direct 效果

**其他组合可以试试：**

```cmd
.venv\Scripts\python evaluate.py --dataset=multimedia --llm=CodeLlama-13b --method=direct
.venv\Scripts\python evaluate.py --dataset=huggingface --llm=Mistral-7b --method=direct
```

---

## 第四步：跑实验二（SGC 图增强）

首先进入 trainfree 目录：

```cmd
cd trainfree
```

然后运行 SGC：

```cmd
..\.venv\Scripts\python sgc.py --dataset=huggingface --llm_name=CodeLlama-13b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0
```

**你会在屏幕上看到（约 25 秒）：**

```
Namespace(dataset='huggingface', ... device='cuda:0', ...)

Loading weights: 100% |████████████████████| 391/391
[Model] Number of parameters: 335141888

+-------------+---------------+---------------+----------+---------+---------+----------+
| huggingface | CodeLlama-13b |   e5-large   |  Direct  |  0.5755 |  0.2888 |  0.1429  |
| huggingface | CodeLlama-13b |   e5-large   |   0.8    |  0.658  |  0.4021 |  0.2354  |
+-------------+---------------+---------------+----------+---------+---------+----------+
```

**关键看最后几行：** Direct 行和最佳 α 行。

如果报错 `CUDA out of memory`，可以加 `--device=cpu` 用 CPU 跑（但要慢很多）。

**其他组合：**

```cmd
..\.venv\Scripts\python sgc.py --dataset=multimedia --llm_name=CodeLlama-13b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0
..\.venv\Scripts\python sgc.py --dataset=huggingface --llm_name=Mistral-7b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0
..\.venv\Scripts\python sgc.py --dataset=multimedia --llm_name=Mistral-7b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0
```

---

## 第五步：双击 demo.bat（答辩演示）

回到 `GNN4TaskPlan` 目录，找到 `demo.bat`，**直接双击**。

它会自动依次执行：
1. 显示系统信息（GPU、Python 版本）
2. 跑一遍 evaluate.py
3. 跑一遍 sgc.py
4. 打印论文 vs 本地对比表
5. 在最后显示 `按任意键继续...` —— 窗口不会自动关闭，方便老师查看

---

## 第六步：查看展示网页

双击 `GNN4TaskPlan\showcase\index.html`，用浏览器打开。

网页共 8 节，全部滚完可以看到第 8 节「本地复现验证」，里面有终端截图风格的运行记录、对比表和时间线。

---

## 常见问题

| 问题 | 解决方法 |
|------|---------|
| `.venv\Scripts\python` 不是命令？ | 确保你在 `GNN4TaskPlan` 目录下，且该目录下有 `.venv` 文件夹 |
| GPU 显存不足？ | 加 `--device=cpu` 参数（慢一些但也能跑） |
| 中文显示乱码？ | cmd 里先输入 `chcp 65001` 回车 |
| 找不到文件？ | 当前工作目录不对。先 `cd /d E:\工作\天目启航结题\fjw负责的论文\GNN4TaskPlan` |
| sgc.py 报 import 错误？ | 确保从 `trainfree` 目录运行，且模型路径正确 |

---

## 操作速查卡

```
实验一（评估）：   .venv\Scripts\python evaluate.py --dataset=huggingface --llm=CodeLlama-13b --method=direct
实验二（SGC）：    cd trainfree && ..\.venv\Scripts\python sgc.py --dataset=huggingface --llm_name=CodeLlama-13b --lm_name=../hf_models/intfloat/e5-large --use_graph=1 --device=cuda:0
演示脚本：         双击 demo.bat
展示网页：         双击 showcase\index.html
```
