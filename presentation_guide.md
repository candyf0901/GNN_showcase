# 答辩展示操作指南

## 一、静态展示（网页）

打开 `GNN4TaskPlan/showcase/index.html`，用浏览器打开即可。
网页包含 8 个章节，从论文概述到本地复现验证全覆盖。

## 二、现场演示（demo.bat）

### 事前准备

1. 确保电脑**接通电源**（GPU 跑实验耗电较大）
2. 确保之前安装的 `.venv` 和环境没有被删除
3. 提前双击一次 `demo.bat` 测试是否能正常运行

### 答辩时演示

**方式一：直接双击 `demo.bat`**

位于 `GNN4TaskPlan/demo.bat`，双击后会依次：
1. 显示系统信息（GPU、Python版本）→ 证明是在你电脑上跑的
2. 跑 evaluate.py（约 1 秒）→ 验证 LLM 直接推理的效果
3. 跑 sgc.py（约 25 秒，需要 GPU）→ 验证 SGC 图增强的效果
4. 打印论文 vs 本地对比表 → 证明结果一致

**方式二：如果时间紧，只跑 evaluate.py**

```bash
cd GNN4TaskPlan
.venv\Scripts\python evaluate.py --dataset=huggingface --llm=CodeLlama-13b --method=direct
```

只需 1 秒，展示 LLM baseline 的结果。

### 如果 sgc.py 跑得太慢

SGC 需要加载 e5-large 模型和 GPU 计算，约 25 秒。
如果答辩时间紧，可以提前跑好并展示保存的终端输出（在 `showcase/evidence/` 目录下），
或者只跑一个数据集（HuggingFace 就够了）。

## 三、录屏推荐

- **Windows 自带**：按 `Win + G` 打开 Xbox Game Bar 录屏
- **OBS Studio**：免费开源录屏软件，效果更好
- **截图**：按 `Win + Shift + S` 截取终端窗口

建议录制：打开 demo.bat → 等待全部跑完 → 展示对比结果，整个过程约 30 秒。

## 四、文件结构速查

```
GNN4TaskPlan/
├── demo.bat                    ← 双击即可重新运行全部实验
├── showcase/
│   ├── index.html              ← 展示网页（打开给老师看）
│   ├── paper_guide.md          ← 论文翻译精读
│   ├── experiment_guide.md     ← 实验总览
│   ├── experiment2_sgc_detail.md  ← SGC 实验详解
│   └── evidence/               ← 终端输出证据文件
├── evaluate.py                 ← 评估代码
├── trainfree/sgc.py            ← SGC 代码
├── hf_models/                  ← e5-large 模型文件
└── .venv/                      ← Python 虚拟环境
```
