# Tutorial Coverage Audit and Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the tutorial against the ML-DL source corpus, fill missing technical coverage into the nearest existing chapters, and repair structural or Mermaid issues uncovered during the audit.

**Architecture:** First build a reproducible audit artifact that maps source topics to tutorial chapters and flags high-confidence gaps. Then repair structural inconsistencies and Mermaid risks before expanding the smallest set of tutorial chapters needed to absorb the missing topics. Finish with a repository-wide consistency sweep so README, SUMMARY, chapter titles, counts, and new chapter links all agree.

**Tech Stack:** Markdown, ripgrep, Python 3, git-free local file edits, Mermaid source validation heuristics

---

## File Structure

- Create: `/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/source-topic-map.tsv` — filtered source material inventory with normalized topic tags
- Create: `/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/tutorial-topic-map.tsv` — tutorial chapter inventory with normalized topic tags
- Create: `/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/coverage-gaps.tsv` — source-to-tutorial coverage decisions and suspected gaps
- Create: `/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/mermaid-audit.txt` — chapter-level Mermaid findings and suspected parse risks
- Modify: `README.md` — chapter totals, navigation blurbs, and any new supplemental chapter links
- Modify: `SUMMARY.md` — chapter navigation and numbering consistency
- Modify: `chapters/PART2-机器学习基础/2-4-特征工程.md` — absorb source-side data preprocessing topics missing from the tutorial
- Modify: `chapters/PART10-训练优化与部署/10-1-学习率与Batch-Size.md` — absorb large-batch, scheduler, and training-curve guidance gaps
- Modify: `chapters/PART10-训练优化与部署/10-2-混合精度与分布式训练.md` — absorb mixed-precision caveats and distributed-training notes if missing
- Modify: `chapters/PART10-训练优化与部署/10-3-模型压缩与优化.md` — absorb deployment-side quantization distinctions if missing
- Modify: `chapters/PART11-计算机视觉进阶/11-1-目标检测进阶.md` — absorb dataset formats, annotation flow, Mosaic/occlusion tricks, and detector evolution notes
- Modify: `chapters/PART11-计算机视觉进阶/11-2-语义分割与实例分割.md` — absorb segmentation practice notes and metric details
- Modify: `chapters/PART11-计算机视觉进阶/11-3-图像生成与编辑.md` — absorb OCR / feature extraction / image editing task framing only if the audit shows they belong here
- Modify: `chapters/PART12-多模态与Agent/12-2-AI-Agent基础.md` — absorb “外脑” / memory-system framing if the source corpus contains concepts not already covered
- Modify: `chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md` — repair numbering/title inconsistency and align with source-side KG/QA notes
- Create if needed: `chapters/PART11-计算机视觉进阶/11-5-视觉应用补充阅读.md` — only if OCR / fine-grained recognition / motion detection cannot be cleanly merged into existing Part 11 chapters

### Task 1: Build the audit artifacts

**Files:**
- Create: `/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/source-topic-map.tsv`
- Create: `/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/tutorial-topic-map.tsv`
- Create: `/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/coverage-gaps.tsv`

- [ ] **Step 1: Write the audit extraction script**

```python
from pathlib import Path
import re

SOURCE_ROOT = Path("/home/david/文档/Obsidian/03-Resources/技术/ML-DL")
TUTORIAL_ROOT = Path("/home/david/AI-Tutorial")
SESSION_DIR = Path("/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files")

SOURCE_DIR_HINTS = ("原理性知识", "基础网络结构", "应用领域", "数据处理", "优化技巧", "Agent", "lmm")
SKIP_SEGMENTS = ("/案例/", "/assets/", "/插件/")
SKIP_PREFIXES = ("sr-",)

def read_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem

def normalize_topics(text: str) -> list[str]:
    text = re.sub(r"[（）()【】\\[\\]：:,_\\-]+", " ", text)
    terms = []
    for token in text.split():
        if len(token) >= 2:
            terms.append(token.lower())
    return sorted(set(terms))
```

- [ ] **Step 2: Run the script skeleton to make sure it executes**

Run:
```bash
python /tmp/tutorial_audit_extract.py
```

Expected: FAIL because the file does not exist yet.

- [ ] **Step 3: Save the full script and emit TSV artifacts**

```python
source_rows = []
for path in SOURCE_ROOT.rglob("*.md"):
    raw = str(path)
    if not any(hint in raw for hint in SOURCE_DIR_HINTS):
        continue
    if any(seg in raw for seg in SKIP_SEGMENTS):
        continue
    if path.name.startswith(SKIP_PREFIXES):
        continue
    title = read_title(path)
    topics = normalize_topics(path.stem + " " + title)
    source_rows.append((str(path.relative_to(SOURCE_ROOT)), title, "|".join(topics)))

tutorial_rows = []
for path in (TUTORIAL_ROOT / "chapters").rglob("*.md"):
    title = read_title(path)
    topics = normalize_topics(path.stem + " " + title)
    tutorial_rows.append((str(path.relative_to(TUTORIAL_ROOT)), title, "|".join(topics)))

(SESSION_DIR / "source-topic-map.tsv").write_text(
    "path\ttitle\ttopics\n" + "\n".join("\t".join(row) for row in source_rows),
    encoding="utf-8",
)
(SESSION_DIR / "tutorial-topic-map.tsv").write_text(
    "path\ttitle\ttopics\n" + "\n".join("\t".join(row) for row in tutorial_rows),
    encoding="utf-8",
)
```

- [ ] **Step 4: Run the extractor and confirm the artifacts exist**

Run:
```bash
python /tmp/tutorial_audit_extract.py && ls -1 /home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/{source-topic-map.tsv,tutorial-topic-map.tsv}
```

Expected:
```text
/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/source-topic-map.tsv
/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/tutorial-topic-map.tsv
```

- [ ] **Step 5: Generate the first-pass gap file**

```python
KEY_TARGETS = {
    "voc": "chapters/PART11-计算机视觉进阶/11-1-目标检测进阶.md",
    "coco": "chapters/PART11-计算机视觉进阶/11-1-目标检测进阶.md",
    "数据增强": "chapters/PART2-机器学习基础/2-4-特征工程.md",
    "mosaic": "chapters/PART11-计算机视觉进阶/11-1-目标检测进阶.md",
    "ocr": "chapters/PART11-计算机视觉进阶/11-5-视觉应用补充阅读.md",
    "细粒度识别": "chapters/PART11-计算机视觉进阶/11-5-视觉应用补充阅读.md",
    "运动检测": "chapters/PART11-计算机视觉进阶/11-5-视觉应用补充阅读.md",
    "白化": "chapters/PART2-机器学习基础/2-4-特征工程.md",
    "离散化": "chapters/PART2-机器学习基础/2-4-特征工程.md",
    "ema": "chapters/PART10-训练优化与部署/10-1-学习率与Batch-Size.md",
}
```

- [ ] **Step 6: Run the gap pass and inspect the likely-missing rows**

Run:
```bash
python /tmp/tutorial_audit_gaps.py && rg '待增强|缺失' /home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/coverage-gaps.tsv
```

Expected: one or more rows for OCR, VOC/COCO, data augmentation, whitening/discretization, or training-trick topics that are absent or shallow in the current tutorial.

### Task 2: Fix structural mismatches and capture Mermaid failures first

**Files:**
- Create: `/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/mermaid-audit.txt`
- Modify: `README.md`
- Modify: `SUMMARY.md`
- Modify: `chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md`

- [ ] **Step 1: Confirm the known title/count mismatches are present**

Run:
```bash
rg '^# ' README.md SUMMARY.md chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md
```

Expected:
```text
README.md:| 总章节数 | 60 |
SUMMARY.md:- **总章节数：** 61（含6章新增补充阅读）
chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md:# 9-5 知识图谱与智能问答系统
```

- [ ] **Step 2: Write the Mermaid audit helper**

```python
from pathlib import Path
import re

chapter_root = Path("/home/david/AI-Tutorial/chapters")
out_path = Path("/home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/mermaid-audit.txt")
issues = []

for path in chapter_root.rglob("*.md"):
    text = path.read_text(encoding="utf-8")
    if text.count("```mermaid") != text.count("```") // 2:
        issues.append(f"{path}: fenced block count needs review")
    if re.search(r"```mermaid\\n\\s*graph\\s+[A-Z]+\\n.*[\\[\\(].*[\"“”].*[\\]\\)]", text, re.S):
        issues.append(f"{path}: review node labels with quotes or mixed brackets")

out_path.write_text("\\n".join(issues) + "\\n", encoding="utf-8")
```

- [ ] **Step 3: Run the helper and capture the flagged files**

Run:
```bash
python /tmp/mermaid_audit.py && sed -n '1,40p' /home/david/.copilot/session-state/a1bb940e-49d5-43a7-86c8-31e9f95c65f0/files/mermaid-audit.txt
```

Expected: a short list of chapters requiring manual Mermaid inspection.

- [ ] **Step 4: Repair the known numbering and count issues**

```markdown
<!-- chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md -->
# 9-6 知识图谱与智能问答系统

<!-- README.md -->
| 总章节数 | 61 |

<!-- SUMMARY.md -->
- **总章节数：** 61（含 6 章新增补充阅读）
```

- [ ] **Step 5: Re-run the structural checks**

Run:
```bash
rg '^# 9-6 知识图谱与智能问答系统|总章节数' README.md SUMMARY.md chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md
```

Expected:
```text
README.md:| 总章节数 | 61 |
SUMMARY.md:- **总章节数：** 61（含 6 章新增补充阅读）
chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md:# 9-6 知识图谱与智能问答系统
```

### Task 3: Absorb missing data and training topics into Parts 2 and 10

**Files:**
- Modify: `chapters/PART2-机器学习基础/2-4-特征工程.md`
- Modify: `chapters/PART10-训练优化与部署/10-1-学习率与Batch-Size.md`
- Modify: `chapters/PART10-训练优化与部署/10-2-混合精度与分布式训练.md`
- Modify: `chapters/PART10-训练优化与部署/10-3-模型压缩与优化.md`

- [ ] **Step 1: Confirm the current gaps before editing**

Run:
```bash
rg 'VOC|COCO|数据增强|白化|离散化|EMA|温度系数|learning curve|GIoU|DIoU|CIoU' chapters/PART2-机器学习基础/2-4-特征工程.md chapters/PART10-训练优化与部署/10-1-学习率与Batch-Size.md chapters/PART10-训练优化与部署/10-2-混合精度与分布式训练.md chapters/PART10-训练优化与部署/10-3-模型压缩与优化.md
```

Expected: sparse or empty matches for several of the audited source topics.

- [ ] **Step 2: Add the missing preprocessing and annotation notes to Part 2**

```markdown
### 6. 数据预处理在视觉任务中的扩展

- **标准化 / 归一化 / 白化**：三者目标不同。标准化关注分布平移与缩放，归一化关注数值范围，白化进一步消除特征相关性。
- **离散化**：适用于树模型、分桶统计特征和混合特征工程场景，但要警惕信息损失。
- **数据增强**：在图像任务中常见的有翻转、裁剪、颜色扰动、MixUp、CutMix、Mosaic；增强策略要与任务目标保持一致。
- **标注与数据格式**：目标检测常见 VOC / COCO，两者在标注粒度、字段组织和工具链兼容性上存在差异。
```

- [ ] **Step 3: Add the missing training heuristics to Part 10**

```markdown
### 补充：学习率与 Batch Size 的经验法则

- 大 batch 常配合 **线性学习率缩放** 与 **warmup**
- 训练中应同时观察 **训练曲线** 与 **验证曲线**，只看单点精度容易误判
- **EMA（Exponential Moving Average）** 常用于稳定评估指标，尤其在检测与生成任务中
- softmax 的 **温度系数** 会影响分布平滑程度，常用于蒸馏、采样和对比学习
```

- [ ] **Step 4: Add the missing precision / quantization caveats**

```markdown
### 混合精度训练的常见坑

- FP16 更依赖 loss scaling；BF16 动态范围更大，但硬件支持要求更高
- 梯度同步、梯度累加和混合精度常常同时出现，需要一起看显存和吞吐

### 部署侧量化补充

- 区分 **PTQ（Post-Training Quantization）** 与 **QAT（Quantization-Aware Training）**
- 检测、分割、多模态模型的量化精度回退通常比纯分类模型更明显
```

- [ ] **Step 5: Re-run the coverage checks**

Run:
```bash
rg 'VOC|COCO|数据增强|白化|离散化|EMA|温度系数|PTQ|QAT' chapters/PART2-机器学习基础/2-4-特征工程.md chapters/PART10-训练优化与部署/10-1-学习率与Batch-Size.md chapters/PART10-训练优化与部署/10-2-混合精度与分布式训练.md chapters/PART10-训练优化与部署/10-3-模型压缩与优化.md
```

Expected: all of the added topics now match in the intended chapters.

### Task 4: Absorb computer-vision application gaps into Part 11

**Files:**
- Modify: `chapters/PART11-计算机视觉进阶/11-1-目标检测进阶.md`
- Modify: `chapters/PART11-计算机视觉进阶/11-2-语义分割与实例分割.md`
- Modify: `chapters/PART11-计算机视觉进阶/11-3-图像生成与编辑.md`
- Create if needed: `chapters/PART11-计算机视觉进阶/11-5-视觉应用补充阅读.md`
- Modify: `README.md`
- Modify: `SUMMARY.md`

- [ ] **Step 1: Confirm the current Part 11 coverage before editing**

Run:
```bash
rg 'OCR|细粒度|运动检测|Mosaic|VOC|COCO|遮挡|IoU|mAP|Dice|GraphSim' chapters/PART11-计算机视觉进阶
```

Expected: some topics present, but OCR / fine-grained recognition / motion detection or annotation-flow details remain absent or too shallow.

- [ ] **Step 2: Deepen the detector and segmentation chapters first**

```markdown
### 数据集与标注格式

- **VOC**：结构简单，适合入门和中小规模检测实验
- **COCO**：类别、实例和任务字段更完整，更适合现代检测 / 分割基准

### 训练技巧补充

- **Mosaic / MixUp** 能增强尺度和上下文鲁棒性
- 遮挡场景下要关注小目标、密集目标和正负样本分配策略

### 分割评估补充

- 除 mIoU 外，还应结合 Dice、Boundary F-score 等指标看边界质量
```

- [ ] **Step 3: If the gap file still shows unmatched CV applications, create a supplemental-reading chapter**

```markdown
# 11-5 视觉应用补充阅读

## 1. OCR
- OCR 通常分为文本检测、文本识别、版面分析三层
- 复杂场景 OCR 受透视变形、密集排版和低分辨率影响明显

## 2. 细粒度识别
- 细粒度识别关注同属不同类的细节差异
- 关键难点是局部判别区域建模与长尾样本分布

## 3. 运动检测
- 传统方法多依赖背景建模，深度学习方法更强调时空特征联合建模
```

- [ ] **Step 4: Link the new chapter only if it was actually created**

```markdown
### PART11 计算机视觉进阶
- [11-5 视觉应用补充阅读 ⭐补充阅读](../../../chapters/PART11-计算机视觉进阶/11-5-视觉应用补充阅读.md)
```

- [ ] **Step 5: Re-run the Part 11 gap check**

Run:
```bash
rg 'OCR|细粒度|运动检测|Mosaic|VOC|COCO|Dice' chapters/PART11-计算机视觉进阶 README.md SUMMARY.md
```

Expected: the previously missing CV application topics now appear in Part 11 and, if a new file was created, also in the navigation files.

### Task 5: Align knowledge-graph and agent framing with the source corpus, then finish the final sweep

**Files:**
- Modify: `chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md`
- Modify: `chapters/PART12-多模态与Agent/12-2-AI-Agent基础.md`
- Modify: `README.md`
- Modify: `SUMMARY.md`

- [ ] **Step 1: Confirm the current gaps in KG/QA and Agent memory coverage**

Run:
```bash
rg '问句预处理|语义匹配|外脑|长期记忆|工作记忆|episodic|semantic memory' chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md chapters/PART12-多模态与Agent/12-2-AI-Agent基础.md
```

Expected: little or no coverage for one or more of these concepts.

- [ ] **Step 2: Add the missing KG/QA pipeline notes**

```markdown
### 问答系统流水线补充

1. **问句预处理**：分词、实体识别、指代消解、问句分类
2. **候选检索**：基于实体链接、倒排索引或向量检索缩小范围
3. **语义匹配 / 推理**：用语义匹配模型或图谱推理完成答案筛选
```

- [ ] **Step 3: Add the missing Agent memory framing**

```markdown
### 记忆系统的工程拆分

- **工作记忆**：保存当前任务上下文和中间结果
- **长期记忆**：保存跨任务稳定知识、用户偏好和历史案例
- **外脑**：把记忆、检索、文档和工具执行统一成一个可调用的外部知识层
```

- [ ] **Step 4: Run the final repository consistency sweep**

Run:
```bash
rg '^# ' chapters/**/*.md | head -n 10 && rg '总章节数|11-5 视觉应用补充阅读|9-6 知识图谱与智能问答系统' README.md SUMMARY.md chapters/PART9-高级学习范式/9-6-知识图谱与智能问答系统.md
```

Expected: title, count, and navigation strings are consistent with the edited tutorial.

- [ ] **Step 5: Run the saved audit reports one more time**

Run:
```bash
python /tmp/tutorial_audit_extract.py && python /tmp/tutorial_audit_gaps.py && python /tmp/mermaid_audit.py
```

Expected: refreshed audit artifacts with fewer or no high-confidence gaps remaining for the targeted topics.

## Self-Review

- The plan covers the spec’s three obligations: coverage mapping, tutorial supplementation, and structural/Mermaid repair.
- No commit steps are included because the user explicitly asked the agent not to perform git operations.
- The plan names exact repository files and session artifacts so the executor does not need to rediscover scope before editing.
