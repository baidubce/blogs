Qianfan-OCR 的 Layout-as-Thought 功能详解

> **摘要**：本文面向希望深入了解或使用 Qianfan-OCR 模型的开发者和研究者。Qianfan-OCR 是百度千帆团队推出的 4B 参数端到端文档智能模型，在 OmniDocBench v1.5 上以 93.12 的总分排名所有端到端模型第一，在 OlmOCR Bench 上达到 79.8。其核心创新 **Layout-as-Thought（布局即思考）** 让模型在生成最终输出前，先以版面分析作为"思考"过程，从而在保持端到端架构简洁性的同时，恢复了传统 Pipeline 系统的版面分析能力。本文将介绍 Layout-as-Thought 的核心概念、API 使用方法、训练数据方案，以及在实际文档解析和视觉问答任务上的效果展示与指标分析。
---

## 0. 背景：文档解析方法概述
文档解析（Document Parsing）旨在将文档图像转换为结构化的可编辑文本（如 Markdown、JSON 等）。当前主流的文档解析方法可以归纳为三种技术路线：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=4e65e0983c0f42b7a70979e6932c0af0&docGuid=CQBW8HoJWsu2Y6)

*图：(a) Pipeline/Two-Stage 方式将版面分析和内容识别拆分为多个独立阶段，存在误差传播和视觉上下文丢失的问题；(b) 端到端方式直接从图像生成结果，但缺少版面分析先验，面对复杂排版容易出现信息缺失或幻觉；(c) Layout-as-Thought 方式（Qianfan-OCR）在端到端模型中引入版面思考阶段，在文档解析前先进行版面分析，然后在回答阶段给出更准确的文档解析结果，恢复了版面分析能力的同时保持端到端架构的优势。*

### 0.1 Pipeline / Two-Stage 方式
**Pipeline 方式** 将文档解析拆分为多个独立阶段：首先进行**版面分析**（Layout Analysis），检测文档中的各个区域（文本块、表格、图片等）并确定阅读顺序；然后对每个区域进行**逐块识别**（文本 OCR、表格结构识别、公式识别等）；最后通过**规则化后处理**将各区域结果拼接合并为最终输出。

代表系统包括 PaddleOCR-VL、MinerU 2.5、MonkeyOCR 等。

**优势**：

* 拥有显式的版面分析输出（边界框、元素类型、阅读顺序），用户可以直接获取文档的结构化布局信息
* 各模块可以独立优化，模块化程度高

**不足**：

* **多阶段级联，误差逐级传播**：版面检测的错误会传递到后续识别阶段，最终影响输出质量
* **视觉上下文不可逆丢失**：逐块裁剪识别时，丢失了元素之间的空间关系、图表中的轴标注与数据点关联等全局视觉信息
* 部署复杂，需要协调多个异构组件

### 0.2 端到端方式（End-to-End）
**端到端方式** 使用统一的视觉语言模型（VLM），直接从文档图像生成最终的结构化输出，无需显式的版面分析阶段。

代表模型包括 Dotsocr、DeepSeek-OCR、olmOCR、PointsReader 等。

**优势**：

* 架构简洁统一，单个模型完成所有任务
* 保留完整的视觉上下文，模型可以感知全局空间关系
* 支持灵活的 prompt 驱动任务定义

**不足**：

* **缺少版面分析的先验知识**：模型在生成时没有显式的结构化布局信息指导，面对复杂排版（多栏布局、图表文公式混排、非标准阅读顺序）或者是版面标签多样时，容易出现**信息缺失**（遗漏某些区域）、阅读顺序错误（如第四节中文档解析例子）或**幻觉**（编造不存在的内容）
* 用户无法获取版面分析的中间结果（如各元素的定位信息和类型）

### 0.3 Layout-as-Thought 方式
为了弥合上述两种方法之间的鸿沟，Qianfan-OCR 提出了 **Layout-as-Thought（布局即思考）** 方法。其核心思想是：**在端到端模型中引入版面分析作为"思考"过程** —— 模型在生成最终输出之前，先在思考阶段进行结构化的版面分析（识别每个版面元素的位置、类型、内容摘要，并确定阅读顺序），然后在真正的回答阶段基于这些结构化先验信息来生成文档解析结果。

Qianfan-OCR 既支持端到端模式的文档解析（直接输出结果），也支持 Layout-as-Thought 模式的文档解析（先思考版面布局，再输出结果），用户可根据文档复杂度灵活选择。

这一方法兼具了 Pipeline 方式的版面分析能力和端到端方式的架构优势：

|特性|Pipeline 方式|端到端方式|Layout-as-Thought|
|-|-|-|-|
|架构|多阶段级联|单模型端到端|单模型端到端|
|版面分析|有（独立模块）|无|有（思考阶段）|
|视觉上下文|逐块裁剪后丢失|完整保留|完整保留|
|误差传播|级联累积|无|无|
|信息缺失/幻觉|较少|较多|较少|

---

## 1. 什么是 Layout-as-Thought
### 1.1 核心概念
**Layout-as-Thought（布局即思考）** 是 Qianfan-OCR 引入的一种创新机制，它将版面分析融入模型的"思考"过程。在传统端到端 OCR 模型中，模型直接从文档图像生成最终结果（如 Markdown），而 Layout-as-Thought 在生成最终输出之前，增加了一个可选的"思考阶段"：模型会先生成结构化的版面分析表示（包括边界框 bounding boxes、元素类型 element types、内容摘要 brief、阅读顺序 reading order），然后基于这些结构化先验知识来生成更准确的最终输出。

### 1.2 两大功能方向
Layout-as-Thought 支持两大文档智能任务：

**Layout-as-Thought for Doc Parsing（文档解析）**

* **输入**：文档页面图像
* **输出**：结构化 Markdown 文本
* **思考过程**：模型先识别页面上的每个版面元素（文本块、表格、图片、公式等），确定它们的位置、类型和阅读顺序，再按此顺序生成 Markdown
* **核心价值**：将文档图像准确转换为结构化文本，保留原始排版信息

**Layout-as-Thought for Doc VQA（文档视觉问答）**

* **输入**：文档页面图像 + 用户问题
* **输出**：精准答案
* **思考过程**：模型先通过版面分析定位与问题相关的文档区域（如特定表格、图表），再基于定位结果生成答案
* **核心价值**：通过空间定位提高复杂文档中信息提取的准确性

### 1.3 与非 Thinking 模式对比
|特性|Thinking 模式（Layout-as-Thought）|非 Thinking 模式|
|-|-|-|
|触发方式|设置 `enable_thinking: true`|不设置或设为 `false`|
|输出结构|`<think>...</think>` + 最终结果|直接输出最终结果|
|版面分析|有（bbox / label / brief）|无|
|适用场景|复杂排版、多栏布局、含图表公式的文档|简单纯文本页面|
|推理速度|较慢|较快|

---

## 2. 效果直观展示
在深入介绍技术细节之前，先通过两个真实案例，直观感受 Layout-as-Thought 带来的提升。

### 2.1 文档解析（Doc Parsing）示例
以下是一个英语教材页面的文档解析示例，展示 Thinking 模式与非 Thinking 模式的对比。

**输入**：一张包含标题、图片、列表、诗歌等多种元素的英语教材页面。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=dddabe8d4bde44e2a884609b71fdb3f2&docGuid=CQBW8HoJWsu2Y6)

**Thinking 模式版面分析输出**（部分）：

```
<box>[[<COORD_061>, <COORD_044>, <COORD_282>, <COORD_074>]]</box>
<label>paragraph_title</label>
<brief>标题为“Let's Spell”的章节标题</brief>
<box>[[<COORD_025>, <COORD_088>, <COORD_494>, <COORD_117>]]</box>
<label>paragraph_title</label>
<brief>标题为“A Listen, point and repeat”的章节标题</brief>
<box>[[<COORD_078>, <COORD_126>, <COORD_443>, <COORD_228>]]</box>
<label>image</label>
<brief>图片展示了两个场景：左侧是一个白色瓶子倾倒液体，液体呈黑色；右侧是一个蓝色瓶子倾倒液体，液体呈蓝色。两个瓶子下方均有灰色矩形区域，整体被红色虚线边框包围。</brief>
<box>[[<COORD_505>, <COORD_126>, <COORD_875>, <COORD_228>]]</box>
<label>image</label>
<brief>图片左侧是一只黄色的鸟在空中飞翔，下方有蓝色和灰色的矩形块；右侧是一个白色的碗，下方也有一个灰色的矩形块。整个画面被红色虚线边框包围。</brief>
<box>[[<COORD_078>, <COORD_248>, <COORD_443>, <COORD_351>]]</box>
<label>image</label>
<brief>图片分为上下两部分，上半部分显示一个教室场景，几名学生围坐在桌子旁，其中一名学生正在举手发言，背景中有一辆汽车；下半部分是一个男孩手持紫色物品（可能是手机或相机）的特写，他穿着绿色外套和黑色内搭，面带微笑。两个部分均被红色虚线边框包围，底部各有一个灰色矩形遮挡区域。</brief>
<box>[[<COORD_505>, <COORD_248>, <COORD_875>, <COORD_351>]]</box>
<label>image</label>
<brief>图片分为左右两部分，左侧是一双粉红色带有白色条纹的手套，下方有一个灰色矩形遮挡区域；右侧是一副蓝色镜框的眼镜，下方也有一个灰色矩形遮挡区域。整个画面被红色虚线边框包围。</brief>
<box>[[<COORD_078>, <COORD_368>, <COORD_443>, <COORD_471>]]</box>
<label>image</label>
<brief>图片左侧是一只展翅飞翔的黄色小鸟，背景为浅色天空；右侧是一个红色的矩形旗帜，下方有两个灰色矩形遮挡区域。整体被红色虚线边框包围。</brief>
<box>[[<COORD_505>, <COORD_368>, <COORD_875>, <COORD_471>]]</box>
...
```
模型共解析到 **23 个版面元素**，准确识别了页面中的段落标题（`paragraph_title`）、图片（`image`）、文本（`text`）等多种元素，并按正确的阅读顺序排列。每个图片元素都附带了详细的视觉描述（`<brief>`），有助于后续生成阶段理解图片内容。

**版面可视化**：将 Thinking 阶段输出的 bbox 和 label 绘制到原图上：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=cc40baf651b4463ea3b3e4a4051e4da8&docGuid=63VEytQ5cAiUIi)

**Thinking 模式最终 Markdown 输出**：

```md
## Let's Spell

## A Listen, point and repeat

![image](<box>[[<COORD_078>, <COORD_126>, <COORD_443>, <COORD_228>]]</box>)

![image](<box>[[<COORD_505>, <COORD_126>, <COORD_875>, <COORD_228>]]</box>)

![image](<box>[[<COORD_078>, <COORD_248>, <COORD_443>, <COORD_351>]]</box>)

![image](<box>[[<COORD_505>, <COORD_248>, <COORD_875>, <COORD_351>]]</box>)

![image](<box>[[<COORD_078>, <COORD_368>, <COORD_443>, <COORD_471>]]</box>)

![image](<box>[[<COORD_505>, <COORD_368>, <COORD_875>, <COORD_471>]]</box>)

## B Listen, colour and repeat

1. cl bl ock 2. bl gl ue

2. bl gl ue

3. cl gl ass 4. sl fl ow

4. sl fl ow

5. fl sl y 6. pl cl ay

6. pl cl ay

7. sl pl ant 8. gl sl ide

8. gl sl ide

## © Listen, write and say

Ms ___ is slim. She wears ___. Her gloves are ___.

## D Listen and repeat

My little blue plane Is flying in the sky. It flies very slowly, But it climbs up high.

![image](<box>[[<COORD_568>, <COORD_778>, <COORD_856>, <COORD_928>]]</box>)

8
```
**非 Thinking 模式 Markdown 输出**：

```md
## Let's Spell

## A Listen, point and repeat

![image](<box>[[<COORD_078>, <COORD_126>, <COORD_443>, <COORD_228>]]</box>)

![image](<box>[[<COORD_078>, <COORD_248>, <COORD_443>, <COORD_350>]]</box>)

![image](<box>[[<COORD_078>, <COORD_368>, <COORD_443>, <COORD_471>]]</box>)

![image](<box>[[<COORD_505>, <COORD_126>, <COORD_875>, <COORD_228>]]</box>)

![image](<box>[[<COORD_505>, <COORD_248>, <COORD_875>, <COORD_350>]]</box>)

![image](<box>[[<COORD_505>, <COORD_368>, <COORD_875>, <COORD_470>]]</box>)

## B Listen, colour and repeat

1. cl

2. bl

3. cl

4. sl

5. fl

6. pl

7. sl

8. gl

ock

ue

ass

ow

y

ay

ant

ide

## © Listen, write and say

Ms ___ is slim. She wears ___. Her gloves are ___.

## D Listen and repeat

My little blue plane

Is flying in the sky.

It flies very slowly,

But it climbs up high.

![image](<box>[[<COORD_568>, <COORD_778>, <COORD_856>, <COORD_929>]]</box>)

8
```
**两种模式的关键差异分析**：

|对比维度|Thinking 模式|非 Thinking 模式|
|-|-|-|
|**图片阅读顺序**|正确：按行交替排列（左1→右1→左2→右2→...），符合自然阅读顺序|错误：先列出所有左栏图片，再列出所有右栏图片（按列排列）|
|**列表格式**|正确：每行包含完整的选择题选项（如 `cl bl ock`）|错误：列表编号与内容断裂，选项被拆分到不同行|
|**诗歌格式**|合并为连续文本|保留了原始换行|

可以看到，Thinking 模式通过先进行版面分析（识别每个图片的位置并按阅读顺序排列），有效解决了多栏布局中的阅读顺序问题，而非 Thinking 模式由于缺少版面先验知识，容易按列而非按行排列多栏内容。

### 2.2 文档视觉问答（Doc VQA）示例
以下是对一个包含多个表格的学术论文页面进行视觉问答的示例。

**输入图像**：一个包含 4 个性能对比表格的论文页面。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=530a5bc4d26f48ed89b94cf0b8dbe725&docGuid=CQBW8HoJWsu2Y6)

**问题**：`图中4个表探究的内容分别是什么`

**Thinking 模式版面分析**：

```
<box>[[<COORD_061>, <COORD_053>, <COORD_933>, <COORD_220>]]</box>
<label>table</label>
<brief>展示多个模型在文本、公式、表格、阅读顺序等任务上的性能对比，
包含Pipeline Tools、Expert VLMs、General VLMs三类方法的多语言指标</brief>

<box>[[<COORD_070>, <COORD_271>, <COORD_925>, <COORD_434>]]</box>
<label>table</label>
<brief>展示不同模型在9种PDF页面类型（Book、Slides、Financial Report等）
上的端到端文本识别性能，使用编辑距离评估</brief>

<box>[[<COORD_038>, <COORD_472>, <COORD_465>, <COORD_609>]]</box>
<label>table</label>
<brief>展示不同模型在Fuzzy、Water、Color、None四种页面属性下的
端到端文本识别性能，使用编辑距离评估</brief>

<box>[[<COORD_038>, <COORD_716>, <COORD_465>, <COORD_857>]]</box>
<label>table</label>
<brief>展示不同模型在Single、Double、Three、Complex四种列布局类型下的
端到端阅读顺序评估结果，使用归一化编辑距离评估</brief>
```
可以看到，Thinking 模式精准定位了页面中的 **4 个表格区域**，并为每个表格生成了准确的内容摘要。模型利用这些版面分析结果，最终给出了比非 Thinking 模式更加详尽、结构化的回答。

**版面可视化**：Thinking 模式定位的 4 个表格区域：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=df5fc8d7a9a347b2bc2529ca6387c14d&docGuid=63VEytQ5cAiUIi)
**Thinking 模式答案**：

> 这4个表共同探究了文档理解模型在不同场景下的性能表现：
> 1. **第一张表**（顶部）：探究不同模型类型（Pipeline Tools、Expert VLMs、General VLMs）在多种文档元素（文本、公式、表格、阅读顺序）上的综合性能对比，评估指标包括准确率、召回率等。
> 2. **第二张表**（中部）：探究不同模型在9种PDF页面类型（Book、Slides、Financial Report、Textbook、Exam Paper等）上的端到端文本识别性能，使用编辑距离作为评估指标。
> 3. **第三张表**（中部偏下）：探究不同模型在4种页面属性（Fuzzy、Water、Color、None）下的端到端文本识别性能，评估不同干扰因素对模型的影响。
> 4. **第四张表**（底部）：探究不同模型在4种列布局类型（Single、Double、Three、Complex）下的端到端阅读顺序评估结果，使用归一化编辑距离作为评估指标。
> 
> 这四张表从模型类型、页面类型、页面属性和布局类型四个维度，全面评估了文档理解模型的鲁棒性和泛化能力。
**非 Thinking 模式答案**：

> 图中4个表探究的内容分别是:表2研究文档解析算法在OmniDocBench上的综合评估;表3研究端到端文本识别在OmniDocBench上的性能;表4研究端到端文本识别在OmniDocBench上不同页面属性下的评估;表5研究端到端阅读顺序评估在OmniDocBench上不同列布局类型下的评估。
可以看到，Thinking 模式通过先定位 4 个表格区域并理解每个表格的内容主题，最终给出了更加详尽、结构化的回答，包含了对每个表格内容的深入分析。

---

## 3. 如何在 Qianfan-OCR 中使用 Layout-as-Thought
### 3.1 适用场景
Layout-as-Thought 适用于以下场景：

* **复杂排版文档**：多栏布局、图文混排、非标准阅读顺序的文档（如试卷、技术报告、学术论文、报纸）
* **包含多种元素类型的文档**：同时包含文本、公式、表格、图片、图表等多种元素的页面
* **需要精确空间定位的问答**：需要从复杂文档中精准定位特定区域来回答问题的场景

对于简单纯文本页面、单一类型文档（如纯文本表单），建议使用非 Thinking 模式以获得更快的推理速度。

### 3.2 工作原理
Layout-as-Thought 的工作流程如下：

```
文档图像 + Prompt
        │
        ▼  （设置 enable_thinking: true）
        │  （chat_template.jinja 自动在 query 尾部添加 <think>）
        ▼
┌─────────────────────────┐
│   Thinking 阶段（版面分析）  │
│                         │
│  对每个版面元素生成：         │
│  <box>[[x1,y1,x2,y2]]</box>  │
│  <label>元素类型</label>    │
│  <brief>内容摘要</brief>    │
│  （按阅读顺序排列）          │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│     Generation 阶段      │
│                         │
│  基于版面分析结果，按正确的  │
│  阅读顺序和元素类型生成     │
│  最终的 Markdown / 答案   │
└─────────────────────────┘
```
**关键说明：** 开启 Layout-as-Thought 模式时，用户只需在 API 调用的参数中设置 `enable_thinking: true`，**无需手动在 query 末尾添加 **`<think>`** 标签**。这是因为 Qianfan-OCR 的 `chat_template.jinja` 模板中已经内置了相应的逻辑，当检测到 `enable_thinking: true` 参数时，会自动在 query 尾部追加 `<think>` 标签来触发思考模式。

### 3.3 输出格式详解
Thinking 阶段的输出被包裹在 `<think>...</think>` 标签中，其中包含按阅读顺序排列的版面元素列表。每个元素包含三个字段：

#### `<box>` 标签 — 边界框坐标
```
<box>[[<COORD_061>, <COORD_044>, <COORD_282>, <COORD_074>]]</box>
```
* 坐标格式为 `[x1, y1, x2, y2]`，分别表示左上角和右下角的坐标
* 坐标使用专用特殊 token `<COORD_0>` 到 `<COORD_999>` 表示，取值范围为 [0, 999]
* 这些坐标是**归一化坐标**，与图像实际分辨率无关
* 每个坐标仅使用单个 token 表示（如 `<COORD_779>` 是一个 token），相较于用多位数字（如 "779" 需要 3 个 token）表示，可将 Thinking 输出长度减少约 50%，这对于复杂文档（可能包含 60+ 个版面元素）的实际部署至关重要
* 这些特殊坐标 token 是在 Stage 3 训练阶段与版面分析数据一起引入的，模型通过持续预训练学习到了这些空间表示

**将坐标恢复到原图像素维度**的公式：

```python
# 假设图片尺寸为 img_w x img_h
x1_pixel = int(coord_x1 * img_w / 999)
y1_pixel = int(coord_y1 * img_h / 999)
x2_pixel = int(coord_x2 * img_w / 999)
y2_pixel = int(coord_y2 * img_h / 999)
```
例如：`<COORD_061>` 在宽度为 1000px 的图片上对应的像素位置为 `int(61 * 1000 / 999) ≈ 61px`。

#### `<label>` 标签 — 元素类型
模型支持 25 种细粒度版面类别，分为四大组：

|分组|标签|说明|
|-|-|-|
|**文本元素**（12种）|`text`|普通文本块|
||`vertical_text`|竖排文本|
||`paragraph_title`|段落标题|
||`doc_title`|文档主标题|
||`abstract`|摘要|
||`content`|正文内容|
||`reference`|参考文献标题|
||`reference_content`|参考文献具体内容|
||`number`|编号（如列表编号）|
||`aside_text`|侧边栏文本 / 边注|
||`header`|页眉文本|
||`footer`|页脚文本|
|**页眉页脚**（4种）|`header_image`|页眉图片|
||`footer_image`|页脚图片|
||`footnote`|脚注|
||`vision_footnote`|与视觉元素相关的脚注|
|**图表类**（6种）|`image`|图片|
||`chart`|图表（柱状图、折线图等）|
||`table`|表格|
||`figure_title`|图标题 / 图注|
||`seal`|印章|
||`algorithm`|算法块 / 伪代码|
|**公式类**（3种）|`display_formula`|行间公式|
||`inline_formula`|行内公式|
||`formula_number`|公式编号|

#### `<brief>` 标签 — 内容摘要
为每个版面元素提供简洁的文字描述，例如：

* 文本元素：`<brief>标题为"Let's Spell"的章节标题</brief>`
* 图片元素：`<brief>图片展示了两个场景：左侧是一个白色瓶子倾倒液体...右侧是一个蓝色瓶子倾倒液体...</brief>`
* 表格元素：`<brief>展示多个模型在文本、公式、表格等任务上的性能对比</brief>`

这些结构化先验信息指导后续的生成阶段：

* **元素类型感知生成**：根据元素类型选择合适的输出格式——公式用 `$...$` 或 `$$...$$` 包裹，表格转换为 HTML，图片使用 `![label](bbox)` 占位符
* **阅读顺序引导排列**：Thinking 阶段按自然阅读顺序（从上到下、从左到右、先左栏后右栏）枚举所有元素，确保最终输出按正确顺序排列

### 3.4 服务调用示例
Qianfan-OCR 通过百度智能云千帆平台提供服务，使用兼容 OpenAI 的接口调用。**开启 Layout-as-Thought 功能只需在请求参数中添加 **`enable_thinking: true`**，模型服务会自动处理 **`<think>`** 标签的注入，用户无需手动添加。**

#### 3.4.1 文档解析（Doc Parsing）
```python
import requests
import json
import base64

# ========== 配置参数 ==========
VLLM_URL = "<百度云 API 服务地址>"  # 替换为实际的服务地址
API_KEY = "<你的 API Key>"
MODEL_NAME = "<模型名称>"

# ========== 图片转 Base64 ==========
def image_to_base64(image_path: str) -> str:
    """将图片转换为 base64 编码"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ========== 定义文档解析 Prompt ==========
user_prompt = """
你是一个专门将 PDF 中提取的文档页面图像（单页或多页）转换为 Markdown 的 AI 助手。

你的任务是严格按照以下规则，将图像中所有可见内容准确转换为 Markdown。不得添加任何解释、评论或推断内容。

1. 页面：
- 输入可能包含一页或多页文档图像。
- 必须严格保持输入提供的页面顺序。
- 如果包含多页，请使用以下标记分隔页面：
  --- Page N ---
  （N 从 1 开始）
- 如果只有一页，请不要输出任何页面分隔符。

2. 文本识别：
- 准确转换所有可见文本内容。
- 不得猜测、推断、改写或纠正文本。
- 保留原始文档结构，包括但不限于：标题、段落、列表、图注、脚注等。
- 必须完整保留每一页中的页眉和页脚文本。

3. 阅读顺序：
- 按照自上而下、从左到右的顺序进行内容读取。
- 对于多栏排版，必须先完整读取左栏，再读取右栏。
- 不得为了语义或逻辑清晰度而调整内容顺序。

4. 数学公式：
- 将所有数学表达式转换为 LaTeX 格式。
- 行内公式必须使用 $...$。
- 行间（块级）公式必须使用：

  $$
  ...
  $$

- 必须严格保留原有符号、结构和排版。
- 不得编造、简化、规范化或纠正公式内容。

5. 表格：
- 所有表格必须转换为 HTML 格式。
- 使用 <table> 和 </table> 包裹整个表格。
- 保留原有的行列结构，包括合并单元格（rowspan、colspan）和空单元格。
- 不得重新组织或重新解释表格内容。

6. 图片：
- 不得描述图片内容。
- 必须使用以下格式保留所有图片元素：
  ![label](<box>[[x1, y1, x2, y2]]</box>)
- 允许的 label 仅包括：
  image, chart, header_image, footer_image, seal
- 不得引入新的 label。
- 不得删除、合并或重排图片元素。

7. 无法识别或缺失内容：
- 如果文本、符号或表格单元格无法识别，应保留其位置并将内容置空。
- 不得猜测或补全缺失内容。

8. 输出要求：
- 仅输出 Markdown 内容。
- 尽可能保留原始布局、间距和结构。
- 使用适当的换行清晰分隔不同元素。
- 不得包含任何解释性文字、元信息或注释。
""".strip()

# ========== 调用模型 ==========
def call_model(user_prompt: str, image_path: str, enable_thinking: bool = True) -> str:
    """调用模型进行文档解析

    Args:
        user_prompt: 文档解析的 prompt
        image_path: 文档图像路径
        enable_thinking: 是否开启 Layout-as-Thought 模式
            设为 True 时，服务端会自动在 query 尾部添加 <think> 触发思考模式，
            无需手动在 prompt 中添加 <think> 标签。
    """
    image_base64 = image_to_base64(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": user_prompt}
            ]
        }],
        "max_tokens": 16384,
        "temperature": 0.0,
        "top_p": 0.9,
        "enable_thinking": enable_thinking,  # 开启 Layout-as-Thought
        "skip_special_tokens": False,
        "max_patch_num": 24
    }

    resp = requests.post(
        VLLM_URL,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY},
        data=json.dumps(payload),
        timeout=120
    )
    return resp.json()["choices"][0]["message"]["content"]

# ========== 调用示例 ==========
image_path = "your_document_page.jpg"

# Thinking 模式（开启 Layout-as-Thought）
response_thinking = call_model(user_prompt, image_path, enable_thinking=True)

# 非 Thinking 模式
response_non_thinking = call_model(user_prompt, image_path, enable_thinking=False)
```
> **注意**：`enable_thinking: true` 参数会由服务端的 `chat_template.jinja` 模板自动在 query 尾部追加 `<think>` 标签来触发 Layout-as-Thought 模式。用户**不需要**也**不应该**在 prompt 文本中手动添加 `<think>` 标签。
**提取 Thinking 部分和最终 Markdown**：

```python
import re

def extract_thinking(response_text: str) -> str:
    """提取 Thinking 版面分析内容"""
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_markdown(response_text: str) -> str:
    """提取最终的 Markdown 内容"""
    if "</think>" in response_text:
        return response_text.split("</think>", 1)[1].strip()
    return response_text.strip()

# 提取版面分析结果
thinking_content = extract_thinking(response_thinking)
# 提取最终 Markdown
markdown_output = extract_markdown(response_thinking)

print("版面分析结果：")
print(thinking_content[:500])
print("\n最终 Markdown 输出：")
print(markdown_output)
```
#### 3.4.2 文档视觉问答（Doc VQA）
```python
def call_model_vqa(image_path: str, question: str, enable_thinking: bool = True) -> str:
    """调用模型进行文档视觉问答

    Args:
        image_path: 文档图像路径
        question: 用户问题
        enable_thinking: 是否开启 Layout-as-Thought 模式
            设为 True 时，服务端会自动在 query 尾部添加 <think>，无需手动添加。
    """
    image_base64 = image_to_base64(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": question}
            ]
        }],
        "max_tokens": 8192,
        "temperature": 0.0,
        "enable_thinking": enable_thinking,  # 开启 Layout-as-Thought
        "skip_special_tokens": False,
        "max_patch_num": 12
    }

    resp = requests.post(
        VLLM_URL,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + API_KEY},
        data=json.dumps(payload),
        timeout=120
    )
    return resp.json()["choices"][0]["message"]["content"]

# ========== 调用示例 ==========
image_path = "your_document_page.png"
question = "图中4个表探究的内容分别是什么"

# Thinking 模式（开启 Layout-as-Thought）
response = call_model_vqa(image_path, question, enable_thinking=True)

# 提取答案
answer = extract_markdown(response)  # 复用上面的 extract_markdown 函数
print(answer)
```
### 3.5 解析版面分析结果并可视化
以下代码可以从模型的 Thinking 输出中提取边界框信息，并绘制到原图上进行可视化：

```python
import re
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont

def parse_bbox_from_response(response_text: str) -> List[Dict]:
    """从 Thinking 内容中解析 bbox、label、brief 信息"""
    results = []
    coord_pattern = r'(?:<COORD_(\d+)>|(\d+))'

    for open_br, close_br in [(r'\[', r'\]'), (r'\[\[', r'\]'), (r'\[\[', r'\]\]')]:
        pattern = (
            r'<box>' + open_br + coord_pattern + r'[,\s]+' + coord_pattern + r'[,\s]+'
            + coord_pattern + r'[,\s]+' + coord_pattern + close_br + r'</box>'
            + r'\s*<label>(.*?)</label>\s*<brief>(.*?)</brief>'
        )
        for match in re.finditer(pattern, response_text, re.DOTALL):
            groups = match.groups()
            results.append({
                'bbox': [int(groups[0] or groups[1]), int(groups[2] or groups[3]),
                         int(groups[4] or groups[5]), int(groups[6] or groups[7])],
                'label': groups[8].strip(),
                'brief': groups[9].strip()
            })
        if results:
            break
    return results


def draw_bboxes_on_image(image_path: str, bbox_data: List[Dict]) -> Image.Image:
    """在图片上绘制边界框和标签"""
    img = Image.open(image_path)
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    COLOR_MAP = {
        'text': (255, 0, 0), 'vertical_text': (200, 0, 0),
        'paragraph_title': (255, 0, 255), 'doc_title': (0, 255, 255),
        'abstract': (0, 200, 200), 'content': (128, 128, 0),
        'reference': (160, 82, 45), 'reference_content': (139, 69, 19),
        'image': (255, 255, 0), 'chart': (0, 255, 0),
        'table': (0, 0, 255), 'display_formula': (0, 191, 255),
        'header': (64, 224, 208), 'footer': (100, 149, 237),
        'figure_title': (255, 128, 0), 'algorithm': (50, 205, 50),
        'aside_text': (255, 20, 147), 'vision_footnote': (148, 0, 211),
    }

    for idx, item in enumerate(bbox_data):
        # 坐标从 [0, 999] 恢复到原图像素维度
        x1 = int(item['bbox'][0] * img_w / 999)
        y1 = int(item['bbox'][1] * img_h / 999)
        x2 = int(item['bbox'][2] * img_w / 999)
        y2 = int(item['bbox'][3] * img_h / 999)
        color = COLOR_MAP.get(item['label'], (255, 165, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{idx+1}. {item['label']}"
        draw.text((x1, y1 - 15), label_text, fill=color, font=font)

    return img

# 使用示例
bbox_data = parse_bbox_from_response(response_thinking)
vis_img = draw_bboxes_on_image("your_document_page.jpg", bbox_data)
vis_img.save("visualization.jpg")
```
---

## 4. 数据方案
Layout-as-Thought 功能的训练数据构造是其成功的关键之一。以下介绍两种任务的数据构造方法。

### 4.1 Layout-as-Thought for Doc Parsing 数据
#### 数据构造流程
1. **版面检测与内容识别**：使用 **PaddleOCR-VL** 对文档页面进行版面检测和内容识别，获取每个文档块的边界框（bbox）和类别标签（label）。边界框坐标归一化到 [0, 999] 范围，以实现分辨率无关性。
2. **内容简要描述（brief）生成**：对于每一个检测到的文档块，使用 **Qwen3-VL-32B** 对其进行简要描述（brief），生成对该区域内容的概括性文字。这些简要描述将作为版面分析结果的一部分，帮助模型更好地理解每个区域的内容语义。
3. **Layout-as-Thought 数据格式化**：将上述信息组装成 `<think>` 标签内的结构化版面分析数据，包含 `<box>`、`<label>`、`<brief>` 三个字段，并按阅读顺序排列，作为训练数据中的中间推理过程。
4. **最终输出**：结合版面分析结果，生成对应的结构化 Markdown 作为最终输出标注。

#### 标签体系选择
* 采用 PaddleOCR-VL 的标签体系（而非 MinerU 2.5），因为其提供了 25 种更细粒度的类别标签
* PaddleOCR-VL 提供了如 `text`、`vertical_text`、`paragraph_title`、`doc_title`、`abstract`、`content`、`reference`、`reference_content`、`aside_text` 等细粒度标签，而 MinerU 2.5 仅使用 `text`、`title`、`list`、`aside_text` 等较粗粒度标签
* 细粒度标签直接有利于下游任务，例如区分 `abstract` 和 `content` 可以支持论文结构化提取，区分 `reference` 和 `reference_content` 可以实现参考文献的干净解析

#### 训练阶段
* **Stage 3（领域增强阶段，800B tokens）**：加入了**千万级别**的 Layout-as-Thought Doc Parsing 数据。此阶段维持 70% 领域特定数据 + 30% 通用数据的混合比例，在增强 OCR 特定能力的同时防止灾难性遗忘。同时在此阶段引入了坐标特殊 token（`<COORD_0>` 到 `<COORD_999>`），通过持续预训练让模型学习空间表示。
* **Stage 4（指令调优阶段）**：加入**质量更高的少量精选数据**，进一步优化模型在文档解析任务上的表现。此阶段通过公开数据收集、指令重写和逆向合成三种互补策略构建指令数据。

#### 踩坑经验
在 Layout-as-Thought for Doc Parsing 的落地过程中，我们经历了两个典型的阶段性问题，记录如下供参考。

**问题一：Think 模式指标长期低于非 Think 模式**

最初，我们只在 Stage 4（指令调优阶段）引入少量 Layout-as-Thought Doc Parsing 数据，且数据量少于非 think 模式的数据。结果是：在 OmniDocBench v1.5 上，think 模式的指标始终低于非 think 模式。

根本原因在于 think 数据量不足——Stage 4 本身数据量较小，少量的 think 格式数据不足以让模型充分学会在思考阶段进行有效版面推理。**解决方案**：在 Stage 3 和 Stage 4 两个阶段都引入大量 think 模式的 Doc Parsing 数据，将非 think 模式的数据全部合成为 think 模式的对应数据。调整后，think 模式与非 think 模式在 OmniDocBench v1.5 上的指标达到相当水平。

**问题二：引入坐标 token 后 Grounding 指标下降**

坐标特殊 token（`<COORD_0>` 到 `<COORD_999>`）最初仅在 Stage 4 引入，并对所有包含坐标的数据进行了格式修改。结果发现，模型在 Grounding 相关 benchmark 上明显低于未修改坐标 token 格式的基线版本。

原因同样是数据量的问题：新词表中加入 1000 个坐标 token，需要足够多的数据让模型学习这些 token 的语义，而 Stage 4 数据量偏少，难以完成充分训练。**解决方案**：将坐标 token 的引入提前到 Stage 3，同时在 Stage 3 和 Stage 4 两个阶段都对所有包含坐标的数据进行相应修改。调整后，模型在 Grounding 相关 benchmark 上与未修改版本持平。

**总结**：以上两个问题都揭示了同一个规律——**新格式或新词表的引入需要足够大的数据量支撑**，仅在指令调优阶段（Stage 4）做少量修改往往不够，需要将改动前移到持续预训练阶段（Stage 3），并保证足够的数据覆盖。

### 4.2 Layout-as-Thought for Doc VQA 数据
#### 数据构造流程
1. **文档块选取**：从文档的版面分析结果中，**随机选取一个或多个文档块**作为问答对的信息来源。
2. **问答对生成**：使用 **GPT-5** 针对选中文档块的内容设计问答对：

    * **文本区域**：直接基于文本内容生成问答对
    * **图表区域**：先对图表进行图片描述，再基于描述内容设计问答对

3. **数据格式化**：将版面分析结果（仅包含与问题相关的文档块）作为 `<think>` 内容，答案作为最终输出。

#### 训练阶段
* **Stage 4（指令调优阶段）**：加入**少量**高质量的 Doc VQA 数据。由于 Doc VQA 任务更注重理解和推理能力，在指令调优阶段引入效果最佳。

---

## 5. 指标分析
根据论文中的实验结果，Layout-as-Thought 在以下场景中指标表现更优：

#### OmniDocBench v1.5 上的 Thinking 模式分析
|指标|非 Thinking 模式|Thinking 模式|差异|
|-|-|-|-|
|Overall|**93.12**|92.64|-0.48|
|Text^Edit|**0.041**|0.052|+0.011|
|Formula^CDM|**92.43**|91.92|-0.51|
|Table^TEDs|91.02|**91.21**|**+0.19**|
|Table^TEDss|93.85|**94.03**|**+0.18**|
|R-order^Edit|**0.049**|0.051|+0.002|

虽然整体分数上 Thinking 模式略低（92.64 vs. 93.12），但**在表格相关指标上有明显提升**（Table^TEDs +0.19，Table^TEDss +0.18）。更重要的是，逐样本分析揭示了关键规律：

#### 按版面复杂度分析
论文中将 OmniDocBench v1.5 的样本按版面标签熵（layout label entropy）从高到低排列，绘制了累积分数曲线（如下图所示）：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=15e4eb10993a4bd1930f91330a13715e&docGuid=CQBW8HoJWsu2Y6)

*图：OmniDocBench v1.5 累积分数曲线（样本按版面标签熵降序排列）。在高熵区域（左侧），开启 Thinking 模式（蓝色实线）始终优于不开启（红色虚线），提供稳定的得分优势；随着低熵样本逐渐加入，差距缩小并最终反转，不开启 Thinking 模式的总分更高。*

* **高熵区域（复杂布局）**：包含混合文本、公式、表格、图片等多种元素的页面，**Thinking 模式始终优于非 Thinking 模式**，提供稳定的得分优势
* **低熵区域（简单布局）**：纯文本、单一类型的页面，非 Thinking 模式效果更好，因为显式的版面推理在结构简单的文档上引入了不必要的开销，甚至可能干扰直接识别

**实践指导**：

* **适合开启 Thinking 模式的文档**：试卷、技术报告、学术论文、报纸等含混合元素类型的复杂页面
* **适合关闭 Thinking 模式的文档**：纯文本页面、简单表单等单一类型文档

#### Qianfan-OCR 整体 Benchmark 成绩
|Benchmark|Qianfan-OCR 成绩|亮点|
|-|-|-|
|OmniDocBench v1.5|**93.12**|端到端模型第一，超越 DeepSeek-OCR-v2 (91.09)、Gemini-3 Pro (90.33)|
|OlmOCR Bench|**79.8**|端到端模型第一|
|OCRBench|**880**|超越 Qwen3-VL-4B (873)，所有模型第一|
|OCRBenchv2 (zh)|**60.77**|中文识别最佳，超越所有专用 OCR 模型|
|CCOCR-multilan|**76.7**|超越 Qwen3-VL-4B (74.2)，多语言 OCR 领先|
|CCOCR-overall|**79.3**|超越 Qwen3-VL-4B (76.5)|
|KIE 综合|**87.9**|超越所有商用模型（Gemini-3.1-Pro 79.2）和开源模型|
|DocVQA|92.8|接近 Qwen3-VL-4B (94.9)|
|CharXiv_DQ|**94.0**|图表理解能力突出|
|CharXiv_RQ|**85.2**|图表推理能力第一|
|ChartQA|**88.1**|图表问答第一|
|ChartBench|**85.9**|综合图表评估第一|

---

## 6. 总结
Layout-as-Thought 是 Qianfan-OCR 的核心创新之一，它通过将版面分析嵌入模型的"思考"过程，在保持端到端架构简洁性的同时，恢复了传统 Pipeline 系统的版面分析能力。

**核心优势**：

1. **功能恢复**：用户可以直接从端到端模型获取结构化的版面分析结果（元素定位、类型分类、空间接地），弥补了端到端 OCR 模型与 Pipeline 系统之间的功能鸿沟
2. **精度提升**：在复杂排版文档上，显式的结构化先验帮助解决版面歧义、多栏布局、非标准阅读顺序等挑战，尤其在表格相关指标上表现优异
3. **灵活可控**：通过简单的 `enable_thinking: true` 参数控制开关，用户可以根据文档复杂度灵活选择是否启用

**使用建议**：

* 对于复杂排版（多栏、多元素混合）的文档，建议开启 Thinking 模式以获取更准确的结果
* 对于简单纯文本页面，使用非 Thinking 模式即可，兼顾速度和准确性
* Thinking 模式的版面分析输出还可以作为下游系统的输入，实现更灵活的文档处理流水线

Qianfan-OCR 模型通过百度智能云千帆平台公开提供服务，更多使用示例和最佳实践请参考：[https://github.com/baidubce/qianfan-models-cookbook/tree/main/qianfan-ocr](https://github.com/baidubce/qianfan-models-cookbook/tree/main/qianfan-ocr)