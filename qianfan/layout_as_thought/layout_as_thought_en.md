A Deep Dive into Qianfan-OCR's Layout-as-Thought

> **Abstract**: This article is intended for developers and researchers who want to gain an in-depth understanding of or use the Qianfan-OCR model. Qianfan-OCR is a 4B-parameter end-to-end document intelligence model developed by Baidu's Qianfan team. It ranks first among all end-to-end models on OmniDocBench v1.5 with an overall score of 93.12, and achieves 79.8 on OlmOCR Bench. Its core innovation, **Layout-as-Thought**, enables the model to perform layout analysis as a "thinking" process before generating the final output, thereby restoring the layout analysis capabilities of traditional pipeline systems while maintaining the simplicity of an end-to-end architecture. This article covers the core concepts of Layout-as-Thought, API usage, training data strategies, and showcases its performance on real-world document parsing and visual question answering tasks with detailed metric analysis.
---

## 0. Background: Overview of Document Parsing Methods
Document Parsing aims to convert document images into structured, editable text (such as Markdown, JSON, etc.). Current mainstream document parsing methods can be categorized into three technical approaches:

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=6891fefaee1d4a0fae7535506f197a22&docGuid=57lowAtoWuahje)
*Figure: (a) The Pipeline/Two-Stage approach splits layout analysis and content recognition into multiple independent stages, suffering from error propagation and loss of visual context; (b) The End-to-End approach directly generates results from images but lacks layout analysis priors, leading to information loss or hallucinations when facing complex layouts; (c) The Layout-as-Thought approach (Qianfan-OCR) introduces a layout thinking stage within the end-to-end model — performing layout analysis before document parsing, then producing more accurate results in the answer stage, restoring layout analysis capabilities while preserving the advantages of the end-to-end architecture.*

### 0.1 Pipeline / Two-Stage Approach
The **Pipeline approach** splits document parsing into multiple independent stages: first performing **Layout Analysis** to detect various regions in the document (text blocks, tables, images, etc.) and determine reading order; then performing **block-wise recognition** (text OCR, table structure recognition, formula recognition, etc.) for each region; and finally using **rule-based post-processing** to concatenate and merge results from all regions into the final output.

Representative systems include PaddleOCR-VL, MinerU 2.5, MonkeyOCR, etc.

**Advantages**:

* Provides explicit layout analysis output (bounding boxes, element types, reading order), allowing users to directly obtain structured layout information of the document
* Each module can be independently optimized, offering high modularity

**Disadvantages**:

* **Multi-stage cascading with error propagation**: Errors in layout detection propagate to subsequent recognition stages, ultimately affecting output quality
* **Irreversible loss of visual context**: Block-wise cropping and recognition loses spatial relationships between elements, axis-label-to-data-point associations in charts, and other global visual information
* Complex deployment requiring coordination of multiple heterogeneous components

### 0.2 End-to-End Approach
The **End-to-End approach** uses a unified Vision-Language Model (VLM) to directly generate structured output from document images, without requiring an explicit layout analysis stage.

Representative models include Dotsocr, DeepSeek-OCR, olmOCR, PointsReader, etc.

**Advantages**:

* Clean and unified architecture — a single model handles all tasks
* Preserves complete visual context, enabling the model to perceive global spatial relationships
* Supports flexible prompt-driven task definition

**Disadvantages**:

* **Lacks layout analysis priors**: The model has no explicit structured layout information to guide generation. When facing complex layouts (multi-column, mixed figures/tables/formulas, non-standard reading orders) or diverse layout labels, it is prone to **information loss** (missing certain regions), incorrect reading order (as shown in the document parsing example in Section 4), or **hallucinations** (fabricating non-existent content)
* Users cannot access intermediate layout analysis results (such as element positioning and type information)

### 0.3 Layout-as-Thought Approach
To bridge the gap between the above two methods, Qianfan-OCR proposes the **Layout-as-Thought** approach. The core idea is: **introducing layout analysis as a "thinking" process within the end-to-end model** — before generating the final output, the model first performs structured layout analysis in the thinking stage (identifying the position, type, content summary of each layout element, and determining reading order), then generates document parsing results based on these structured priors in the actual answer stage.

Qianfan-OCR supports both end-to-end document parsing (direct output) and Layout-as-Thought document parsing (think about layout first, then output), allowing users to choose flexibly based on document complexity.

This approach combines the layout analysis capabilities of the Pipeline approach with the architectural advantages of the End-to-End approach:

|Feature|Pipeline Approach|End-to-End Approach|Layout-as-Thought|
|-|-|-|-|
|Architecture|Multi-stage cascading|Single-model end-to-end|Single-model end-to-end|
|Layout Analysis|Yes (independent module)|No|Yes (thinking stage)|
|Visual Context|Lost after block-wise cropping|Fully preserved|Fully preserved|
|Error Propagation|Cascading accumulation|None|None|
|Information Loss / Hallucinations|Less|More|Less|

---

## 1. What is Layout-as-Thought
### 1.1 Core Concept
**Layout-as-Thought** is an innovative mechanism introduced by Qianfan-OCR that integrates layout analysis into the model's "thinking" process. In traditional end-to-end OCR models, the model directly generates the final result (e.g., Markdown) from document images. Layout-as-Thought adds an optional "thinking stage" before generating the final output: the model first generates a structured layout analysis representation (including bounding boxes, element types, content briefs, and reading order), then produces more accurate final output based on these structured priors.

### 1.2 Two Major Capability Directions
Layout-as-Thought supports two major document intelligence tasks:

**Layout-as-Thought for Doc Parsing**

* **Input**: Document page image
* **Output**: Structured Markdown text
* **Thinking Process**: The model first identifies each layout element on the page (text blocks, tables, images, formulas, etc.), determines their positions, types, and reading order, then generates Markdown following this order
* **Core Value**: Accurately converts document images into structured text while preserving original layout information

**Layout-as-Thought for Doc VQA (Document Visual Question Answering)**

* **Input**: Document page image + user question
* **Output**: Precise answer
* **Thinking Process**: The model first locates document regions relevant to the question through layout analysis (e.g., specific tables, charts), then generates answers based on the localization results
* **Core Value**: Improves accuracy of information extraction from complex documents through spatial localization

### 1.3 Comparison with Non-Thinking Mode
|Feature|Thinking Mode (Layout-as-Thought)|Non-Thinking Mode|
|-|-|-|
|Trigger|Set `enable_thinking: true`|Not set or set to `false`|
|Output Structure|`<think>...</think>` + final result|Direct final result|
|Layout Analysis|Yes (bbox / label / brief)|No|
|Applicable Scenarios|Complex layouts, multi-column, documents with charts and formulas|Simple plain-text pages|
|Inference Speed|Slower|Faster|

---

## 2. Visual Demonstration of Results
Before diving into technical details, let's intuitively experience the improvements brought by Layout-as-Thought through two real-world examples.

### 2.1 Document Parsing (Doc Parsing) Example
Below is a document parsing example of an English textbook page, comparing Thinking mode with non-Thinking mode.

**Input**: An English textbook page containing various elements including titles, images, lists, and poetry.

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=dddabe8d4bde44e2a884609b71fdb3f2&docGuid=CQBW8HoJWsu2Y6)
**Thinking Mode Layout Analysis Output** (partial):

```
<box>[[<COORD_061>, <COORD_044>, <COORD_282>, <COORD_074>]]</box>
<label>paragraph_title</label>
<brief>Section title "Let's Spell"</brief>
<box>[[<COORD_025>, <COORD_088>, <COORD_494>, <COORD_117>]]</box>
<label>paragraph_title</label>
<brief>Section title "A Listen, point and repeat"</brief>
<box>[[<COORD_078>, <COORD_126>, <COORD_443>, <COORD_228>]]</box>
<label>image</label>
<brief>The image shows two scenes: on the left, a white bottle pouring liquid (black in color); on the right, a blue bottle pouring liquid (blue in color). Both bottles have gray rectangular areas below them, and the entire scene is enclosed by a red dashed border.</brief>
<box>[[<COORD_505>, <COORD_126>, <COORD_875>, <COORD_228>]]</box>
<label>image</label>
<brief>On the left side of the image is a yellow bird flying in the air, with blue and gray rectangular blocks below; on the right side is a white bowl, also with a gray rectangular block below. The entire scene is enclosed by a red dashed border.</brief>
<box>[[<COORD_078>, <COORD_248>, <COORD_443>, <COORD_351>]]</box>
<label>image</label>
<brief>The image is divided into upper and lower parts. The upper part shows a classroom scene with several students sitting around a table, one student raising their hand to speak, with a car visible in the background; the lower part is a close-up of a boy holding a purple object (possibly a phone or camera), wearing a green jacket and black undershirt, smiling. Both parts are enclosed by red dashed borders, with a gray rectangular area at the bottom of each.</brief>
<box>[[<COORD_505>, <COORD_248>, <COORD_875>, <COORD_351>]]</box>
<label>image</label>
<brief>The image is divided into left and right parts. The left side shows a pair of pink gloves with white stripes, with a gray rectangular area below; the right side shows a pair of blue-framed glasses, also with a gray rectangular area below. The entire scene is enclosed by a red dashed border.</brief>
<box>[[<COORD_078>, <COORD_368>, <COORD_443>, <COORD_471>]]</box>
<label>image</label>
<brief>On the left side of the image is a yellow bird with spread wings flying against a light sky; on the right side is a red rectangular flag, with two gray rectangular areas below. The entire scene is enclosed by a red dashed border.</brief>
<box>[[<COORD_505>, <COORD_368>, <COORD_875>, <COORD_471>]]</box>
...
```
The model detected a total of **23 layout elements**, accurately identifying paragraph titles (`paragraph_title`), images (`image`), text (`text`), and other element types on the page, arranged in the correct reading order. Each image element includes a detailed visual description (`<brief>`) to help the subsequent generation stage understand the image content.

**Layout Visualization**: Drawing the bboxes and labels from the Thinking stage onto the original image:

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=cc40baf651b4463ea3b3e4a4051e4da8&docGuid=63VEytQ5cAiUIi)
**Thinking Mode Final Markdown Output**:

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
**Non-Thinking Mode Markdown Output**:

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
**Key Differences Between the Two Modes**:

|Comparison Dimension|Thinking Mode|Non-Thinking Mode|
|-|-|-|
|**Image Reading Order**|Correct: Alternating by row (left1→right1→left2→right2→...), following natural reading order|Incorrect: Lists all left-column images first, then all right-column images (column-wise arrangement)|
|**List Format**|Correct: Each line contains the complete multiple-choice options (e.g., `cl bl ock`)|Incorrect: List numbers and content are disconnected, options split across different lines|
|**Poetry Format**|Merged into continuous text|Preserved original line breaks|

As shown, Thinking mode effectively resolves reading order issues in multi-column layouts by first performing layout analysis (identifying each image's position and arranging them in reading order), while non-Thinking mode, lacking layout priors, tends to arrange multi-column content by column rather than by row.

### 2.2 Document Visual Question Answering (Doc VQA) Example
Below is a visual question answering example on an academic paper page containing multiple tables.

**Input Image**: A paper page containing 4 performance comparison tables.

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=530a5bc4d26f48ed89b94cf0b8dbe725&docGuid=CQBW8HoJWsu2Y6)
**Question**: `What are the four tables in the figure investigating respectively?`

**Thinking Mode Layout Analysis**:

```
<box>[[<COORD_061>, <COORD_053>, <COORD_933>, <COORD_220>]]</box>
<label>table</label>
<brief>Performance comparison of multiple models across tasks including text, formula,
table, and reading order, covering metrics for three categories: Pipeline Tools,
Expert VLMs, and General VLMs in multiple languages</brief>

<box>[[<COORD_070>, <COORD_271>, <COORD_925>, <COORD_434>]]</box>
<label>table</label>
<brief>End-to-end text recognition performance of different models across 9 PDF page
types (Book, Slides, Financial Report, etc.), evaluated using edit distance</brief>

<box>[[<COORD_038>, <COORD_472>, <COORD_465>, <COORD_609>]]</box>
<label>table</label>
<brief>End-to-end text recognition performance of different models under four page
attributes (Fuzzy, Water, Color, None), evaluated using edit distance</brief>

<box>[[<COORD_038>, <COORD_716>, <COORD_465>, <COORD_857>]]</box>
<label>table</label>
<brief>End-to-end reading order evaluation results of different models under four
column layout types (Single, Double, Three, Complex), evaluated using
normalized edit distance</brief>
```
As shown, Thinking mode precisely locates the **4 table regions** on the page and generates accurate content summaries for each table. The model leverages these layout analysis results to ultimately provide a more detailed and structured answer than non-Thinking mode.

**Layout Visualization**: The 4 table regions located by Thinking mode:

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=df5fc8d7a9a347b2bc2529ca6387c14d&docGuid=63VEytQ5cAiUIi)

**Thinking Mode Answer**:

> These four tables collectively investigate the performance of document understanding models across different scenarios:
> 1. **Table 1** (top): Investigates comprehensive performance comparisons across different model types (Pipeline Tools, Expert VLMs, General VLMs) on various document elements (text, formulas, tables, reading order), with evaluation metrics including accuracy and recall.
> 2. **Table 2** (middle): Investigates end-to-end text recognition performance of different models across 9 PDF page types (Book, Slides, Financial Report, Textbook, Exam Paper, etc.), using edit distance as the evaluation metric.
> 3. **Table 3** (middle-lower): Investigates end-to-end text recognition performance of different models under 4 page attributes (Fuzzy, Water, Color, None), evaluating the impact of different interference factors on models.
> 4. **Table 4** (bottom): Investigates end-to-end reading order evaluation results of different models under 4 column layout types (Single, Double, Three, Complex), using normalized edit distance as the evaluation metric.
> 
> These four tables comprehensively evaluate the robustness and generalization capabilities of document understanding models from four dimensions: model type, page type, page attributes, and layout type.
**Non-Thinking Mode Answer**:

> The four tables in the figure investigate the following: Table 2 studies the comprehensive evaluation of document parsing algorithms on OmniDocBench; Table 3 studies end-to-end text recognition performance on OmniDocBench; Table 4 studies end-to-end text recognition evaluation under different page attributes on OmniDocBench; Table 5 studies end-to-end reading order evaluation under different column layout types on OmniDocBench.
As shown, Thinking mode provides a more detailed and structured answer by first locating the 4 table regions and understanding each table's content theme, including in-depth analysis of each table's content.

---

## 3. How to Use Layout-as-Thought in Qianfan-OCR
### 3.1 Applicable Scenarios
Layout-as-Thought is suitable for the following scenarios:

* **Complex layout documents**: Documents with multi-column layouts, mixed text and images, and non-standard reading orders (e.g., exam papers, technical reports, academic papers, newspapers)
* **Documents with multiple element types**: Pages simultaneously containing text, formulas, tables, images, charts, and other element types
* **Question answering requiring precise spatial localization**: Scenarios where precise localization of specific regions in complex documents is needed to answer questions

For simple plain-text pages or single-type documents (e.g., plain text forms), it is recommended to use non-Thinking mode for faster inference speed.

### 3.2 How It Works
The Layout-as-Thought workflow is as follows:

```
Document Image + Prompt
        │
        ▼  (Set enable_thinking: true)
        │  (chat_template.jinja automatically appends <think> to the end of the query)
        ▼
┌─────────────────────────┐
│   Thinking Stage          │
│   (Layout Analysis)       │
│                           │
│  For each layout element: │
│  <box>[[x1,y1,x2,y2]]</box>  │
│  <label>element type</label>  │
│  <brief>content summary</brief>│
│  (arranged in reading order)   │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Generation Stage        │
│                           │
│  Based on layout analysis │
│  results, generate final  │
│  Markdown / answer in     │
│  correct reading order    │
│  and element types        │
└─────────────────────────┘
```
**Key Note:** When enabling Layout-as-Thought mode, users only need to set `enable_thinking: true` in the API call parameters — **there is no need to manually add the **`<think>`** tag at the end of the query**. This is because Qianfan-OCR's `chat_template.jinja` template has built-in logic that automatically appends the `<think>` tag at the end of the query when it detects the `enable_thinking: true` parameter, triggering the thinking mode.

### 3.3 Output Format Details
The Thinking stage output is wrapped in `<think>...</think>` tags, containing a list of layout elements arranged in reading order. Each element contains three fields:

#### `<box>` Tag — Bounding Box Coordinates
```
<box>[[<COORD_061>, <COORD_044>, <COORD_282>, <COORD_074>]]</box>
```
* The coordinate format is `[x1, y1, x2, y2]`, representing the top-left and bottom-right corners respectively
* Coordinates use dedicated special tokens `<COORD_0>` to `<COORD_999>`, with values ranging from [0, 999]
* These coordinates are **normalized coordinates**, independent of the actual image resolution
* Each coordinate is represented by a single token (e.g., `<COORD_779>` is one token) — compared to using multi-digit numbers (e.g., "779" requires 3 tokens), this reduces Thinking output length by approximately 50%, which is critical for practical deployment of complex documents (which may contain 60+ layout elements)
* These special coordinate tokens were introduced in Stage 3 training along with the layout analysis data, and the model learned these spatial representations through continual pre-training

**Formula to convert coordinates back to original image pixel dimensions**:

```python
# Assuming image dimensions are img_w x img_h
x1_pixel = int(coord_x1 * img_w / 999)
y1_pixel = int(coord_y1 * img_h / 999)
x2_pixel = int(coord_x2 * img_w / 999)
y2_pixel = int(coord_y2 * img_h / 999)
```
For example: `<COORD_061>` on an image with width 1000px corresponds to pixel position `int(61 * 1000 / 999) ≈ 61px`.

#### `<label>` Tag — Element Type
The model supports 25 fine-grained layout categories, divided into four groups:

|Group|Label|Description|
|-|-|-|
|**Text Elements** (12 types)|`text`|Plain text block|
||`vertical_text`|Vertical text|
||`paragraph_title`|Paragraph title|
||`doc_title`|Document main title|
||`abstract`|Abstract|
||`content`|Body content|
||`reference`|Reference section title|
||`reference_content`|Reference content details|
||`number`|Numbering (e.g., list numbers)|
||`aside_text`|Sidebar text / marginal notes|
||`header`|Header text|
||`footer`|Footer text|
|**Headers & Footers** (4 types)|`header_image`|Header image|
||`footer_image`|Footer image|
||`footnote`|Footnote|
||`vision_footnote`|Footnote related to visual elements|
|**Charts & Figures** (6 types)|`image`|Image|
||`chart`|Chart (bar chart, line chart, etc.)|
||`table`|Table|
||`figure_title`|Figure title / caption|
||`seal`|Seal/stamp|
||`algorithm`|Algorithm block / pseudocode|
|**Formulas** (3 types)|`display_formula`|Display formula (block-level)|
||`inline_formula`|Inline formula|
||`formula_number`|Formula number|

#### `<brief>` Tag — Content Summary
Provides a concise text description for each layout element, for example:

* Text elements: `<brief>Section title "Let's Spell"</brief>`
* Image elements: `<brief>The image shows two scenes: on the left, a white bottle pouring liquid... on the right, a blue bottle pouring liquid...</brief>`
* Table elements: `<brief>Performance comparison of multiple models on tasks including text, formulas, tables, etc.</brief>`

These structured priors guide the subsequent generation stage:

* **Element-type-aware generation**: Selects appropriate output format based on element type — formulas are wrapped in `$...$` or `$$...$$`, tables are converted to HTML, and images use `![label](bbox)` placeholders
* **Reading-order-guided arrangement**: The Thinking stage enumerates all elements in natural reading order (top-to-bottom, left-to-right, left-column-first-then-right-column), ensuring the final output is arranged in the correct order

### 3.4 Service Call Examples
Qianfan-OCR is available through the Baidu Intelligent Cloud Qianfan Platform, using an OpenAI-compatible API interface. **Enabling Layout-as-Thought only requires adding **`enable_thinking: true`** to the request parameters — the model service automatically handles the injection of the **`<think>`** tag, with no manual addition required from the user.**

#### 3.4.1 Document Parsing (Doc Parsing)
```python
import requests
import json
import base64

# ========== Configuration ==========
VLLM_URL = "<Baidu Cloud API service URL>"  # Replace with actual service URL
API_KEY = "<Your API Key>"
MODEL_NAME = "<Model Name>"

# ========== Image to Base64 ==========
def image_to_base64(image_path: str) -> str:
    """Convert image to base64 encoding"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ========== Define Document Parsing Prompt ==========
user_prompt = """
You are an AI assistant specialized in converting document page images (single or multiple pages) extracted from PDFs into Markdown.

Your task is to accurately convert all visible content in the images into Markdown strictly following the rules below. Do not add any explanations, comments, or inferred content.

1. Pages:
- Input may contain one or more document page images.
- You must strictly maintain the page order as provided in the input.
- If multiple pages are included, separate pages using the following marker:
  --- Page N ---
  (N starts from 1)
- If there is only one page, do not output any page separator.

2. Text Recognition:
- Accurately convert all visible text content.
- Do not guess, infer, rephrase, or correct text.
- Preserve the original document structure, including but not limited to: headings, paragraphs, lists, captions, footnotes, etc.
- You must fully preserve header and footer text on each page.

3. Reading Order:
- Read content in top-to-bottom, left-to-right order.
- For multi-column layouts, you must read the left column completely first, then the right column.
- Do not rearrange content order for semantic or logical clarity.

4. Mathematical Formulas:
- Convert all mathematical expressions to LaTeX format.
- Inline formulas must use $...$.
- Display (block-level) formulas must use:

  $$
  ...
  $$

- You must strictly preserve the original symbols, structure, and layout.
- Do not fabricate, simplify, normalize, or correct formula content.

5. Tables:
- All tables must be converted to HTML format.
- Wrap entire tables with <table> and </table>.
- Preserve the original row/column structure, including merged cells (rowspan, colspan) and empty cells.
- Do not reorganize or reinterpret table content.

6. Images:
- Do not describe image content.
- You must preserve all image elements using the following format:
  ![label](<box>[[x1, y1, x2, y2]]</box>)
- Allowed labels include only:
  image, chart, header_image, footer_image, seal
- Do not introduce new labels.
- Do not delete, merge, or rearrange image elements.

7. Unrecognizable or Missing Content:
- If text, symbols, or table cells cannot be recognized, preserve their position and leave the content empty.
- Do not guess or fill in missing content.

8. Output Requirements:
- Output only Markdown content.
- Preserve the original layout, spacing, and structure as much as possible.
- Use appropriate line breaks to clearly separate different elements.
- Do not include any explanatory text, meta-information, or annotations.
""".strip()

# ========== Call Model ==========
def call_model(user_prompt: str, image_path: str, enable_thinking: bool = True) -> str:
    """Call the model for document parsing

    Args:
        user_prompt: The document parsing prompt
        image_path: Path to the document image
        enable_thinking: Whether to enable Layout-as-Thought mode
            When set to True, the server automatically appends <think> to the
            end of the query to trigger thinking mode — no need to manually
            add the <think> tag in the prompt.
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
        "enable_thinking": enable_thinking,  # Enable Layout-as-Thought
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

# ========== Usage Example ==========
image_path = "your_document_page.jpg"

# Thinking mode (Layout-as-Thought enabled)
response_thinking = call_model(user_prompt, image_path, enable_thinking=True)

# Non-Thinking mode
response_non_thinking = call_model(user_prompt, image_path, enable_thinking=False)
```
> **Note**: The `enable_thinking: true` parameter triggers the server-side `chat_template.jinja` template to automatically append the `<think>` tag at the end of the query to activate Layout-as-Thought mode. Users **do not need to** and **should not** manually add the `<think>` tag in the prompt text.
**Extracting the Thinking Part and Final Markdown**:

```python
import re

def extract_thinking(response_text: str) -> str:
    """Extract the Thinking layout analysis content"""
    match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_markdown(response_text: str) -> str:
    """Extract the final Markdown content"""
    if "</think>" in response_text:
        return response_text.split("</think>", 1)[1].strip()
    return response_text.strip()

# Extract layout analysis results
thinking_content = extract_thinking(response_thinking)
# Extract final Markdown
markdown_output = extract_markdown(response_thinking)

print("Layout Analysis Results:")
print(thinking_content[:500])
print("\nFinal Markdown Output:")
print(markdown_output)
```
#### 3.4.2 Document Visual Question Answering (Doc VQA)
```python
def call_model_vqa(image_path: str, question: str, enable_thinking: bool = True) -> str:
    """Call the model for document visual question answering

    Args:
        image_path: Path to the document image
        question: User question
        enable_thinking: Whether to enable Layout-as-Thought mode
            When set to True, the server automatically appends <think> to the
            end of the query — no need to manually add it.
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
        "enable_thinking": enable_thinking,  # Enable Layout-as-Thought
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

# ========== Usage Example ==========
image_path = "your_document_page.png"
question = "What are the four tables in the figure investigating respectively?"

# Thinking mode (Layout-as-Thought enabled)
response = call_model_vqa(image_path, question, enable_thinking=True)

# Extract answer
answer = extract_markdown(response)  # Reuse the extract_markdown function above
print(answer)
```
### 3.5 Parsing Layout Analysis Results and Visualization
The following code extracts bounding box information from the model's Thinking output and draws it on the original image for visualization:

```python
import re
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont

def parse_bbox_from_response(response_text: str) -> List[Dict]:
    """Parse bbox, label, and brief information from Thinking content"""
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
    """Draw bounding boxes and labels on the image"""
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
        # Convert coordinates from [0, 999] back to original image pixel dimensions
        x1 = int(item['bbox'][0] * img_w / 999)
        y1 = int(item['bbox'][1] * img_h / 999)
        x2 = int(item['bbox'][2] * img_w / 999)
        y2 = int(item['bbox'][3] * img_h / 999)
        color = COLOR_MAP.get(item['label'], (255, 165, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{idx+1}. {item['label']}"
        draw.text((x1, y1 - 15), label_text, fill=color, font=font)

    return img

# Usage example
bbox_data = parse_bbox_from_response(response_thinking)
vis_img = draw_bboxes_on_image("your_document_page.jpg", bbox_data)
vis_img.save("visualization.jpg")
```
---

## 4. Training Data Strategy
The construction of training data for Layout-as-Thought is one of the keys to its success. Below we describe the data construction methods for the two tasks.

### 4.1 Layout-as-Thought for Doc Parsing Data
#### Data Construction Pipeline
1. **Layout Detection and Content Recognition**: Use **PaddleOCR-VL** to perform layout detection and content recognition on document pages, obtaining bounding boxes (bbox) and category labels (label) for each document block. Bounding box coordinates are normalized to the [0, 999] range to achieve resolution independence.
2. **Content Brief Generation**: For each detected document block, use **Qwen3-VL-32B** to generate a brief description (brief) — a concise textual summary of the region's content. These brief descriptions serve as part of the layout analysis results, helping the model better understand the semantic content of each region.
3. **Layout-as-Thought Data Formatting**: Assemble the above information into structured layout analysis data within `<think>` tags, containing three fields: `<box>`, `<label>`, and `<brief>`, arranged in reading order, serving as the intermediate reasoning process in the training data.
4. **Final Output**: Combined with layout analysis results, generate the corresponding structured Markdown as the final output annotation.

#### Label System Selection
* The PaddleOCR-VL label system is adopted (rather than MinerU 2.5), as it provides 25 more fine-grained category labels
* PaddleOCR-VL provides fine-grained labels such as `text`, `vertical_text`, `paragraph_title`, `doc_title`, `abstract`, `content`, `reference`, `reference_content`, `aside_text`, etc., while MinerU 2.5 only uses coarser labels like `text`, `title`, `list`, `aside_text`
* Fine-grained labels directly benefit downstream tasks — for example, distinguishing `abstract` from `content` supports structured extraction of academic papers, and distinguishing `reference` from `reference_content` enables clean parsing of bibliographies

#### Training Stages
* **Stage 3 (Domain Enhancement, 800B tokens)**: **Tens of millions** of Layout-as-Thought Doc Parsing data samples were introduced. This stage maintains a 70% domain-specific data + 30% general data mix ratio, enhancing OCR-specific capabilities while preventing catastrophic forgetting. Coordinate special tokens (`<COORD_0>` to `<COORD_999>`) were also introduced at this stage, with the model learning spatial representations through continual pre-training.
* **Stage 4 (Instruction Tuning)**: A **smaller set of higher-quality curated data** was added to further optimize the model's performance on document parsing tasks. This stage constructs instruction data through three complementary strategies: public data collection, instruction rewriting, and reverse synthesis.

#### Lessons Learned
During the implementation of Layout-as-Thought for Doc Parsing, we encountered two typical phased issues, documented here for reference.

**Issue 1: Thinking Mode Metrics Consistently Lower Than Non-Thinking Mode**

Initially, we only introduced a small amount of Layout-as-Thought Doc Parsing data in Stage 4 (instruction tuning), and the data volume was less than that for non-thinking mode. The result was: on OmniDocBench v1.5, thinking mode metrics were consistently lower than non-thinking mode.

The root cause was insufficient thinking data — Stage 4 itself has a small data volume, and a limited amount of thinking-format data was insufficient for the model to fully learn effective layout reasoning in the thinking stage. **Solution**: Introduce large volumes of thinking-mode Doc Parsing data in both Stage 3 and Stage 4, converting all non-thinking mode data into their thinking-mode counterparts. After this adjustment, thinking mode and non-thinking mode achieved comparable metrics on OmniDocBench v1.5.

**Issue 2: Grounding Metrics Dropped After Introducing Coordinate Tokens**

The coordinate special tokens (`<COORD_0>` to `<COORD_999>`) were initially introduced only in Stage 4, with coordinate format modifications applied to all coordinate-containing data. The result was that the model showed noticeably lower performance on Grounding-related benchmarks compared to the baseline version without coordinate token format modifications.

The cause was again a data volume issue: adding 1000 coordinate tokens to the vocabulary requires sufficient data for the model to learn the semantics of these tokens, and the relatively small data volume of Stage 4 was insufficient for adequate training. **Solution**: Move the introduction of coordinate tokens forward to Stage 3, and apply corresponding modifications to all coordinate-containing data in both Stage 3 and Stage 4. After this adjustment, the model's performance on Grounding-related benchmarks matched the unmodified version.

**Summary**: Both issues reveal the same pattern — **introducing new formats or vocabularies requires sufficient data volume to support them**. Making only minor modifications during the instruction tuning stage (Stage 4) is often insufficient; changes need to be moved earlier to the continual pre-training stage (Stage 3) with adequate data coverage.

### 4.2 Layout-as-Thought for Doc VQA Data
#### Data Construction Pipeline
1. **Document Block Selection**: From the document's layout analysis results, **randomly select one or more document blocks** as the information source for QA pairs.
2. **QA Pair Generation**: Use **GPT-5** to design QA pairs based on the content of the selected document blocks:

    * **Text regions**: Generate QA pairs directly based on text content
    * **Chart/figure regions**: First generate an image description of the chart/figure, then design QA pairs based on the description

3. **Data Formatting**: Use the layout analysis results (containing only document blocks relevant to the question) as the `<think>` content, with the answer as the final output.

#### Training Stage
* **Stage 4 (Instruction Tuning)**: A **small amount** of high-quality Doc VQA data is added. Since Doc VQA tasks focus more on comprehension and reasoning abilities, introducing them during the instruction tuning stage yields the best results.

---

## 5. Metric Analysis
Based on experimental results in the paper, Layout-as-Thought achieves better metrics in the following scenarios:

#### Thinking Mode Analysis on OmniDocBench v1.5
|Metric|Non-Thinking Mode|Thinking Mode|Difference|
|-|-|-|-|
|Overall|**93.12**|92.64|-0.48|
|Text^Edit|**0.041**|0.052|+0.011|
|Formula^CDM|**92.43**|91.92|-0.51|
|Table^TEDs|91.02|**91.21**|**+0.19**|
|Table^TEDss|93.85|**94.03**|**+0.18**|
|R-order^Edit|**0.049**|0.051|+0.002|

Although the overall score of Thinking mode is slightly lower (92.64 vs. 93.12), **there are notable improvements in table-related metrics** (Table^TEDs +0.19, Table^TEDss +0.18). More importantly, per-sample analysis reveals key patterns:

#### Analysis by Layout Complexity
In the paper, samples from OmniDocBench v1.5 are sorted by layout label entropy from high to low, and cumulative score curves are plotted (as shown below):

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=15e4eb10993a4bd1930f91330a13715e&docGuid=CQBW8HoJWsu2Y6)
*Figure: Cumulative score curve on OmniDocBench v1.5 (samples sorted by layout label entropy in descending order). In the high-entropy region (left), enabling Thinking mode (solid blue line) consistently outperforms not enabling it (dashed red line), providing a stable score advantage; as low-entropy samples are gradually included, the gap narrows and eventually reverses, with non-Thinking mode achieving a higher total score.*

* **High-entropy region (complex layouts)**: Pages containing mixed text, formulas, tables, images, and other element types — **Thinking mode consistently outperforms non-Thinking mode**, providing a stable score advantage
* **Low-entropy region (simple layouts)**: Pure text, single-type pages — non-Thinking mode performs better, as explicit layout reasoning introduces unnecessary overhead for structurally simple documents and may even interfere with direct recognition

**Practical Guidance**:

* **Documents suited for Thinking mode**: Exam papers, technical reports, academic papers, newspapers, and other complex pages with mixed element types
* **Documents suited for non-Thinking mode**: Pure text pages, simple forms, and other single-type documents

#### Qianfan-OCR Overall Benchmark Results
|Benchmark|Qianfan-OCR Score|Highlights|
|-|-|-|
|OmniDocBench v1.5|**93.12**|#1 among all end-to-end models, surpassing DeepSeek-OCR-v2 (91.09) and Gemini-3 Pro (90.33)|
|OlmOCR Bench|**79.8**|#1 among all end-to-end models|
|OCRBench|**880**|Surpassing Qwen3-VL-4B (873), #1 among all models|
|OCRBenchv2 (zh)|**60.77**|Best Chinese recognition, surpassing all dedicated OCR models|
|CCOCR-multilan|**76.7**|Surpassing Qwen3-VL-4B (74.2), leading in multilingual OCR|
|CCOCR-overall|**79.3**|Surpassing Qwen3-VL-4B (76.5)|
|KIE Overall|**87.9**|Surpassing all commercial models (Gemini-3.1-Pro 79.2) and open-source models|
|DocVQA|92.8|Close to Qwen3-VL-4B (94.9)|
|CharXiv_DQ|**94.0**|Outstanding chart understanding capability|
|CharXiv_RQ|**85.2**|#1 in chart reasoning|
|ChartQA|**88.1**|#1 in chart QA|
|ChartBench|**85.9**|#1 in comprehensive chart evaluation|

---

## 6. Conclusion
Layout-as-Thought is one of the core innovations of Qianfan-OCR. By embedding layout analysis into the model's "thinking" process, it restores the layout analysis capabilities of traditional pipeline systems while maintaining the simplicity of the end-to-end architecture.

**Core Advantages**:

1. **Capability Restoration**: Users can directly obtain structured layout analysis results (element localization, type classification, spatial grounding) from an end-to-end model, bridging the functional gap between end-to-end OCR models and pipeline systems
2. **Accuracy Improvement**: On complex layout documents, explicit structured priors help resolve layout ambiguities, multi-column layouts, non-standard reading orders, and other challenges, with particularly strong performance on table-related metrics
3. **Flexible Control**: Controlled through a simple `enable_thinking: true` parameter toggle, allowing users to flexibly choose whether to enable it based on document complexity

**Usage Recommendations**:

* For complex layouts (multi-column, mixed multi-element) documents, it is recommended to enable Thinking mode for more accurate results
* For simple plain-text pages, non-Thinking mode is sufficient, balancing speed and accuracy
* The layout analysis output from Thinking mode can also serve as input for downstream systems, enabling more flexible document processing pipelines

The Qianfan-OCR model is publicly available through the Baidu Intelligent Cloud Qianfan Platform. For more usage examples and best practices, please refer to: [https://github.com/baidubce/qianfan-models-cookbook/tree/main/qianfan-ocr](https://github.com/baidubce/qianfan-models-cookbook/tree/main/qianfan-ocr)