# The Evolution of VLM Document Parsing Architectures: From "Sensory Precision" to "Logical Alignment" and the Fight for Pixel Sovereignty

## 1. The Industry Inflection Point: Why "Semantic Error Correction" Is No Longer a Silver Bullet

As VLMs enter the "deep water zone," the industry has reached a harsh consensus: **if the Vision Encoder loses the subscripts/superscripts of $\sum$ or the dashed borders of a table at the input stage, no amount of LLM parameters at the backend can compensate — it's building castles on sand.**

We once relied on the powerful semantic association capabilities of LLMs to complete ambiguous OCR results. However, in zero-tolerance scenarios such as financial statements and scientific papers, this "association" often degenerates into fatal "hallucinations."

Qianfan-OCR achieved a score of **93.12** on the rigorous **OmniDocBench v1.5** benchmark, leading all end-to-end models. The core logic behind this achievement: **rather than patching things up at the backend, it's better to achieve "perceptual sovereignty" over raw pixels at the frontend.**

---

## 2. The Spatial Discretization Path: The "Pixel Brute-Force Aesthetics" of AnyRes

**Representative models:** Qianfan-OCR (arXiv:2603.13398), InternVL 2.5

### 2.1 The "Low-Fidelity" Dilemma of High Resolution

The traditional AnyRes (AnyResolution) architecture preserves detail by splitting images into $448 \times 448$ patches. However, many models set very low patch count minimums (e.g., 1 or 4) when processing simple documents to save tokens.

**Qianfan-OCR's Engineering Intuition:**
When parsing dense documents (such as newspapers and audit reports), we found that while aspect ratio adaptation is important, **sampling density** is the true lifeline.

![](images/image_1.jpg)

### 2.2 Why Is AnyRes More Advantageous in Production?

Although NaViT theoretically offers higher pixel utilization, AnyRes demonstrated exceptional **production stability** in Qianfan-OCR's technology selection:

1. **Global-Local Dual-Path Navigation:** AnyRes mandatorily retains a low-frequency global stream (Thumbnail) that provides an invaluable "bird's-eye view." This is critical for understanding macro structures like cross-column layouts and full-width headlines, whereas NaViT's pure sequential mode can sometimes get lost in long local sequences.
2. **Operator Maturity:** AnyRes is fully compatible with standard TensorRT acceleration and fixed-size operator optimization. On a 4B model that pursues the ultimate inference cost-performance ratio, AnyRes delivers more stable throughput.

---

## 3. Deep Dive: The "Myth" vs. "Reality" of Parsing Complex Newspaper-Style Data

There was once a belief in the industry that extreme data like newspapers — with their elongated aspect ratios and multi-column layouts — could only be handled by NaViT's elastic packing (Native Resolution). **Qianfan-OCR shattered this myth.**

### 3.1 Pixel Density vs. Aspect Ratio Adaptation

The pain point of newspaper parsing is not that it's "long," but that it's "dense." An A3-sized newspaper can contain tens of thousands of characters.

* **NaViT's Theoretical Advantage and Practical Risk:** Although NaViT allows setting pixel upper and lower bounds, its core logic is "geometric adaptation." When processing newspapers with extreme aspect ratios, to maintain global geometric relationships, sampling points are often diluted across lengthy pixel sequences. Under a fixed token budget, NaViT easily falls into the awkward situation of "seeing everything, but with detail features averaged out."
* **Qianfan-OCR's Enhanced AnyRes:** We abandoned the pursuit of extreme pixel utilization and opted for a **"perceptual floor" strategy**. By forcibly raising $min\_patch$ to 8, we reserve at least 2,048 visual tokens as the "base salary" for every document.

    The ingenuity of this design lies in the fact that even for newspapers with the most complex layouts, each atomic patch can be encoded under the native field of view of $448 \times 448$. This means we are not "scaling down" the newspaper; instead, we are using 8 to 24 "high-powered microscopes" to scan it. This brute-force aesthetics of high sampling density physically eliminates the "semantic hallucinations" caused by visual ambiguity.

![](images/image_2.jpg)

---

## 4. In-Depth Technical Comparison: AnyRes vs. NaViT

| Dimension | AnyRes (Qianfan-OCR) | NaViT (Native Geometry Path) |
|---|---|---|
| **Core Logic** | High-frequency sampling + global macro navigation | Geometric reconstruction + low-redundancy perception |
| **Small/Dense Characters** | Extremely strong (mandatory minimum patch count, zero omission for dense text) | Constrained by token budget; sampling density decreases for large images |
| **Newspapers/Complex Layouts** | Global thumbnail anchors layout; cross-region semantic coherence | Accurate geometric reconstruction, but lacks global view; multi-region semantics risk fragmentation |
| **Production Deployment** | Standard operators, easy quantization, high inference throughput | Variable-length sequences require custom Attention operators; higher deployment cost |
| **Typical Performance** | Leading accuracy in formula, table, and chart structural parsing | Stronger geometric robustness for tilted/non-standard aspect ratio scenarios |

![AnyRes vs NaViT Comparison](images/image_3.png)
*Source: https://arxiv.org/pdf/2603.13398*

---

## 5. The Endgame of Evolution: From Perceptual Precision to Logical Constraints

Improving the "sensory precision" of visual perception is only the first step. Qianfan-OCR's future evolution will focus on **"logical alignment."**

1. **Architecture Side:** We plan to introduce a **Token Packing** mechanism on top of AnyRes, maintaining high resolution while masking out invalid padding regions to further squeeze inference performance.
2. **Alignment Side:** Drawing on the **GRPO (Group Relative Policy Optimization)** paradigm, we will introduce structural validity constraints during the generation phase.

    * **No longer just "image captioning":** Through reinforcement learning reward models, we force the model to close tags when outputting tables and follow LaTeX syntax when outputting formulas.
    * This mechanism ensures that when processing noisy scenarios (e.g., blurring, glare), even if perception is slightly impaired, the model can still generate correct document representations based on logical consistency.

---

## 6. Conclusion

From Qianfan-OCR's practice, the decisive factor in VLM document parsing is not architectural "alchemy," but rather **the relentless pursuit of engineering boundaries.**

We chose AnyRes and pushed it to its performance peak through an $8 \sim 24$ patch strategy because, under current hardware and application environments, **"seeing clearly" will always be more important than "pretty layout."** Moving forward, we will continue to maintain pixel sovereignty while exploring the logical constraints brought by reinforcement learning, enabling VLMs to truly understand every complex document.

---
