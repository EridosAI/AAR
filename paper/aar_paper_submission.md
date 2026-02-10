# Association ≠ Similarity: Learning Corpus-Specific Associations for Multi-Hop Retrieval

**Jason Dury**
Eridos AI, Perth, Australia

---

## Abstract

Dense retrieval systems rank passages by embedding similarity to a query, but multi-hop questions require passages that are *associatively related* through shared reasoning chains. We introduce Association-Augmented Retrieval (AAR), a lightweight transductive reranking method that trains a small MLP (4.2M parameters) to learn associative relationships between passages in embedding space using contrastive learning on co-occurrence annotations. At inference time, AAR reranks an initial dense retrieval candidate set using bi-directional association scoring. On HotpotQA, AAR improves passage Recall@5 from 0.831 to 0.916 (+8.6 points) without evaluation-set tuning, with gains concentrated on hard questions where the dense baseline fails (+28.5 points). On MuSiQue, AAR achieves +10.1 points in the transductive setting. An inductive model trained on training-split associations and evaluated on unseen validation associations shows no significant improvement, suggesting that the method captures corpus-specific co-occurrences rather than transferable patterns. Ablation studies support this interpretation: training on semantically similar but non-associated passage pairs degrades retrieval below the baseline, while shuffling association pairs causes severe degradation. A downstream QA evaluation shows retrieval gains translate to +6.4 exact match improvement. The method adds 3.7ms per query, trains in under two minutes on a single GPU, and requires no LLM-based indexing.

---

## 1. Introduction

Consider the question: *"What is the birthplace of the director of Pulp Fiction?"* A retrieval system must find a passage about Quentin Tarantino directing Pulp Fiction and a separate passage about Tarantino's birthplace. The first passage ranks highly because it shares entities with the query. The second, about a person born in Knoxville, Tennessee, bears little surface similarity to a question about film directing. Yet both passages are required for a correct answer.

This failure pattern is systematic in multi-hop question answering. Dense retrievers find passages that resemble the query but miss passages that are needed alongside other retrieved passages to complete a reasoning chain. For multi-hop QA, relevance to the query and relevance to other supporting passages are distinct signals; dense retrieval handles the first well but often misses the second.

The Predictive Associative Memory (PAM) framework (Dury, 2026) provides formal grounding for this distinction, arguing that cosine similarity and learned associative retrieval produce different retrieval rankings. PAM proposes that association recovers items that were experientially co-present, regardless of their perceptual similarity, and predicts that such retrieval should be specific to experienced co-occurrences.

We operationalise this insight as Association-Augmented Retrieval (AAR), a transductive method that learns passage-to-passage associations from co-occurrence annotations and uses them to rerank dense retrieval results. The approach is deliberately minimal: a 4-layer MLP trained with contrastive loss on passage pairs that co-occur as supporting facts for the same question. At inference time, bi-directional association scoring blends learned association strength with cosine similarity to rerank an initial candidate set.

AAR learns associations over the target corpus and is evaluated on questions drawn from that corpus. This mirrors how RAG systems are typically deployed: the retrieval index and any auxiliary structures are built for a specific document collection. We show that this transductive approach improves multi-hop retrieval on both benchmarks tested, while an inductive variant (trained on training-split associations only) does not, providing empirical support for corpus-specific association learning.

Our contributions:

1. Empirically, a lightweight transductive association function improves multi-hop passage retrieval by +8.6 Recall@5 on HotpotQA (without evaluation-set tuning) and +10.1 on MuSiQue, with gains concentrated on questions where dense retrieval fails (+28.5 points on hard questions).

2. Ablations indicate that association and similarity produce opposite effects in multi-hop retrieval: training on semantically similar but non-associated pairs *degrades* performance, while shuffling associative pairs causes severe degradation — both on HotpotQA.

3. An inductive model trained on training-split associations shows no significant improvement on either dataset, pointing to corpus-specific co-occurrence learning.

4. Operationally, the method requires 4.2M parameters, two minutes of training, and 3.7ms per-query overhead, with no LLM-based indexing.

---

## 2. Related Work

### 2.1 Dense Retrieval for RAG

Retrieval-augmented generation (Lewis et al., 2020) conditions language model outputs on retrieved passages, reducing hallucination and enabling knowledge-intensive tasks. Dense Passage Retrieval (DPR; Karpukhin et al., 2020) established the paradigm of encoding queries and passages into a shared embedding space and retrieving by inner product. Modern embedding models such as BGE (Xiao et al., 2024) improve retrieval quality through instruction-tuned contrastive training but remain similarity-based.

### 2.2 Reranking Approaches

Cross-encoder rerankers score query–passage pairs jointly and improve precision over bi-encoder retrieval, but at significant computational cost since every candidate requires a full forward pass. Learned rerankers typically optimise for relevance to the query. AAR scores *inter-passage association* instead, capturing a complementary signal.

### 2.3 Graph-Augmented RAG

GraphRAG (Edge et al., 2024) constructs knowledge graphs from corpora using LLM extraction, enabling structured traversal for multi-hop queries. RAPTOR (Sarthi et al., 2024) builds hierarchical document trees through recursive summarisation. These approaches improve multi-hop retrieval but require extensive LLM-based preprocessing: GraphRAG processes every passage through an entity and relationship extraction pipeline, incurring millions of LLM tokens for corpus indexing. AAR pursues comparable goals at lower computational cost, though direct performance comparison is not possible due to differences in experimental setup. We compare computational cost and architectural complexity only; controlled performance comparisons would require matched conditions.

### 2.4 Neurobiologically-Inspired Approaches

HippoRAG (Gutiérrez et al., 2024) draws on hippocampal indexing theory to construct a knowledge graph where passages are linked through shared entities extracted by an LLM. HippoRAG 2 (Gutiérrez et al., 2025) extends this with improved entity linking and retrieval mechanisms. Both HippoRAG and AAR learn passage-to-passage relationships, but through different mechanisms: HippoRAG builds an explicit knowledge graph via LLM entity extraction, while AAR learns an implicit association function in embedding space. Both are transductive.

### 2.5 Predictive Associative Memory

The PAM framework (Dury, 2026) formalises the distinction between similarity-based and association-based retrieval using JEPA-inspired (LeCun, 2022) predictive architectures. PAM argues that a predictor trained on co-occurrence patterns retrieves items that cosine similarity misses, and predicts that such retrieval should be specific to experienced associations. The present work tests both predictions on standard multi-hop QA benchmarks: transductive association learning improves retrieval on both datasets, while inductive transfer fails.

---

## 3. Method

### 3.1 Problem Formulation

Let $\mathcal{C} = \{p_1, \ldots, p_N\}$ be a corpus of $N$ passages and $e: \mathcal{C} \to \mathbb{R}^d$ an embedding function mapping passages to unit vectors. Standard dense retrieval ranks passages by cosine similarity to a query $q$:

$$\text{sim}(q, p_i) = e(q)^\top e(p_i)$$

Multi-hop questions require passage sets $\{p_a, p_b\}$ where $p_a$ may be similar to $q$ but $p_b$ is connected to $p_a$ through a reasoning chain rather than to $q$ directly.

We define an *association* between passages $p_a$ and $p_b$ as a relationship indicating that both are required to answer the same question — they co-occur as supporting facts. Associated passages need not be similar: a passage about a film director and a passage about a city's demographics may be associated (through the director's birthplace) while being distant in embedding space.

### 3.2 Association Model

We train a function $f: \mathbb{R}^d \to \mathbb{R}^d$ to map passage embeddings into an *association space* where associated passages are close and unassociated passages are distant. The architecture is a 4-layer MLP with LayerNorm, GELU activations, and a learned residual connection:

$$f(x) = \text{normalize}(\alpha \cdot x + (1 - \alpha) \cdot g(x))$$

where $g$ is the MLP transformation, $\alpha$ is a learned scalar blending the input with the transformed output, and the result is L2-normalised to lie on the unit hypersphere. The hidden dimension matches the input dimension (1024), yielding 4,204,545 parameters.

The residual connection preserves the original embedding's information while learning an associative perturbation. The learned $\alpha$ controls how far the output deviates from the input.

### 3.3 Training

Given a set of association pairs $\mathcal{A} = \{(p_a^{(i)}, p_b^{(i)})\}$ derived from co-occurrence annotations (passages that serve as supporting facts for the same question), we train $f$ with a symmetric contrastive loss (Radford et al., 2021). For a batch of $B$ pairs, we compute:

$$s_{ij} = f(e(p_a^{(i)}))^\top f(e(p_b^{(j)})) / \tau$$

where $\tau$ is a temperature parameter. The loss is the average of row-wise and column-wise cross-entropy:

$$\mathcal{L} = \frac{1}{2}\left[\text{CE}(\mathbf{S}, \mathbf{y}) + \text{CE}(\mathbf{S}^\top, \mathbf{y})\right]$$

where $\mathbf{y} = (0, 1, \ldots, B-1)$ are the diagonal targets. With batch size $B = 512$, each positive pair is contrasted against 511 in-batch negatives. We use AdamW with learning rate $3 \times 10^{-4}$, cosine annealing, and temperature $\tau = 0.05$ for 100 epochs. Training completes in approximately two minutes on an RTX 4080 Super.

For the primary (transductive) evaluation, we train on all available co-occurrence pairs from both training and validation splits (20,742 pairs after deduplication). For the inductive evaluation (Section 5.3), we train on training-split pairs only (8,758 pairs). This represents a supervision budget of approximately 311 annotated co-occurrence pairs per 1,000 corpus passages, or approximately 2.8 pairs per question.

### 3.4 Bi-Directional Association Scoring

At inference time, we score query–passage associations bi-directionally. Given a query embedding $e(q)$ and a candidate passage embedding $e(p)$, the association score is:

$$a(q, p) = \frac{1}{2}\left[f(e(q))^\top e(p) + f(e(p))^\top e(q)\right]$$

This differs from the training objective, where both elements pass through $f$ (i.e., $f(e(p_a))^\top f(e(p_b))$). At inference, only one element per directional term is transformed. The query $q$ is not a corpus passage, so applying $f$ to both sides would require the model to handle out-of-distribution inputs.

Table 1 bears this out. Using the training-matched formulation ($f(q) \cdot f(p)$, both-transformed) at inference degrades retrieval, indicating that queries are out-of-distribution for $f$. The reverse direction ($f(p) \cdot q$) carries most of the signal, as expected given that $f$ was trained on passage embeddings. Mixed bi-directional scoring is a conservative choice; reverse-only scoring yields higher R@5 on both HotpotQA (+9.8 vs +8.8) and MuSiQue (+1.3 points above mixed bidi).

**Table 1: Scoring Method Ablation (HotpotQA, Transductive, λ=0.60)**

| Scoring Method | Formula | R@5 | ΔR@5 |
|----------------|---------|-----|------|
| Dense baseline | $e(q) \cdot e(p)$ | 0.831 | — |
| Forward only | $f(e(q)) \cdot e(p)$ | 0.825 | −0.5 |
| Both-transformed | $f(e(q)) \cdot f(e(p))$ | 0.808 | −2.2 |
| Mixed bidi (used) | $\frac{1}{2}[f(q) \cdot p + f(p) \cdot q]$ | **0.918** | **+8.8** |
| Reverse only | $f(e(p)) \cdot e(q)$ | 0.928 | +9.8 |

We report mixed bi-directional scoring throughout as the more conservative formulation. The reverse-only result is noted as a potential improvement.

### 3.5 Retrieval Pipeline

The full retrieval pipeline operates in two stages:

1. **Candidate retrieval.** Use FAISS (exact inner product on L2-normalised vectors) to retrieve the top-$K$ passages by cosine similarity to the query. We use $K = 100$.

2. **Association reranking.** For each candidate $p_i$ in the top-$K$, compute a blended score:

$$\text{score}(q, p_i) = (1 - \lambda) \cdot \text{sim}(q, p_i) + \lambda \cdot a(q, p_i)$$

where $\lambda$ controls the blend between cosine similarity and association strength. The candidates are reranked by this blended score and the top-$k$ are returned. All passage embeddings through $f$ can be precomputed offline, so the per-query cost is limited to one forward pass through $f$ for the query embedding plus $K$ dot products for scoring.

### 3.6 Transductive Evaluation

AAR is transductive: the association model is trained on co-occurrence pairs drawn from the same corpus on which it is evaluated. This parallels how RAG systems are deployed in practice, where auxiliary retrieval structures (FAISS indices, knowledge graphs, entity stores) are built over the target corpus. The cosine similarity baseline is unchanged — it never uses association information — so the comparison remains valid.

We also evaluate an inductive variant (Section 5.3) to test whether learned associations transfer beyond experienced co-occurrences.

---

## 4. Experimental Setup

### 4.1 Datasets

**HotpotQA** (Yang et al., 2018) is a multi-hop QA dataset where each question requires reasoning over exactly two supporting passages. We use the distractor setting of the validation split: 7,405 questions over a corpus of 66,581 unique passages. Each question contributes 2 gold passages and 8 distractor passages, with passage sharing across questions.

**MuSiQue** (Trivedi et al., 2022) is a multi-hop QA dataset with 2-to-4-hop questions requiring reasoning chains of varying depth. We evaluate on 2,417 validation questions over a corpus of 84,459 passages. MuSiQue's deeper reasoning chains (3–4 hops) present a harder association learning problem than HotpotQA's uniform 2-hop structure.

### 4.2 Embedding Model

We use BGE-large-en-v1.5 (Xiao et al., 2024) to encode all queries and passages into 1024-dimensional L2-normalised vectors. All embeddings are precomputed and stored. The association model operates entirely in this embedding space.

### 4.3 Baselines

Our primary baseline is dense cosine retrieval over the full passage corpus using FAISS IndexFlatIP (exact search on normalised vectors). We additionally evaluate BM25 reranking of the same top-100 candidate pool (Section 5.5). We provide contextual cost comparisons with HippoRAG and HippoRAG 2 but do not claim direct performance comparison, as differences in corpus construction, embedding models, and evaluation setup preclude controlled comparison.

### 4.4 Metrics

**Passage Recall@k (R@k):** The fraction of gold supporting passages found in the top-$k$ retrieved passages, averaged over all questions. For HotpotQA, each question has exactly 2 gold passages; R@5 = 1.0 means both are in the top 5.

**Answer Coverage@k:** The fraction of questions where the gold answer string appears in at least one passage in the top-$k$.

**Exact Match (EM) and Token F1:** Downstream QA accuracy using an LLM reader (Section 5.6). EM is 1 if the normalised prediction matches the normalised gold answer; token F1 measures word-level overlap.

**Easy/Hard split:** We partition HotpotQA questions by whether the dense baseline retrieves both gold passages in the top 5. *Easy* questions ($n = 5{,}046$) have R@5 = 1.0 under dense retrieval. *Hard* questions ($n = 2{,}359$) have at least one gold passage outside the top 5.

Throughout the paper, R@k values are rounded to three decimal places for display. All deltas (ΔR@k) are computed from unrounded values and then rounded, so they may differ slightly from arithmetic on displayed values.

### 4.5 Association Data and Corpus Overlap

For HotpotQA, the transductive model trains on 20,742 co-occurrence pairs (combined train and validation splits, deduplicated), representing approximately 311 pairs per 1,000 corpus passages. For MuSiQue, it trains on all available association pairs from the evaluation corpus.

HotpotQA exhibits passage overlap between dataset splits: 61.7% of validation passage IDs appear somewhere in training contexts, and 59.6% of validation gold passage titles appear in the training set's gold passages. However, only 18.0% of validation association pairs are exact duplicates of training pairs — the majority of passage-to-passage links in the validation set are unique, even when individual passages are familiar.

### 4.6 Hyperparameter Selection

The scoring blend parameter $\lambda$ was selected on the HotpotQA evaluation set (the same 7,405 validation questions used for reporting). We disclose this and note that sensitivity is minimal: the optimal $\lambda = 0.60$ yields R@5 = 0.918, while a fixed $\lambda = 0.50$ (requiring no tuning) yields R@5 = 0.916 (+8.6 points over baseline). The full $\lambda$ sweep is reported in Appendix C. For MuSiQue, $\lambda = 0.50$ was used without tuning.

All other hyperparameters (learning rate, temperature, batch size, architecture) were selected during development iterations prior to the final evaluation.

---

## 5. Results

### 5.1 Main Results

Table 2 presents the core results on HotpotQA.

**Table 2: HotpotQA Validation Results (n = 7,405, Transductive)**

| Method | λ | R@5 | R@10 | R@20 | ΔR@5 | 95% CI |
|--------|---|-----|------|------|------|--------|
| Dense baseline | — | 0.831 | 0.878 | 0.913 | — | — |
| AAR (λ=0.50, fixed) | 0.50 | **0.916** | 0.942 | 0.952 | **+8.57** | [+8.11, +9.03] |
| AAR (λ=0.60, tuned on eval) | 0.60 | 0.918 | 0.941 | 0.951 | +8.78 | [+8.30, +9.26] |

The primary result uses a fixed $\lambda = 0.50$ with no hyperparameter selection on the evaluation set: R@5 = 0.916, an improvement of +8.57 points (95% CI [+8.11, +9.03]). Tuning $\lambda$ on the evaluation set (Section 4.6) yields a marginal further gain of 0.21 points. Training uses 20,742 co-occurrence pairs from both dataset splits and achieves 97% training accuracy.

### 5.2 Easy vs. Hard Analysis

AAR's gains are concentrated where the dense baseline fails.

**Table 3: Easy/Hard Subset Results (HotpotQA, Transductive)**

| Subset | n | Dense R@5 | AAR R@5 | ΔR@5 | 95% CI |
|--------|---|-----------|---------|------|--------|
| Easy | 5,046 | 1.000 | 0.996 | −0.44 | — |
| Hard | 2,359 | 0.468 | 0.753 | **+28.51** | [+27.36, +29.65] |

On easy questions, where the dense baseline already retrieves both gold passages, AAR causes negligible degradation. On hard questions, AAR recovers 28.5 additional Recall@5 points. Inspection of 50 rescued questions reveals a dominant pattern: bridge questions where the first gold passage ranks highly in both systems and the second, the "bridge target," is promoted from rank 6–90 in dense retrieval to rank 2–5 by AAR. For example, passages about Ron Dermer (49 → 2), Theatre of the Absurd (65 → 2), the 1964 NY Jets season (90 → 4), and Byron De La Beckwith (58 → 5) were all pulled into the top 5 by AAR after being buried in the dense ranking.

### 5.3 Inductive Evaluation: Association Does Not Generalise

To test whether the association model learns transferable patterns or captures corpus-specific co-occurrences, we train an inductive variant using only training-split pairs (8,758 pairs, with no overlap with validation associations).

**Table 4: Inductive vs. Transductive (HotpotQA)**

| Setting | Training Pairs | Training Acc. | Best λ | R@5 | ΔR@5 | 95% CI |
|---------|---------------|---------------|--------|-----|------|--------|
| Dense baseline | — | — | — | 0.831 | — | — |
| Inductive | 8,758 (train only) | 94.5% | 0.30 | 0.832 | +0.10 | [−0.25, +0.47] |
| Transductive | 20,742 (combined) | 97.2% | 0.60 | **0.918** | **+8.78** | [+8.30, +9.26] |

The inductive model shows no significant improvement (ΔR@5 = +0.10, 95% CI includes zero). Its optimal $\lambda = 0.30$, much lower than the transductive model's 0.60, suggests the association signal is weak and the model falls back to cosine similarity. At $\lambda = 0.50$, the inductive model actively degrades retrieval by 2.8 points (Appendix F).

On the hard subset, the inductive model achieves +6.3 ΔR@5, suggesting some structural transfer for the hardest questions, but far less than the transductive model's +28.5.

**Table 5: MuSiQue Validation Results (n = 2,417)**

| Setting | R@5 | ΔR@5 |
|---------|-----|------|
| Dense baseline | 0.387 | — |
| Inductive | 0.310 | −7.63 |
| Transductive | **0.488** | **+10.12** |

On MuSiQue the same pattern holds: inductive training hurts (−7.6 points), transductive training helps (+10.1 points).

### 5.4 Ablation Studies

Table 6 presents ablations isolating the sources of AAR's improvement. All ablations use the transductive setting.

**Table 6: Ablation Study (HotpotQA, Transductive)**

| Condition | Training Data | R@5 | ΔR@5 |
|-----------|--------------|-----|------|
| Dense baseline | — | 0.831 | — |
| Full AAR | 20,742 co-occurrence pairs | **0.918** | **+8.78** |
| Random negatives | 14,622 pairs, random negatives | 0.915 | +8.43 |
| Similar positives | 50K FAISS-nearest pairs | 0.810 | −2.08 |
| Shuffled associations | 14,622 pairs, shuffled | 0.730 | −10.01 |

The ablations isolate three effects:

**Positive pairs carry the signal, not negative mining** (random negatives). Replacing in-batch negatives with randomly sampled negatives yields R@5 = 0.915, within 0.4 points of the full model. The contrastive loss learns primarily from which passages are associated, not from which are not.

**Similarity hurts** (similar positives). Training the same architecture on semantically similar passage pairs (the 50K nearest neighbours in embedding space) *degrades* retrieval by 2.1 points below baseline. A model that learns "which passages are similar" actively harms multi-hop retrieval.

**Arbitrary pairings fail** (shuffled associations). Randomly permuting the association pairs while preserving the training procedure degrades performance by 10.0 points below baseline (R@5 = 0.730 vs. 0.918). The model does not learn useful representations from arbitrary pairings.

Combined with the inductive evaluation (Section 5.3), these ablations point in the same direction: AAR's improvement comes from learning specific co-occurrence relationships between passages. The inductive failure rules out abstract pattern learning, the similar-positives result rules out similarity, and the shuffled result rules out artefacts of the training procedure.

### 5.5 Comparison with BM25 Reranking

To compare AAR against a non-learned reranking baseline, we evaluate BM25 scoring over the same FAISS top-100 candidate pool.

**Table 7: BM25 vs. AAR Reranking (HotpotQA)**

| Method | Best λ | R@5 | ΔR@5 |
|--------|--------|-----|------|
| Dense baseline | — | 0.831 | — |
| BM25 reranking (best) | 0.10 | 0.838 | +0.76 |
| AAR (transductive) | 0.60 | **0.918** | **+8.78** |

BM25 reranking yields at best +0.76 R@5 (95% CI [+0.41, +1.13]), and harms retrieval at higher blend weights ($\lambda \geq 0.20$). BM25 matches query vocabulary against passage vocabulary — still a similarity measure, just a lexical one. In this candidate-pool setup, neither dense nor lexical similarity-based reranking closes the multi-hop gap.

### 5.6 Downstream QA Evaluation

To verify that retrieval improvements translate to end-to-end question answering, we evaluate 500 randomly sampled HotpotQA validation questions (seed=42) using Claude Sonnet 4 (Anthropic, 2025; model string `claude-sonnet-4-20250514`) as the reader. The reader receives the top-5 retrieved passages with a zero-shot prompt (Appendix G).

**Table 8: Downstream QA Results (n = 500, Claude Sonnet 4 Reader)**

| Condition | EM | F1 |
|-----------|----|----|
| Dense baseline top-5 | 16.6% | 31.1% |
| AAR top-5 | 23.0% | 39.1% |
| Δ | **+6.4** | **+8.1** |

Bootstrap 95% CIs (paired, 10,000 resamples): ΔEM [+3.4%, +9.6%], ΔF1 [+5.4%, +10.8%]. Both intervals exclude zero.

The absolute scores are modest, reflecting the difficulty of multi-hop QA with a zero-shot reader and no chain-of-thought prompting. What matters is the delta: +6.4 EM / +8.1 F1, indicating that the passages AAR surfaces contain information useful for answering the question.

### 5.7 Answer Coverage

**Table 9: Answer Coverage@k (Transductive, Matched HP)**

| k | Dense Baseline | AAR | Δ |
|---|----------------|-----|---|
| 3 | 70.1% | 83.4% | +13.4 |
| 5 | 76.8% | 87.7% | +10.9 |
| 10 | 82.7% | 90.2% | +7.5 |
| 20 | 87.1% | 91.6% | +4.5 |

At $k = 5$, the answer string appears in the retrieved passages for 87.7% of questions versus 76.8% under dense retrieval.

### 5.8 What Did Not Work

Several directions were explored without success. **Inductive training** (Section 5.3) failed on both datasets. **Projection heads** (reducing the association space to 256 dimensions) degraded performance by 2.1 R@5 points. **Cross-dataset transfer** (training on HotpotQA, evaluating on MuSiQue) failed. **Larger models** (6-layer, 2048 hidden, 21M parameters) underperformed the 4-layer model. **Temperature tuning** beyond the initial sweep provided marginal gains, with $\tau = 0.05$ proving robust.

---

## 6. Discussion

### 6.1 AAR as Transductive Retrieval Augmentation

The inductive failure and transductive success, observed on both datasets, position AAR as a corpus-specific retrieval augmentation. Like a FAISS index or knowledge graph, it is built for a specific document collection. The advantage over graph-based alternatives is cost: two minutes of MLP training versus millions of LLM tokens for entity extraction.

### 6.2 Why Inductive Transfer Fails

The inductive model achieves 94.5% training accuracy — it learns the associations it is shown. But these associations do not transfer to unseen passage pairs, even when individual passages overlap between splits (61.7% passage overlap). The best explanation we have is that the MLP learns specific relational mappings between passage embeddings, not features that generalise across co-occurrence boundaries.

Consider the Tarantino example again. A passage about Tarantino and a passage about Knoxville are associated because they co-occur in a question about Tarantino's birthplace. Without experiencing that specific co-occurrence, the model has no basis for the association. This aligns with PAM's prediction that association should be tied to experienced co-occurrences.

### 6.3 Cost Comparison with Graph-Based Methods

Graph-augmented methods such as GraphRAG and HippoRAG require LLM-based entity extraction and relationship annotation during indexing. For a corpus of 66,581 passages, this involves millions of LLM tokens. AAR requires co-occurrence annotations and two minutes of MLP training. This is a comparison of computational cost, not retrieval quality; controlled performance comparisons would require matched experimental conditions.

**Table 10: Latency Breakdown (RTX 4080 Super)**

| Component | Mean (ms) | P95 (ms) |
|-----------|-----------|----------|
| FAISS top-100 | 10.5 | 12.4 |
| AAR bi-directional scoring | 3.7 | 5.9 |
| Total pipeline | 14.1 | 17.6 |

AAR adds 3.7ms mean overhead (33% increase over dense-only retrieval). Total query latency remains under 18ms at P95.

### 6.4 Generating Training Signal Without Gold Annotations

The current implementation uses gold supporting fact annotations as training signal, at a supervision budget of approximately 311 pairs per 1,000 corpus passages. For new corpora without such annotations, alternative sources include LLM-generated multi-hop questions and their supporting passages, citation structure in academic or legal corpora, user interaction data (passages frequently co-accessed within a session), and temporal co-occurrence in streaming document collections. Since AAR is transductive, the quality of these annotations directly determines retrieval quality.

### 6.5 Connection to Predictive Associative Memory

AAR provides empirical support for two PAM predictions. First, the similar-positives ablation (Section 5.4) shows that optimising for similarity degrades multi-hop retrieval, supporting the claim that association and similarity produce different retrieval behaviour. Second, the inductive failure (Section 5.3) on both datasets is best explained by association being tied to experienced co-occurrences. The JEPA-inspired theoretical lineage suggests a broader programme where learned predictors navigate embedding spaces to capture relational structure.

---

## 7. Limitations

We acknowledge several limitations that scope the claims made in this work.

First, AAR is transductive: it requires association annotations over the target corpus. This limits applicability to settings where such annotations can be obtained or generated.

Second, results are reported on two multi-hop QA datasets: HotpotQA (2-hop) and MuSiQue (2–4 hop). The method has not been evaluated on 2WikiMultiHopQA (Ho et al., 2020), CRAG (Yang et al., 2024), or domain-specific corpora.

Third, while we report downstream QA results (Section 5.6), the evaluation uses a zero-shot reader on a 500-question subset. A full-scale evaluation with chain-of-thought prompting or fine-tuned readers would provide a more complete picture.

Fourth, $\lambda$ was selected on the HotpotQA evaluation set (Section 4.6). The sensitivity is minimal (0.21 percentage points), but this should be noted when interpreting the primary result.

Fifth, the MLP achieves 97% training accuracy on HotpotQA's 2-hop associations but only 72% on MuSiQue's 3–4-hop chains, suggesting an architectural ceiling for deeper reasoning.

Sixth, experiments use a single embedding model (BGE-large-en-v1.5). Whether the approach works with other embedding models is untested.

Seventh, the similar-positives and shuffled ablations use 14,622 pairs (validation-split co-occurrences) while the full AAR model uses 20,742 pairs (combined). The pair count difference is a potential confound, though the shuffled ablation — which uses the same 14,622 pairs with randomised pairings — rules out pair count as the driver of improvement.

Eighth, all results are from single training runs. The bootstrap confidence intervals capture evaluation variance but not variance across training seeds. The flat $\lambda$ sensitivity curve (Appendix C) suggests robustness, but multi-seed estimates would strengthen the claims.

---

## 8. Future Work

**Deeper association chains.** The MLP learns single-step associations effectively (97% on 2-hop) but struggles with deeper chains (72% on 3–4-hop). A predictor operating over sets of embeddings — predicting the next hop given a chain of previous hops — could enable iterative traversal of multi-hop paths.

**Alternative training signals.** LLM-generated annotations, citation structure, user interaction data, and temporal co-access patterns could serve as association training signal for corpora without gold annotations.

**Scaling.** Current validation uses corpora of 66K–84K passages. Behaviour at millions of passages is unknown and the top-$K$ expansion approach may require adaptive $K$ selection.

**Toward inductive association.** The inductive failure suggests the MLP learns specific relational mappings. Architectures with explicit relational inductive biases — graph neural networks, attention over passage neighbourhoods, or meta-learning — may enable some degree of transfer.

**Multiple embedding models.** Testing across embedding architectures (E5, GTE, Cohere Embed) would establish whether the approach is embedding-agnostic.

---

## 9. Conclusion

Learning corpus-specific associations improves multi-hop passage retrieval on both benchmarks tested. On HotpotQA, AAR improves Recall@5 by 8.6 points without evaluation-set tuning, with a 28.5-point gain on the hardest questions. On MuSiQue, it achieves +10.1 points. These gains translate to +6.4 exact match and +8.1 F1 in downstream QA.

Transductive training is essential in our experiments: an inductive variant shows no significant improvement on either dataset. The ablations tell a consistent story — training on similar but non-associated pairs degrades retrieval, shuffling association pairs degrades it further, and only real co-occurrence structure produces gains.

AAR is lightweight (4.2M parameters, 3.7ms overhead) and operates as a drop-in reranking stage requiring no LLM-based indexing. For RAG systems where multi-hop retrieval matters and passage co-occurrence annotations are available or can be generated, it offers a practical augmentation to existing dense retrieval pipelines.

---

## References

Anthropic. (2025). Claude Sonnet 4. Model string: `claude-sonnet-4-20250514`. https://www.anthropic.com

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Balestriero, R. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. In *Proceedings of CVPR 2023*.

Dury, J. (2026). Predictive associative memory: Unified retrieval, imagination, and creative recombination through predictive traversal of meaning space. *Zenodo*. https://doi.org/10.5281/zenodo.18595537

Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). From local to global: A graph RAG approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*.

Gutiérrez, B. J., Zhu, Y., Huang, Z., Li, M., & Su, Y. (2024). HippoRAG: Neurobiologically inspired long-term memory for large language models. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*.

Gutiérrez, B. J., Zhu, Y., Huang, Z., & Su, Y. (2025). From RAG to memory: Non-parametric continual learning for large language models. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*.

Ho, X., Nguyen, A.-K., Sugawara, S., & Aizawa, A. (2020). Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In *Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)*.

Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-t. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)*.

LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview preprint*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. In *Proceedings of the 38th International Conference on Machine Learning (ICML 2021)*.

Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, L., Holzleitner, M., Pavlović, M., Sandve, G. K., Unterthiner, T., & Hochreiter, S. (2021). Hopfield networks is all you need. In *Proceedings of the 9th International Conference on Learning Representations (ICLR 2021)*.

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389.

Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024). RAPTOR: Recursive abstractive processing for tree-organized retrieval. In *Proceedings of the 12th International Conference on Learning Representations (ICLR 2024)*.

Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). MuSiQue: Multihop questions via single hop question composition. *Transactions of the Association for Computational Linguistics, 10*, 539–554.

Weston, J., Chopra, S., & Bordes, A. (2015). Memory networks. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*.

Xiao, S., Liu, Z., Zhang, P., & Muennighoff, N. (2024). C-Pack: Packaged resources to advance general Chinese embedding. In *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024)*.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018)*.

Yang, X., Sun, K., Xin, H., Sun, Y., Bhalla, S., Chen, X., Ghosh, S., Li, S., Srinivasan, J., Feng, T., & others. (2024). CRAG — Comprehensive RAG benchmark. *arXiv preprint arXiv:2406.04744*.

---

## Appendix A: Symbol Reference

| Symbol | Meaning |
|--------|---------|
| $e(\cdot)$ | Embedding function (BGE-large-en-v1.5) |
| $f(\cdot)$ | Association model (4-layer MLP) |
| $g(\cdot)$ | MLP transformation (before residual) |
| $\alpha$ | Learned residual weight in MLP |
| $\lambda$ | Scoring blend parameter (cosine vs. association) |
| $\tau$ | Temperature in contrastive loss |
| $K$ | Candidate pool size (FAISS expansion depth) |
| $k$ | Number of passages returned after reranking |

## Appendix B: Candidate Pool Sensitivity

**Table B1: R@k as a Function of FAISS Expansion Depth (Transductive)**

| Depth $K$ | R@5 | ΔR@5 | R@10 | R@20 |
|-----------|-----|------|------|------|
| 10 | 0.870 | +3.9 | 0.878 | 0.878 |
| 20 | 0.893 | +6.2 | 0.909 | 0.913 |
| 50 | 0.909 | +7.8 | 0.930 | 0.939 |
| 100 | 0.917 | +8.6 | 0.941 | 0.951 |
| 200 | 0.921 | +9.0 | 0.947 | 0.959 |

Performance increases with expansion depth but with diminishing returns. Depth 200 adds only +0.4 R@5 over depth 100 while doubling the scoring cost. At $K = 10$, R@10 and R@20 cannot exceed the dense baseline's values because the reranked candidate set contains only 10 passages. Note: all rows use the same model; the K=100 R@5 of 0.917 differs from Table 2's 0.916/0.918 due to minor differences in $\lambda$ selection across experimental runs.

## Appendix C: Scoring Blend Parameter

**Table C1: R@5 as a Function of λ (Transductive, Matched HP)**

| λ | R@5 | ΔR@5 |
|---|-----|------|
| 0.30 | 0.894 | +6.4 |
| 0.40 | 0.907 | +7.7 |
| 0.50 | 0.916 | +8.6 |
| 0.60 | **0.918** | **+8.8** |
| 0.70 | 0.915 | +8.4 |

The curve is flat across 0.40–0.70, with less than 1.2 points separating the best and worst values in this range. The inductive model's optimal $\lambda = 0.30$ (Appendix F) reflects its weaker association signal.

## Appendix D: Bootstrap Confidence Intervals

All confidence intervals are computed using paired bootstrap resampling with 10,000 iterations. Each resample draws $n$ questions with replacement from the evaluation set and computes the metric of interest (R@5 delta between conditions) on the resampled set. The 95% CI is the 2.5th and 97.5th percentiles of the bootstrap distribution.

**Table D1: 95% Confidence Intervals (10,000 Paired Bootstrap Resamples)**

| Comparison | ΔR@5 | 95% CI |
|------------|------|--------|
| AAR (transductive, λ=0.50) vs. baseline, overall | +8.57 | [+8.11, +9.03] |
| AAR (transductive, λ=0.50) vs. baseline, hard subset | +27.47 | [+26.37, +28.57] |
| AAR (transductive, λ=0.60) vs. baseline, overall | +8.78 | [+8.30, +9.26] |
| AAR (transductive, λ=0.60) vs. baseline, hard subset | +28.51 | [+27.36, +29.65] |
| AAR (inductive) vs. baseline, overall | +0.10 | [−0.25, +0.47] |
| QA ΔEM | +6.4 | [+3.4, +9.6] |
| QA ΔF1 | +8.1 | [+5.4, +10.8] |

## Appendix E: Error Taxonomy

Inspection of 50 questions rescued by AAR (where dense fails but AAR succeeds in top-5) and 50 questions where both methods fail.

**Rescued by AAR (n = 1,296 total; 50 inspected).** The dominant pattern is bridge questions where the first gold passage ranks highly in both systems and the second "bridge target" is promoted from rank 6–90 in dense retrieval to rank 2–5 by AAR. Typical rank improvements: Ron Dermer (49 → 2), Theatre of the Absurd (65 → 2), 1964 NY Jets season (90 → 4), Byron De La Beckwith (58 → 5).

**Still missed by both (n = 1,063 total; 50 inspected).** Three failure modes: approximately 40% have at least one gold passage absent from the FAISS top-100 (unreachable by any reranking method); approximately 30% have gold passages in the top-100 but AAR cannot promote them enough; approximately 30% involve common-entity confusion where the correct passage is lost among many candidates about popular entities.

## Appendix F: Inductive Lambda Sensitivity

**Table F1: Inductive Model R@5 as a Function of λ**

| λ | R@5 | ΔR@5 |
|---|-----|------|
| 0.30 | **0.832** | **+0.10** |
| 0.40 | 0.822 | −0.82 |
| 0.50 | 0.803 | −2.77 |
| 0.60 | 0.770 | −6.01 |
| 0.70 | 0.717 | −11.31 |

Performance degrades monotonically as $\lambda$ increases, indicating the inductive model's association signal is noisy.

## Appendix G: QA Evaluation Details

**Reader model:** Claude Sonnet 4 (Anthropic, 2025), model string `claude-sonnet-4-20250514`.

**Decoding parameters:** temperature=0 (deterministic), max_tokens=100.

**Prompt template:**
```
System: Answer the question using only the provided passages. 
Give only a short factual answer, nothing else.

User: Passages:

[Passage 1 text]
---
[Passage 2 text]
---
[Passage 3 text]
---
[Passage 4 text]
---
[Passage 5 text]

Question: [question text]
```

**Normalisation:** Lowercase, strip articles (a, an, the), strip punctuation, collapse whitespace. Applied to both prediction and gold answer before computing EM and F1.

**Sample:** 500 questions, randomly sampled with seed=42. Single run (deterministic decoding).
