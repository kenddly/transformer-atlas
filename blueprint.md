# From Transformer Fundamentals to Cell Segmentation and Tracking

A comprehensive, implementation-led learning and research curriculum.

**Edition:** Public educational edition. Personal learning context has been omitted; the technical curriculum and research guidance are retained.

**Document status:** educational blueprint and curriculum. Proposed repository components, model training, datasets, and experiments remain planned until separately implemented and verified. The companion website provides lesson text and small educational demonstrations; it does not establish completion of the full research pipeline.

**Research context reviewed:** 17 October 2026. Verify paper versions, software APIs, checkpoint provenance, and available datasets when implementation begins.

**Navigation:** [Purpose](#1-purpose-and-intended-outcome) · [Prerequisites](#2-learning-context-and-prerequisites) · [Teaching method](#3-how-the-future-teaching-project-should-operate) · [Roadmap](#5-roadmap-and-dependencies) · [Notation](#6-notation-and-tensor-conventions) · [Cross-domain comparison](#11-a-shared-explanation-across-nlp-vision-and-cells) · [Coverage checklist](#12-detailed-component-coverage-checklist) · [Research readings](#14-research-paper-study) · [Experiments](#16-experiment-sequence-and-ablation-matrix) · [Starting guide](#22-starting-guide-for-learners).

## 1. Purpose and intended outcome

Build a learning repository that explains how transformers emerged, how their components work mathematically and computationally, how language and vision use the same underlying operations differently, and how those ideas become useful in microscopy segmentation and cell tracking.

The project should develop understanding through four connected activities:

1. Read the research question and historical motivation.
2. Derive the relevant computation with explicit dimensions.
3. Implement and inspect a small working example.
4. Test a hypothesis through an experiment, including a failure case.

The destination is an explainable, reproducible microscopy learning pipeline: images become object instances; instances become tokens or track queries; learned scores and decoding rules become trajectories and lineages; evaluation checks whether the results support biological measurements.

By completion, the learner should be able to:

- Explain the transition from recurrent sequence models to attention and transformers.
- Derive attention, its gradients, and its computational costs.
- Implement a small transformer without hiding the central operations behind a high-level transformer layer.
- Explain masking, positional representations, residual paths, normalization, feed-forward networks, training objectives, and inference.
- Map language tokens, image patches, volumetric patches, and cell detections onto compatible tensor representations.
- Distinguish a feature backbone from a prediction head, mask reconstruction procedure, and tracking decoder.
- Explain ViT, hierarchical vision transformers, object queries, promptable segmentation, and detection-token tracking.
- Build classical and learned segmentation and tracking baselines.
- Compare transformer methods with those baselines under defensible experimental conditions.
- Read CBIA papers critically and discuss a bounded research proposal without inventing novelty.

This is a sequence for developing mastery. Large-scale pretraining is an optional research extension; it is not a prerequisite for understanding the architecture.

## 2. Learning context and prerequisites

This public educational edition is designed for learners building a first-principles understanding of transformers and microscopy analysis. No prior microscopy project or complete knowledge of transformer conventions is assumed.

Before beginning, record brief explanations of attention, image segmentation, object association, and lineage reconstruction. Review these explanations after the related modules to make changes in understanding visible.

Useful starting skills include basic Python, ordinary array operations, and a willingness to trace small numerical examples. Module 01 introduces the mathematical and numerical tools used throughout. Learners should revisit prerequisites whenever an axis, gradient, or probability convention is unclear.

The priorities are transformer internals, microscopy targets, instance separation, lineage reconstruction, and defensible evaluation. Begin with intuitive examples, derive the computation, then implement and inspect a limited experiment.

Carry a consistent experimental discipline through the curriculum: a mechanism can improve one endpoint while harming another. Distinguish mask quality from lineage quality, attention visualizations from causal explanations, educational demonstrations from reproductions, and deterministic reruns from statistical replication. Measure tradeoffs rather than assuming that a better score on one endpoint establishes a better complete system.

## 3. How the future teaching project should operate

### 3.1 Lesson contract

Every substantive lesson should contain:

1. **Question:** the concrete problem being solved.
2. **Intuition:** a plain-language account using a tiny example.
3. **Mechanism:** the relevant computation and information flow.
4. **Tensor trace:** every axis, shape, projection, and reshape.
5. **Numerical example:** small enough to calculate by hand.
6. **Implementation exercise:** a limited, inspectable task.
7. **Failure demonstration:** a broken mask, misleading result, or counterexample.
8. **Research connection:** where the idea appears in a primary paper.
9. **Understanding check:** explanation or prediction before execution.
10. **Evidence:** saved configuration, outputs, and a short interpretation.

Explain a symbol before using it. Expand an acronym on first use. Separate established facts, paper-specific choices, and proposed experiments. Present results as measured only after the experiment has actually run.

### 3.2 Depth levels

Use three depth levels inside each lesson:

- **Essential:** intuition, information flow, and a working minimal example.
- **Technical:** equations, gradients, numerical stability, and implementation details.
- **Research:** assumptions, alternative designs, ablations, generalization, and open questions.

Essential material is sequential. Technical material should follow while the example is still familiar. Research extensions can branch after the learner can explain the underlying mechanism.

### 3.3 Implementation philosophy

Use ordinary tensor operations and automatic differentiation first. Implement attention projections, masking, head reshaping, and the transformer block explicitly. Later compare against optimized library implementations.

The objective is to understand the computation, not to rewrite GPU kernels, tokenization infrastructure, image readers, or scientific optimization solvers.

Start with CPU-sized examples. Profile memory and one training step before scaling. Select a maintained Python/PyTorch environment after inspecting the actual machine; record resolved versions rather than installing an assumed "latest" stack.

### 3.4 What counts as passing a module

A module passes when the learner can:

- Explain why the component exists.
- Trace its inputs and outputs.
- Predict the effect of changing one design choice.
- Diagnose at least one meaningful failure.
- Reproduce the small experiment from a saved configuration.

A successful loss curve alone does not establish understanding. A carefully explained negative experiment can satisfy a research milestone.

## 4. Proposed repository organization

The following tree is a future design, not a list of files already created:

~~~text
transformers-to-cell-lineages/
  README.md
  pyproject.toml
  environment/
    resolved-dependencies.txt
    hardware-and-runtime.md
  lessons/
    00-orientation.md
    01-mathematical-tools.md
    ...
    24-capstone.md
  notebooks/
    01-tensors-and-gradients.ipynb
    02-recurrent-sequences.ipynb
    03-attention-by-hand.ipynb
    04-masks-and-causality.ipynb
    05-transformer-block.ipynb
    06-language-objectives.ipynb
    07-positions-and-geometry.ipynb
    08-image-patches.ipynb
    09-touching-cell-instances.ipynb
    10-set-prediction.ipynb
    11-detection-token-tracking.ipynb
    12-lineage-decoding.ipynb
  src/
    fundamentals/
      attention.py
      normalization.py
      positional.py
      blocks.py
      masking.py
    language/
      tokenization_lab.py
      causal_model.py
      encoder_decoder.py
    vision/
      patches.py
      vit.py
      dense_heads.py
      set_prediction.py
    microscopy/
      synthetic.py
      instance_targets.py
      reconstruction.py
      detections.py
      association.py
      lineage.py
      metrics.py
    training/
      loops.py
      checkpoints.py
      reproducibility.py
  configs/
    educational/
    baselines/
    capstone/
  data/
    README.md
    manifests/
    splits/
  research/
    reading-ledger.md
    paper-cards/
    hypothesis-ledger.md
    decisions.md
    limitations.md
  experiments/
    registry.csv
    summaries/
  reports/
    figures/
    error-galleries/
    final-report.md
  tests/
    attention-and-masks/
    data-and-labels/
    lineage-invariants/
~~~

Keep learning notebooks readable and put reused model code in modules. Store dataset provenance and split manifests separately from raw data. Save the meaningful results and configurations of an experiment, not every temporary artifact.

## 5. Roadmap and dependencies

| Module | Central question | Main deliverable |
|---|---|---|
| 00 | What are we trying to learn and measure? | Baseline knowledge and project plan |
| 01 | Which mathematical operations are required? | Tensor and gradient notebook |
| 02 | Why did attention emerge? | Recurrent seq2seq comparison |
| 03 | How does text become vectors? | Tokenization and embedding lab |
| 04 | What does attention actually calculate? | Hand-derived attention and backward pass |
| 05 | How do masks control information? | Causality and padding demonstrations |
| 06 | What else is inside a transformer block? | Explicit block implementation |
| 07 | How are encoder, decoder, and objectives related? | Small language-model experiments |
| 08 | How is position represented? | Position and geometry lab |
| 09 | How are transformers trained and executed efficiently? | Training and inference profile |
| 10 | How does an image become a token sequence? | Patchification and tiny ViT |
| 11 | Why use hierarchy, windows, or pretraining? | Backbone comparison |
| 12 | What does cell segmentation need to recover? | Classical and U-Net baselines |
| 13 | How does a transformer produce dense labels? | Shared-head segmentation comparison |
| 14 | How do object queries and set losses work? | Tiny DETR-style set predictor |
| 15 | How do mask queries and prompts work? | Mask/prompt interpretation lab |
| 16 | What changes in 3D microscopy? | Small volumetric study |
| 17 | What changes when time is added? | Spatiotemporal token comparison |
| 18 | What must tracking solve before attention? | Association baselines |
| 19 | How do cells become transformer tokens? | Detection-token data pipeline |
| 20 | Can attention learn better associations? | Small contextual association model |
| 21 | How do scores become valid lineages? | Division-aware graph decoder |
| 22 | Can tracking help choose segmentation? | Joint-hypothesis experiment |
| 23 | How should the biological task be evaluated? | Evaluation and uncertainty audit |
| 24 | What bounded research claim survives testing? | Reproducible capstone and explanation |

~~~mermaid
flowchart TD
    A["Math, sequence history, embeddings"] --> B["Attention, masks, blocks, objectives"]
    B --> C["Positions, training, inference"]
    C --> D["Image tokens and ViT"]
    D --> E["Hierarchy and pretraining"]
    E --> F["Dense labels, sets, masks, prompts"]
    F --> G["3D and video representations"]
    B --> H["Classical association and cell tokens"]
    G --> I["Contextual association and lineages"]
    H --> I
    I --> J["Joint hypotheses, evaluation, capstone"]
~~~

## 6. Notation and tensor conventions

Agree on conventions before writing models. Use different symbols for image height and attention-head count.

| Symbol | Meaning |
|---|---|
| B | Batch size |
| T | Text sequence length, unless a lesson explicitly uses T for time |
| N | Number of tokens or detections |
| D | Model embedding width |
| h | Number of attention heads |
| d_h | Width of one head; commonly D/h |
| T_q, T_k | Query length and key/value length |
| V_vocab | Vocabulary size |
| C | Image channels |
| H, W | Image height and width |
| Z | Volume depth |
| F | Number of video frames |
| P, P_z, P_y, P_x | Patch dimensions |
| K_queries | Number of object or track queries |

Common layouts:

~~~text
Token IDs:                [B, T]
Token embeddings:         [B, T, D]
One-head Q:               [B, T_q, d_h]
One-head K and V:         [B, T_k, d_h]
Multihead Q/K/V:          [B, h, T, d_h]
Attention scores:         [B, h, T_q, T_k]
2D image:                 [B, C, H, W]
3D volume:                [B, C, Z, H, W]
Video, chosen convention: [B, F, C, H, W]
Padded cell tokens:       [B, N_max, D]
Association scores:       [B, N_source, N_target]
~~~

The equal Q/K/V head width above is a convenient self-attention convention. In general, Q and K must share their compatibility width, while V can have a different content width. Cross-attention can also project streams with different original feature widths.

Every lesson must state whether time belongs in the batch, token sequence, or a separate axis. A reshape changes organization; it does not create or remove information by itself. A projection can change information and dimension.

## 7. Modules 00–03: prerequisites and historical motivation

### Module 00 — Orientation and a concrete microscopy example

**Question:** What should a complete system produce?

Use a synthetic movie containing two moving nuclei. Later add a division, a temporary missing detection, and a crowded region. Define image intensity, an instance label image, detections, tracks, and parent–daughter relationships.

Distinguish:

- A physical cell from its nucleus.
- A biological object from the structure highlighted by a fluorescent label.
- Foreground segmentation from individual instances.
- Per-frame instance numbers from persistent track identities.
- A segmentation split from biological division.
- A merged mask from actual cell fusion.
- Appearance/disappearance in an image from biological birth/death.

Record the learner's current explanations of attention, segmentation, and tracking before teaching them. Revisit the same questions later to make progress visible.

**Build later:** a tiny synthetic data generator with reference masks and lineages.

**Pass when:** each output can be described independently and one error can be followed from mask to biological measurement.

**Read:** [CTC overview](https://celltrackingchallenge.net/), [CBIA organoid example](https://cbia.fi.muni.cz/research/spatiotemporal/organoids.html).

### Module 01 — Mathematical and numerical tools

**Question:** What operations will every later diagram depend on?

Teach and implement:

- Vectors, matrices, dot products, matrix multiplication, and broadcasting.
- Linear projections versus elementwise nonlinear functions.
- Tensor indexing, transpose, permutation, reshape, contiguous storage, and views.
- Probability distributions, logits, softmax, log-softmax, cross-entropy, and entropy.
- The chain rule, automatic differentiation, and gradient accumulation.
- Training/evaluation modes and why dropout changes behavior.
- Floating-point precision, numerical tolerances, and overflow/underflow.

For a softmax row z, calculate softmax(z - max(z)) and show that subtracting the row maximum preserves the probabilities while improving numerical stability.

Use a two-layer network to trace gradients by hand, then compare automatic differentiation with finite differences on a very small problem. Explain why finite differences become unreliable with poor step sizes or low precision.

Separate four quantities throughout: parameter count, activation storage, arithmetic operations, and elapsed runtime. They are related but not interchangeable.

**Build later:** a shape-tracing utility and a notebook of small numerical checks.

**Pass when:** the learner can predict a matrix-product shape and explain the source of each gradient.

**Read:** [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html), [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html), [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html).

### Module 02 — From recurrence to attention to transformers

**Question:** Which problems motivated the architectural changes?

Construct the history as a sequence of design problems, rather than a claim that one invention made all previous methods obsolete:

| Milestone | Design issue to understand | Reading |
|---|---|---|
| Recurrent models and LSTM | State propagated through time; gated control of information and gradients | [Original LSTM paper](https://doi.org/10.1162/neco.1997.9.8.1735) |
| Sequence-to-sequence learning, 2014 | Encode one sequence and generate another | [Sutskever et al.](https://arxiv.org/abs/1409.3215) |
| Attention for translation, 2014 preprint / 2015 publication | Let each decoder step access different encoder states | [Bahdanau et al.](https://arxiv.org/abs/1409.0473) |
| Transformer, 2017 | Organize sequence interactions around attention without recurrent layers in the proposed architecture | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |

Explain an RNN hidden-state update, then introduce LSTM input, forget, and output gates and its cell-state pathway. Gates help control information flow; they do not guarantee perfect long-term learning.

Use translation as the motivating example: a decoder may need different source words at different output steps. Bahdanau-style attention constructs a context from weighted encoder states while retaining recurrence.

Draw the original transformer encoder, masked decoder self-attention, and encoder–decoder cross-attention. Identify which components remain besides attention: embeddings, position information, feed-forward networks, normalization, residual connections, and output projections.

**Build later:** small copy/reversal tasks with recurrent seq2seq models, with and without attention. Treat them as demonstrations, not reproductions of translation results.

**Pass when:** the learner can explain why attention predates transformers and which operations remain sequential during autoregressive generation.

### Module 03 — Tokens, vocabulary, and embeddings

**Question:** How does discrete text become differentiable computation?

Compare character, byte, word, and subword units. Explain the vocabulary-size/sequence-length tradeoff, normalization, special tokens, beginning/end markers, and padding.

Perform several byte-pair encoding merges by hand. Explain that tokenization constructs discrete units; embedding learning assigns trainable vectors to those units. SentencePiece supports tokenization approaches rather than naming one universal algorithm.

An embedding table has shape [V_vocab,D]. Integer IDs [B,T] select rows, giving [B,T,D]. ID 100 is not semantically closer to ID 101 than to ID 900 because of their numerical values.

Distinguish a context-independent input embedding from a contextual hidden state after transformer layers. Trace how repeated appearances of a token accumulate gradients in the same embedding row.

**Build later:** a toy tokenizer and embedding lookup; inspect English phrases, common names, and microscopy terms. Train learned tokenization only on the training corpus.

**Pass when:** token IDs, input vectors, contextual features, and output vocabulary logits can be explained separately.

**Read:** [Subword units](https://arxiv.org/abs/1508.07909), [SentencePiece](https://arxiv.org/abs/1808.06226), [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html).

## 8. Modules 04–09: transformer architecture in detail

### Module 04 — Attention from a numerical example

**Question:** How does one token gather information from other tokens?

Start with three two-dimensional input vectors. Choose small projection matrices and calculate every intermediate quantity before batching or adding multiple heads.

For one head:

$$
Q=X_qW_Q,\quad K=X_kW_K,\quad V=X_kW_V
$$

$$
S=\frac{QK^\top}{\sqrt{d_h}}+M,\qquad
A=\operatorname{softmax}_{\text{keys}}(S),\qquad O=AV.
$$

Explain each role:

- Q describes the learned compatibility features of a receiving position.
- K supplies compatibility features of candidate source positions.
- V supplies the content to mix.
- Each row of A weights source positions for one query.
- M controls which interactions are allowed.

The query/key/value analogy is useful, but these are learned projections rather than literal linguistic questions and answers. A head has no guaranteed human-interpretable role.

**Required shape trace:**

~~~text
X_q             [B,T_q,D]
X_k             [B,T_k,D]
W_Q,W_K,W_V     [D,d_h] each, for this simple head
Q               [B,T_q,d_h]
K,V             [B,T_k,d_h]
Q K^T           [B,T_q,T_k]
A               [B,T_q,T_k]
A V             [B,T_q,d_h]
~~~

Derive the scaling motivation under explicit assumptions: if independent zero-mean query/key components have unit variance, the variance of their dot product grows with head width. Scaling moderates score magnitude before softmax.

Then calculate the backward pass. With upstream gradient G and without dropout:

~~~text
dV = A^T G
dA = G V^T
dS_row = A_row * (dA_row - sum(A_row * dA_row))
dQ = dS K / sqrt(d_h)
dK = dS^T Q / sqrt(d_h)
~~~

Continue through the projection matrices and input embeddings. Compare with automatic differentiation and finite differences. State which dimensions are summed.

For the projections, include:

~~~text
dW_Q = sum_batch(X_q^T dQ)
dW_K = sum_batch(X_k^T dK)
dW_V = sum_batch(X_k^T dV)
dX_q = dQ W_Q^T
dX_k = dK W_K^T + dV W_V^T
~~~

For self-attention, the shared input receives all three projection-path contributions. Keep a fixed blocking mask separate from a learned additive positional bias: the latter has trainable parameters and receives gradients.

Extend to h heads: project, reshape to [B,h,T,d_h], compute attention, concatenate outputs, and apply W_O. Explain that each head's projection can use every input feature; heads are not necessarily assigned disjoint semantic subsets of the original input.

**Build later:** an editable attention calculator plus a reference implementation using matrix operations.

**Pass when:** the learner can calculate one attention row, identify the softmax axis, and explain the different paths through which Q/K and V affect the output.

**Read:** the original architecture paper for the overall design; [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) for implementation shapes and conventions.

### Module 05 — Masking and information access

**Question:** What information is each prediction allowed to use?

Teach four different masks:

1. **Causal mask:** prevent a position from accessing future positions.
2. **Padding mask:** prevent artificial padded elements from contributing.
3. **Neighborhood mask:** restrict interactions to selected spatial/temporal candidates.
4. **Loss mask:** select outputs that contribute to the objective.

These masks may interact, but they are not interchangeable. A key-padding mask does not automatically zero padded query outputs or exclude padded targets from the loss.

For autoregressive training, show shifted input and labels:

~~~text
Model input:  BOS  cells  can  divide
Target:       cells can  divide EOS
~~~

The full target sequence is known during teacher-forced training, but each prediction must obey its permitted prefix. Generation feeds back newly selected tokens and remains sequential.

Mask forbidden logits before softmax. In a hand-written implementation, an entirely masked row can produce undefined numerical behavior; choose and verify an explicit policy rather than relying on accidental kernel behavior.

API conventions require special care: PyTorch scaled-dot-product attention interprets Boolean True as allowed, while MultiheadAttention Boolean masks use True for blocked positions. Verify the installed version's documentation. SDPA also requires an explicitly zero dropout probability for evaluation when it is called directly.

**Build later:** future-token perturbation, padding-invariance, and deliberately leaked-target examples.

**Pass when:** changing a future token leaves earlier causal outputs unchanged, padded values cannot influence valid predictions, and masks are independently inspected.

**Read:** [SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

### Module 06 — Residuals, normalization, FFNs, and the complete block

**Question:** How does attention fit into a trainable network?

Explain residual addition as an identity pathway alongside a learned transformation. Compare:

$$
\text{post-norm: } y=\operatorname{LN}(x+F(x))
$$

$$
\text{pre-norm: } y=x+F(\operatorname{LN}(x)).
$$

Choose a pre-norm educational implementation and document the final stack normalization. Separately reconstruct the original post-norm design. Normalization placement changes optimization; it is not a cosmetic rewrite.

LayerNorm computes statistics across selected feature dimensions for each token, with learned scale and offset and a numerical epsilon. Compare it with BatchNorm and RMSNorm. Explain what is normalized, not just the formula's name.

For a token with D features, calculate its feature mean and population variance, then normalize as (x - mean)/sqrt(variance + epsilon), followed by learned scale and offset. RMSNorm omits mean subtraction and normalizes through the root mean square; it is not merely LayerNorm with its bias removed.

The complete educational pre-norm block is:

~~~text
u = x + Dropout(MHA(LN1(x)))
y = u + Dropout(FFN(LN2(u)))
~~~

LN1 and LN2 have separate learned parameters. Explain dropout on attention weights separately from dropout on sublayer outputs.

The positionwise feed-forward network mixes feature channels at each position:

$$
\operatorname{FFN}(x)=\phi(xW_1+b_1)W_2+b_2.
$$

It changes the representation without directly mixing different token positions. Attention supplies token-to-token mixing.

Teach hidden expansion, activation, dropout, residual shape compatibility, and output projection. Compare ReLU and GELU, then introduce a gated variant:

$$
\operatorname{SwiGLU}(x)=
[\operatorname{SiLU}(xW_g)\odot(xW_u)]W_d.
$$

Match parameter budgets when comparing gated and ordinary FFNs; identical expansion ratios do not necessarily imply identical parameter counts.

**Build later:** attention sublayer, FFN sublayer, full block, and a small stack. Plot activation and gradient statistics as depth increases.

**Ablate:** remove residuals, change normalization placement, vary width, and compare activations. Change one factor at a time and report instability as an observation.

**Pass when:** the complete forward pass can be reconstructed, each parameter group has a role, and normalization axes can be checked directly.

**Read:** [Residual learning](https://arxiv.org/abs/1512.03385), [Layer Normalization](https://arxiv.org/abs/1607.06450), [normalization placement](https://arxiv.org/abs/2002.04745), [RMSNorm](https://arxiv.org/abs/1910.07467), [GELU](https://arxiv.org/abs/1606.08415), [GLU variants](https://arxiv.org/abs/2002.05202).

### Module 07 — Encoder, decoder, cross-attention, and language objectives

**Question:** Which architecture matches which prediction problem?

Keep topology and training objective separate:

| Family | Information access | Example learning objective |
|---|---|---|
| Encoder-only | Usually bidirectional within the permitted input | Predict masked tokens or sequence labels |
| Decoder-only | Causal access to a prefix | Predict the next token |
| Encoder–decoder | Encoder reads source; decoder reads target prefix and encoder outputs | Conditional sequence generation |

For cross-attention, Q originates in the receiving stream and K/V in a source stream. Their sequence lengths need not match.

Study BERT's masked-language modelling and original next-sentence objective, GPT-style autoregressive likelihood, and T5's text-to-text framing and span corruption. The original next-sentence objective is a historical design choice, not a requirement for every encoder.

Derive next-token negative log-likelihood:

$$
\mathcal L=-\frac{1}{N_{\text{valid}}}\sum_{t\in\text{valid}}\log p_\theta(y_t\mid y_{<t},x).
$$

Define vocabulary projection, logits, probability normalization, ignored targets, and perplexity. Compare perplexities only under compatible tokenization and evaluation conventions.

Include weight tying between input embeddings and the vocabulary projection as a design option, including compatible dimensions and the resulting shared gradients. For variable-length microbatches, accumulate summed losses and valid-token counts when the intended objective is a mean over all valid tokens; independently averaging each microbatch can change its weighting.

**Build later:** a tiny causal language model and a small encoder–decoder task. Implement masked-token learning as a separate objective experiment.

**Pass when:** pretraining, fine-tuning, conditioning, teacher forcing, and evaluation can be distinguished, and the source of every Q/K/V tensor is known.

**Read:** [BERT](https://arxiv.org/abs/1810.04805), [autoregressive language-model research](https://arxiv.org/abs/2005.14165), [T5](https://arxiv.org/abs/1910.10683), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

### Module 08 — Position, order, and geometry

**Question:** How can a content-based operation know where an element belongs?

Demonstrate that an unpositioned, unmasked self-attention layer is permutation equivariant: consistently permuting its inputs permutes its outputs. An order-dependent mask can itself break that symmetry, so state the conditions of the demonstration.

Compare:

- Learned absolute position vectors added to embeddings.
- Sinusoidal vectors with position-dependent frequencies.
- Relative position representations or biases.
- Rotary positional embeddings, which rotate Q/K feature pairs.
- Two-dimensional and three-dimensional position information.
- Time indices and elapsed time in irregularly sampled movies.

Derive a 2D rotation and verify the RoPE relative-offset identity numerically. Distinguish a representation being mathematically defined at a longer index from a trained model behaving reliably at that length.

For images, separate raster order from physical spatial relations. For cell detections, arbitrary ordering within a frame should not define biological meaning.

**Build later:** token shuffling, position removal, short-to-long generalization, and coordinate-scaling experiments.

**Pass when:** absolute position, relative displacement, sequence index, and physical coordinates can be explained separately.

**Read:** [Relative position representations](https://arxiv.org/abs/1803.02155), [RoFormer/RoPE](https://arxiv.org/abs/2104.09864).

### Module 09 — Optimization, inference, and computational efficiency

**Question:** What happens around the architecture during training and use?

Cover:

- Initialization, loss scaling, optimizer states, Adam and decoupled weight decay.
- Learning-rate schedules and warm-up as design choices.
- Gradient clipping, gradient accumulation, effective batch size, and reproducible shuffling.
- Training/validation/test separation, checkpoint selection, and early stopping.
- Mixed precision, gradient scaling where appropriate, and numerical tolerances.
- Checkpoint contents: weights, optimizer, scheduler, random state, configuration, and data provenance.

For inference, follow logits through temperature and greedy or sampled token selection. Explain stopping rules and the difference between generation quality and held-out loss.

Derive KV caching: prior keys and values at each layer can be reused during causal generation. Cached activations are not weight updates or durable learned memory.

For a conventional cache:

~~~text
bytes = 2 × layers × batch × cached_tokens
          × KV_heads × head_width × bytes_per_element
~~~

Introduce multi-query/grouped-query attention as extensions that alter KV-head count. Verify cached and uncached outputs with dropout disabled and matching position conventions.

Reuse is valid when the causal prefix representations remain unchanged under fixed parameters and deterministic evaluation. Changed prefix tokens, attention access, weights, or positional conventions can invalidate the cache.

Cached decoding often has unequal query/key lengths. A query at absolute position offset + i may attend only to keys at positions at most offset + i. A fresh square triangular mask is insufficient, and a library's non-square causal alignment must be checked explicitly. Verify one new query attending to the entire cached prefix and a multi-token continuation. RoPE positions must continue from the existing offset.

For dense self-attention, score storage grows with B × h × N² and pairwise work with approximately N²D. Projection and FFN work also matter; do not characterize the whole model using only its score matrix.

FlashAttention changes memory movement and tiling for exact dense attention, within numerical precision. It is not a claim of linear pairwise arithmetic. Compare efficient kernels, local/sparse interaction patterns, and approximations as distinct ideas.

**Build later:** reference-versus-library attention checks, sequence-length memory plots, and prefill-versus-decoding profiles.

**Pass when:** the learner can explain a memory bottleneck, identify a caching error, and distinguish arithmetic complexity from wall-clock speed.

**Read:** [Adam](https://arxiv.org/abs/1412.6980), [AdamW](https://arxiv.org/abs/1711.05101), [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150), [Grouped-query attention](https://arxiv.org/abs/2305.13245), [FlashAttention](https://arxiv.org/abs/2205.14135).

## 9. Modules 10–17: from language sequences to vision and microscopy

### Module 10 — Patchification and a Vision Transformer

**Question:** How can the same attention block process an image?

Contrast convolution's local, shared filtering with attention's content-dependent token interactions. A CNN can acquire a large receptive field through depth; a transformer is not automatically superior because its interactions can span an image.

Patchify [B,C,H,W] into N patches of P × P pixels. Flatten each patch and project its CP² values into D features:

$$
N=\frac{H}{P}\frac{W}{P},\qquad
X_{\text{patch}}\in\mathbb R^{B\times N\times CP^2},\qquad
E=X_{\text{patch}}W_{\text{patch}}.
$$

Use this concrete trace:

~~~text
Input image:           [B,3,224,224]
Patch size:            16 × 16
Patch grid:            14 × 14
Flattened patches:     [B,196,768]
Projected patch tokens:[B,196,D]
With a class token:    [B,197,D]
Encoder output:        [B,197,D]
Classification head:  [B,number_of_classes]
~~~

The patch vector length happens to be 768 in this example. The embedding width D is a separate design choice.

The formulas assume divisibility by the patch dimensions. For other image sizes, choose and document padding, cropping, or a compatible embedding policy; keep reconstruction coordinates and valid-region masks consistent.

Explain the learned classification token, positional embeddings, final normalization, classifier, and mean-pooling alternative. A patch is a spatial sampling unit, not necessarily one semantic object.

Implement patchification explicitly first. Show its equivalence to a suitably configured convolutional projection with kernel and stride P. Reconstruct the input exactly before the lossy/learned projection to verify patch order.

For dense tasks, retain spatial patch features. A backbone may keep its class token internally while a decoder uses the patch tokens. One pooled image vector is not by itself a dense prediction map.

**Build later:** a tiny ViT and patch inspector on a small classification dataset.

**Ablate:** patch size, position information, class-token versus mean pooling, and data quantity. Halving patch width in 2D quadruples token count and increases a full score matrix sixteenfold.

**Pass when:** every reshape is understood and the difference between classification and dense prediction is explicit.

**Read:** [ViT, 2020 preprint / ICLR 2021](https://arxiv.org/abs/2010.11929).

### Module 11 — Hierarchy, local attention, and pretraining

**Question:** How can vision transformers become more data-efficient or computationally practical?

Study several interventions without conflating them:

| Approach | What changes | Question for this project |
|---|---|---|
| DeiT | Training recipe and distillation, including a distillation token | How much of a result comes from supervision and training rather than backbone topology? |
| Swin | Local windows, shifted windows, patch merging, hierarchical features | How do cross-window communication and multiscale outputs arise? |
| DINO family | Teacher–student self-supervised representation learning | What features transfer to microscopy, and what domain gaps remain? |
| MAE | Masked-patch reconstruction with an asymmetric encoder–decoder | What is learned by reconstruction, and what still needs downstream supervision? |

For Swin, draw window partitions before and after shifting. Explain masking at shifted boundaries, patch merging, resolution changes, and channel changes. Approximately linear attention scaling assumes fixed window size; the rest of the model still has costs.

For DINO, teach multiple views, student/teacher outputs, moving-average teacher updates, and safeguards against collapsed representations. Distinguish the original objective from later DINOv2/DINOv3 versions.

For MAE, distinguish visible encoder tokens from decoder reconstruction positions. A reconstructed patch image is not a cell instance mask.

Treat model scale, dataset scale, augmentation, objectives, and initialization as separate experimental factors.

**Build later:** a small window-attention exercise; then compare frozen features plus a small head, partial fine-tuning, and a randomly initialized small model.

**Pass when:** the learner can identify which approaches change architecture, supervision, computation, or initialization.

**Read:** [DeiT](https://arxiv.org/abs/2012.12877), [Swin](https://arxiv.org/abs/2103.14030), [DINO](https://arxiv.org/abs/2104.14294), [DINOv2](https://arxiv.org/abs/2304.07193), [DINOv3](https://arxiv.org/abs/2508.10104), [MAE](https://arxiv.org/abs/2111.06377).

### Module 12 — Cell instance segmentation before transformer comparison

**Question:** What must a segmentation representation preserve?

Use synthetic touching disks/ellipses with exact instance labels, then inspect a small real microscopy dataset. Keep the labelled structure explicit: nuclei, whole cells, or organoids.

Build these baselines in order:

1. Thresholding and connected components.
2. Distance-transform markers and watershed.
3. Small U-Net predicting foreground and a separating representation.
4. One established pretrained cell-instance baseline, if compatible with the data.

Explain threshold selection, noise, background variation, connectedness, distance to background, marker placement, and watershed flooding. One connected foreground component can contain multiple cells. More markers do not guarantee more accurate instances.

For learned models, derive foreground, boundary, centre, and per-instance distance targets from the annotated instance masks. Be explicit about whether a distance is measured within each reference instance or within the union of foreground.

Distinguish:

- Pixel classification losses, such as cross-entropy.
- Region-overlap losses, such as Dice loss.
- Distance/flow regression losses.
- The instance-level evaluation applied after reconstruction.

Within-frame instance IDs and persistent track IDs are different objects. Arbitrary instance numbers cannot be treated as scalar regression targets: an equivalent relabelling of the masks must not change their segmentation meaning.

Compare StarDist's radial shape proposals and suppression with Cellpose's within-object flow reconstruction. These flows are not physical motion across frames. Include shape assumptions and difficult branched/overlapping cases.

**Build later:** a shared instance-reconstruction interface and a labelled error gallery.

**Ablate:** identical foreground with good, missing, and excessive markers. Separately vary image features and reconstruction rules.

**Pass when:** foreground overlap, instance count, boundary accuracy, merges, and splits can be measured separately.

**Read:** [watershed tutorial](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html), [U-Net](https://arxiv.org/abs/1505.04597), [StarDist](https://arxiv.org/abs/1806.03535), [Cellpose](https://doi.org/10.1038/s41592-020-01018-x), [Omnipose](https://doi.org/10.1038/s41592-022-01639-4).

### Module 13 — Dense prediction from transformer features

**Question:** How do patch features become a decision at each pixel?

Start from an encoder feature tensor [B,N,D]. Remove or separately handle non-spatial tokens, restore the patch grid, and apply a decoder. Discuss linear heads, upsampling, convolutional refinement, and multiscale fusion.

Explain what bilinear interpolation can and cannot recover. Upsampling a coarse map does not independently restore missing boundary evidence.

Study SegFormer as a hierarchical transformer encoder with a lightweight multiscale MLP decoder. Its semantic output does not automatically assign individual identities to touching cells.

Keep three components separate:

~~~text
backbone → prediction heads → instance reconstruction
~~~

**Build later:** a small patch encoder with a dense head. Compare it with U-Net using the same foreground/boundary targets, label budget, reconstruction, and evaluation.

**Required controls:** report pretraining, trainable parameter count, crop size, augmentation, runtime, and memory. A pretrained transformer versus a randomly initialized CNN is a useful practical comparison, but it does not isolate architectural effects.

**Pass when:** the learner can identify where spatial resolution changes and which operation creates separate object instances.

**Read:** [SegFormer](https://arxiv.org/abs/2105.15203).

### Module 14 — DETR: object queries and permutation-invariant set prediction

**Question:** How can a network predict an unordered, variable-sized collection of objects?

Explain a learned object query as a slot that gathers information from image features through cross-attention. Each slot predicts an object class and location; extra slots can predict "no object."

Training needs to match an unordered predicted set to an unordered reference set. Solve a tiny bipartite assignment with three predictions and two reference objects. Then calculate classification and localization losses on matched predictions and no-object supervision on unmatched slots.

Separate **matching cost** from the final **training loss**. Explain box-coordinate loss, generalized IoU, and why matching is performed before differentiable losses are evaluated on selected correspondences.

Important distinctions:

- DETR's training assignment aligns predictions with reference objects in one image.
- Tracking assignment associates observations across time.
- A fixed query index is not automatically a persistent biological identity.
- Query capacity limits how many objects can be represented independently.
- The original DETR formulation avoids a conventional NMS inference stage, but this is a property of that design rather than every transformer detector.

Read Deformable DETR to understand sparse sampling around reference locations and multiscale features. Explain why it changes attention sampling rather than merely lowering image resolution.

**Build later:** a tiny set-prediction model on synthetic shapes, with duplicate predictions, missing objects, and insufficient-query cases.

**Pass when:** permutation-invariant matching, no-object slots, and query capacity can be demonstrated numerically.

**Read:** [DETR](https://arxiv.org/abs/2005.12872), [Deformable DETR](https://arxiv.org/abs/2010.04159).

### Module 15 — Mask queries, prompts, and cell-specific foundation models

**Question:** How can a query specify an object mask instead of only a box?

Study mask classification: a query predicts a class and a mask over spatial features. Explain semantic, instance, and panoptic outputs. Study Mask2Former's restricted cross-attention to predicted mask regions and multiscale feature processing.

Connect mask set training to DETR: matching includes class and mask costs; matched masks receive class and mask supervision, often including binary cross-entropy and Dice terms; unmatched queries receive no-object supervision. Point sampling for matching/loss computation and mask-restricted cross-attention are separate mechanisms.

Then study promptable segmentation through SAM's image encoder, prompt encoder, and mask decoder. Trace point and box prompts, image features, mask candidates, and quality predictions. Automatic mask generation adds a proposal/filtering procedure around the promptable model.

Keep these similarly named methods distinct:

| Method | What to understand |
|---|---|
| SAM | General prompt-conditioned image segmentation |
| SAM 2 | Image/video segmentation with temporal memory; not inherently a cell-lineage model |
| CellSAM | CellFinder-generated prompts combined with SAM-based mask prediction |
| Cellpose-SAM | A SAM transformer backbone adapted to Cellpose's prediction and mask-reconstruction framework |
| DINO-based Cellpose variants | A different pretrained backbone family within the Cellpose software ecosystem |

CellSAM was published in Nature Methods in December 2025. The checked Cellpose-SAM source is a 2025 bioRxiv preprint and released software. Cellpose documentation also lists later 2026 checkpoints; record the exact model rather than treating the family as one condition.

**Build later:** pretrained inference first. Compare reference-derived boxes, perturbed boxes, and automatically detected boxes. Reference-derived prompts are an oracle-prompt condition, not an automatic pipeline.

**Ablate:** prompt quality, backbone adaptation, annotation budget, and threshold/postprocessing choices.

**Pass when:** prompt errors, detection errors, and mask errors can be separated, and a video segmentation result is not mistaken for a division-aware lineage.

**Read:** [Mask2Former](https://arxiv.org/abs/2112.01527), [SAM](https://arxiv.org/abs/2304.02643), [SAM 2](https://arxiv.org/abs/2408.00714), [CellSAM](https://doi.org/10.1038/s41592-025-02879-w), [Cellpose-SAM](https://doi.org/10.1101/2025.04.28.651001), [current Cellpose model inventory](https://cellpose.readthedocs.io/en/latest/models.html).

### Module 16 — Volumetric patches and anisotropic microscopy

**Question:** Which assumptions change when the input is a volume?

Use [B,C,Z,H,W] and non-cubic patches if appropriate:

$$
N=(Z/P_z)(H/P_y)(W/P_x).
$$

Relate sampling to physical scale. For spacing [4,1,1] micrometres in [z,y,x], a [4,16,16] voxel patch has nominal support spanning approximately [16,16,16] micrometres. Equal voxel counts along axes need not describe equal physical extents.

Choose a crop/pad policy when volume dimensions do not divide by patch sizes. Reconcile overlapping tile predictions through geometry and instance matching/stitching; averaging arbitrary integer instance IDs has no meaningful interpretation.

Explain:

- Sampling spacing versus optical resolution.
- Depth-dependent signal and anisotropic blur.
- Slice-wise 2D, neighboring-slice input, and true 3D processing.
- 3D positional encodings and coordinate normalization.
- Crop boundaries, overlap, sliding-window inference, and instance stitching.
- Why maximum-intensity projections can collapse distinct objects.
- Why isotropic resampling does not recreate unobserved detail.

Study UNETR and Swin UNETR as medical dense-prediction examples. Their semantic segmentation designs require suitable outputs and reconstruction to become cell-instance pipelines.

**Build later:** synthetic ellipsoid volumes, then a small real crop. Compare depth continuity and memory before increasing volume size.

**Token-count exercise:** a 128³ volume with 16³ patches has 512 tokens; 8³ patches have 4,096. The ordinary full score matrix grows sixty-fourfold when token count grows eightfold.

**Pass when:** the physical meaning of a patch, a centroid displacement, and a 3D overlap score is explicit.

**Read:** [UNETR](https://arxiv.org/abs/2103.10504), [Swin UNETR](https://arxiv.org/abs/2201.01266), [CTC 3D datasets and metadata](https://celltrackingchallenge.net/3d-datasets/).

### Module 17 — Video, temporal attention, and information availability

**Question:** Should time be represented through image patches, objects, or persistent queries?

Compare three representations:

1. **Patch-level video tokens:** retain image evidence, including ambiguous boundaries; token counts can be large.
2. **Detected-object tokens:** compact representations of individual objects; upstream detection errors matter.
3. **Persistent track queries:** learned states associated with ongoing hypotheses; identity handling depends on the full architecture.

Introduce tubelets, frame-wise features, joint space–time attention, divided spatial/temporal attention, and local spatiotemporal windows. For F frames and N spatial patches per frame, full space–time attention considers approximately (FN)² token pairs.

TimeSformer and Video Swin are useful architectural readings, but their original video-recognition outputs do not themselves supply cell instances or lineages.

Separate online and offline settings:

- An online decision uses only data available up to its decision time.
- An offline window may use future frames to resolve ambiguity.
- A causal mask is a choice determined by the task's information constraints, not a universal requirement for microscopy.

Discuss time interval, irregular sampling, frame-rate changes, and motion ambiguity. A frame index is not the same as elapsed biological time.

**Build later:** a small synthetic movie inspector and attention-cost calculator. Full video-model training is optional.

**Pass when:** the learner can draw each representation's information flow and state whether future observations are available.

**Read:** [TimeSformer](https://arxiv.org/abs/2102.05095), [Video Swin](https://arxiv.org/abs/2106.13230).

## 10. Modules 18–24: association, lineages, evaluation, and research

### Module 18 — Tracking before transformer association

**Question:** What can be solved with explicit geometry and matching?

Freeze segmentation to isolate tracking. Begin with exact synthetic detections, then introduce controlled missing, false, merged, or split detections.

Build:

- Nearest-neighbor linking.
- One-to-one linear assignment using a shared cost matrix.
- Optional motion prediction with a clearly stated dynamical model.
- Explicit appearance/disappearance handling.
- A separate division-aware extension.

An illustrative cost can combine normalized displacement, mask overlap, and appearance differences. State how each term is scaled and which observations are eligible candidates.

Use this assignment example:

| Previous object | Next detection 1 | Next detection 2 |
|---|---:|---:|
| A | 1.0 | 2.0 |
| B | 1.1 | 10.0 |

A greedy A-first choice gives total cost 11. The minimum one-to-one assignment gives 3.1. Explain why jointly optimizing one frame pair is still different from globally optimizing a whole video.

Normal one-to-one assignment does not directly model one mother producing two daughters. Add explicit events rather than hiding division in a nearest-neighbor heuristic.

**Build later:** crossing, division, entry/exit, and gap examples with saved cost matrices.

**Pass when:** a wrong correspondence can be diagnosed from the candidates, score, and decoder independently.

**Read:** [linear assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html), [Conservation Tracking](https://www.cv-foundation.org/openaccess/content_iccv_2013/html/Schiegg_Conservation_Tracking_2013_ICCV_paper.html).

### Module 19 — Turn detections into tokens

**Question:** What is a token in a cell-tracking transformer?

Construct each object description from selected information:

~~~text
time or elapsed time
centroid in physical coordinates
area or volume
intensity summaries
shape descriptors
optional appearance features from an image encoder
~~~

Normalize using training data only. Keep intensity preprocessing and spatial scaling documented. Feature engineering can leak information if it uses future frames or labels unavailable in the intended setting.

Batch variable object counts using padding and valid-token masks. Keep attention masks, eligible-pair masks, and supervision masks separate.

Study absolute versus relative coordinates, temporal windows, temporal offsets, neighborhood gating, and optional crop embeddings.

**Build later:** a detection-token dataset with reference correspondence indices and complete shape traces.

**Ablate:** coordinates only; coordinates plus shape; coordinates plus image features; arbitrary token-index positions versus physical position information.

**Pass when:** reordering detections within a frame, while preserving all mappings, does not invent biological changes; padding cannot change valid outputs.

### Module 20 — Learn contextual associations

**Question:** Can attention resolve ambiguities that independent pairwise features cannot?

Create a small educational transformer that scores possible cross-frame associations. Compare against an independent pairwise MLP using identical detections, splits, and decoding.

Before selecting a loss, define the supervision:

- Which pairs are positives?
- Are divisions allowed to create multiple children?
- Can a child have multiple parents under the chosen model?
- Are missing observations represented?
- Is the objective independent binary classification, parent selection, or another structured normalization?
- How are negative pairs sampled and weighted?

Study Trackastra's original paper as a concrete example. It uses detection features and spatial/temporal context to learn associations, with a normalization designed to accommodate divisions. Its learned scores and subsequent linking are distinct parts of the system.

Its original within-window supervision includes ancestor–descendant relationships, while published linking uses adjacent-frame candidate edges. Gap handling in this learning project is a separate extension. Score normalization and final graph-validity constraints have different roles.

Attention weights inside the network are not automatically the final association probabilities. A correspondence head and its supervision must be inspected separately.

**Build later:** tiny clean overfitting, held-out synthetic movies, then a small real sequence. Label the model an educational adaptation unless the original architecture, training, data, and evaluation are faithfully reproduced.

**Ablate:** context length, positional information, appearance features, and detection noise. Keep the decoder fixed when testing the association model.

**Pass when:** the learner can explain which context helps, show a counterexample, and trace an error to a specific pipeline stage.

**Read:** [Trackastra, ECCV 2024](https://arxiv.org/html/2405.15700v2), [official implementation](https://github.com/weigertlab/trackastra). Pin the version/checkpoint because later implementations include additional variants.

### Module 21 — Decode scores into valid lineages

**Question:** Why are high association scores insufficient?

Define a graph whose nodes are object detections and whose edges represent time-forward relationships. Add candidate continuation, division, appearance, disappearance, and optional gap events.

For an ordinary non-fusing lineage model, specify the constraints:

- A selected detection has at most one parent.
- A continuing cell has one successor.
- An allowed division has the required daughter structure.
- A track can start or terminate under explicit rules.
- Incompatible selections cannot coexist.

Tie event choices to detection selection: every selected detection needs an incoming continuation/gap or an appearance event, and an outgoing termination, continuation/gap, or permitted division. In the initial binary-division model, a division has two daughters; this is a declared assumption rather than a universal biological limit.

Use binary variables for selected candidates and minimize a sum of event costs subject to consistency constraints. Explain the difference between a linear assignment, a flow formulation, and an ILP with richer event/hypothesis constraints.

Reuse the same learned scores across constrained greedy, continuation-only assignment, and event-aware global decoding. This isolates the decoder's effect.

Introduce probability-to-cost transformations carefully. A model score is not automatically a calibrated probability; event penalties and normalization choices affect the solution.

Negative-log event costs require a stated probabilistic interpretation or factorization. Independently calibrated pair scores do not automatically define a coherent joint distribution over lineages.

**Build later:** an event-aware synthetic lineage solver and graph-invariant checks.

**Required counterexamples:** two possible parents for one child; two children for one parent where evidence must distinguish division from a false split; one missing frame; an excluded correct candidate; a segmentation split imitating division.

Internal skip edges need an explicit export policy. Convert gaps and track segments to the chosen evaluator's file/graph conventions; do not assume any internal representation can be submitted unchanged.

**Pass when:** optimal objective value is distinguished from biological correctness and a missing candidate cannot be "recovered" by a better optimizer.

**Read:** [CBIA: ILP and probability scores](https://www.muni.cz/en/research/publications/2385779), [graph-based tracking evaluation](https://doi.org/10.1371/journal.pone.0144959).

### Module 22 — Joint segmentation hypotheses and persistent track queries

**Question:** Must masks be finalized before temporal reasoning?

Explore two directions separately.

**Joint hypothesis selection:** supply alternative interpretations of an ambiguous region, including merged and separated candidates. Ultrack selects compatible segmentation hypotheses and associations through optimization. It is an optimization framework, not itself a transformer backbone.

**Joint learned segmentation/tracking:** study Cell-TRACTR's image features and persistent track queries. Contrast identity represented in ongoing queries with associations predicted between already segmented detections. Its published experiments concern 2D bacterial and mammalian imaging; broader 3D applicability must be tested.

**Build later:** first a toy candidate-selection problem; optionally inspect a published joint model's outputs. Full training of the joint architecture is an advanced extension.

**Ablate:** candidate coverage, association scores, and graph constraints independently. Temporal evidence cannot recover an interpretation absent from the candidate set unless the system can generate new candidates.

**Pass when:** the learner can identify which errors require better images, better candidates, better association, or a better decoder.

**Read:** [Ultrack](https://doi.org/10.1038/s41592-025-02778-0), [Cell-TRACTR](https://doi.org/10.1371/journal.pcbi.1013071).

### Module 23 — Evaluation, uncertainty, and biological relevance

**Question:** What would make the analysis trustworthy for a collaborator?

Use the official CTC implementations/conventions for named challenge metrics rather than silently replacing them with simpler approximations.

| Aspect | What to assess |
|---|---|
| Foreground geometry | Pixel/voxel overlap and relevant boundaries |
| Individual instances | Correct detections, merges, splits, misses, and false objects |
| SEG | Reference-instance mask agreement based on Jaccard/IoU and CTC matching |
| DET | Normalized object-detection graph-edit cost |
| TRA | Normalized graph-edit cost including detection and temporal relationships |
| Complete tracks / track fractions | Entire recovered tracks; the longest continuously matching fragment relative to each reference track |
| Branching correctness | Correct division events within a defined tolerance |
| Division precision/recall | Correct predicted/reference divisions using a declared matching rule and temporal tolerance |
| Cell-cycle accuracy | Correct reconstruction of cycle lengths |
| Practical use | Runtime, memory, manual correction, and usable outputs |

TRA follows:

$$
\operatorname{TRA}=1-
\frac{\min(\operatorname{AOGM},\operatorname{AOGM}_0)}
{\operatorname{AOGM}_0},
$$

where the denominator is the cost of constructing the reference graph from scratch. Handle degenerate cases according to the official evaluator.

Build controlled examples with correct masks but swapped identities, merged objects, missing observations, false divisions, and shifted division times. Determine which metrics respond and which hide the failure.

Cell-HOTA from Cell-TRACTR is an optional complementary diagnostic for detection, association, and division. Document its conventions rather than treating it as a replacement for an official CTC score.

Discuss gold human-derived annotations, silver algorithm-derived annotations, expert disagreement, and synthetic reference truth. Fusion is useful supervision but does not guarantee independent truth.

For uncertainty, distinguish uncertain image boundaries, ambiguous associations, model uncertainty, and domain shift. Attention entropy is not automatically calibrated error probability.

**Build later:** reliability diagrams where probabilities are meaningful, selective-review curves, error galleries, and independent curated evaluation subsets.

**Pass when:** changes in metrics can be connected to migration, morphology, lineage, or division-time measurements.

**Read:** [CTC evaluation](https://celltrackingchallenge.net/evaluation-methodology/), [CTC annotations](https://celltrackingchallenge.net/annotations/), [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z), [metric pitfalls](https://doi.org/10.1038/s41592-023-02150-0).

### Module 24 — Capstone: one question, a complete chain of evidence

**Question:** Which bounded claim can this project actually support?

Recommended initial question:

> Does contextual attention improve cell association over geometric and independent-pair baselines when detections become ambiguous or imperfect?

Begin with a small 2D dataset. Keep two experiments separate:

1. Reference-derived detections to isolate association.
2. Predicted segmentation to evaluate the complete pipeline.

Use a fixed decoder to compare association models. Then freeze association scores to study decoder changes. Only later study joint segmentation selection or 3D.

The baseline set should include nearest-neighbor linking, linear assignment, a pairwise MLP, and one small contextual transformer. Compare a published pretrained method as a separate practical condition when compatible.

Run at least three independent training seeds for the main learned comparison when resources permit a meaningful robustness study. Report the actual number; fewer runs require a narrower claim. Use paired held-out sequences and avoid treating every cell as an independent biological replicate.

The capstone may conclude that no benefit was demonstrated under the tested conditions. The required outcome is a defensible experiment and a clear understanding of its limits.

**Pass when:** configurations, split provenance, numerical checks, representative failures, runtime/memory measurements, and a limited conclusion are all available.

## 11. A shared explanation across NLP, vision, and cells

Use this table repeatedly. It prevents the project from becoming disconnected tutorials.

| Component | Language example | Image example | Cell-tracking example |
|---|---|---|---|
| Input unit | Subword or byte | Spatial patch | Detected cell at a time |
| Input features | Embedding lookup | Flattened/projected patch | Geometry, intensity, shape, appearance |
| Position | Token index | 2D or 3D coordinates | Physical location and time |
| Self-attention | Contextual interaction among words | Interaction among patches | Context among nearby observations |
| Cross-attention | Target prefix reads source text | Object queries read image features | Track/detection queries read another observation stream |
| Mask | Causal/padding/document constraints | Valid patches/windows/regions | Padding, time access, spatial candidates |
| Output head | Vocabulary or label logits | Classes, boxes, distances, or masks | Association or event scores |
| Loss | Token likelihood or masked-token prediction | Pixel, mask, box, or set losses | Pairwise/parent/event supervision |
| Decoding | Greedy or sampled tokens | Reconstructed masks or object sets | Consistent tracks and lineages |
| Main failure | Leakage, poor generalization, wrong token prediction | Merged/split objects, coarse boundaries | Identity switches, false divisions, gaps |

Explicitly compare the overloaded word "decoder": an autoregressive language decoder, a dense upsampling decoder, a query-based mask decoder, and a graph decoder are not interchangeable structures.

## 12. Detailed component coverage checklist

Use this as a coverage audit when expanding the lessons. A checked item means an explanation and demonstrable example exist, not merely that its name appears in the README.

### Inputs and representations

- [ ] Vocabulary, special tokens, padding, BPE, bytes, and normalization.
- [ ] Token lookup versus contextual representations.
- [ ] Patch order, channel order, patch projection, and unpatchification.
- [ ] Class/distillation/object/mask/track tokens and their different roles.
- [ ] Variable object counts, valid masks, spatial units, and time intervals.
- [ ] Training-only feature normalization and preprocessing provenance.

### Attention internals

- [ ] Q/K/V sources, widths, learned projections, optional biases, and fused QKV implementation.
- [ ] Dot-product compatibility, scaling assumptions, and stable softmax.
- [ ] Query axis versus key axis and row normalization.
- [ ] Attention weights versus value content versus final prediction scores.
- [ ] Multiple heads, concatenation, and output projection.
- [ ] Self-attention versus cross-attention.
- [ ] Fixed masks versus learned relative bias.
- [ ] Padding, causal, neighborhood, and supervision masks.
- [ ] Fully masked rows, mixed precision, and numerical tolerances.
- [ ] Gradients through softmax, projections, and residual paths.

### Block and stack

- [ ] Pre-norm, post-norm, final norm, and separate norm parameters.
- [ ] LayerNorm axes, population variance, epsilon, scale, and offset.
- [ ] RMSNorm versus LayerNorm versus BatchNorm.
- [ ] FFN expansion/contraction and channel versus token mixing.
- [ ] ReLU, GELU, SiLU, and gated FFNs.
- [ ] Attention dropout, output dropout, and optional stochastic depth.
- [ ] Initialization, parameter count, and activations across depth.
- [ ] Input/output weight tying and shared parameter gradients.
- [ ] Absolute, sinusoidal, relative, rotary, and multidimensional positions.

### Objectives and execution

- [ ] Shifted targets, teacher forcing, ignored labels, and loss normalization.
- [ ] Masked-token, causal-token, span-corruption, contrastive, and image-reconstruction objectives.
- [ ] Encoder-only, decoder-only, and encoder–decoder information paths.
- [ ] Autoregressive sampling, temperature, stopping, and cache updates.
- [ ] Prefill versus incremental decoding and cache-position alignment.
- [ ] Multi-query/grouped-query attention and cache storage.
- [ ] Reference attention versus optimized exact attention versus sparse/approximate attention.
- [ ] Optimizer states, weight decay, warm-up, clipping, accumulation, and checkpoint restoration.

### Vision and microscopy outputs

- [ ] Foreground, boundaries, centres, distances, flows, boxes, and masks.
- [ ] Watershed markers, StarDist proposals, NMS, and flow integration.
- [ ] Dense upsampling, feature pyramids, and coarse-resolution limitations.
- [ ] Set matching, no-object classes, duplicate predictions, and query capacity.
- [ ] Prompt encoding, oracle prompts, and automatic proposal quality.
- [ ] 3D anisotropy, sliding windows, and instance stitching.
- [ ] Temporal windows, object tokens, persistent queries, and offline/online information.
- [ ] Continuation, division, appearance, disappearance, and gap events.
- [ ] Candidate graphs, optimization constraints, and graph validity.
- [ ] Instance, association, lineage, biological, and practical evaluation.

## 13. Research history beyond the central milestones

Teaching should connect each paper to the question it changes. Keep a distinction between first preprint date and conference/journal publication year.

| Period | Development to examine | Role in this curriculum |
|---|---|---|
| 1997 | LSTM | Controlled recurrent state and long-range optimization motivation |
| 2014–2015 | Seq2seq and recurrent attention | Conditional generation and selective source access |
| 2017 | Transformer | Attention-centered encoder–decoder architecture |
| 2017–2018 | Non-local networks and Image Transformer | Earlier use of attention-like interactions in vision |
| 2018–2020 | BERT, autoregressive scaling, T5 | Different attention access patterns and pretraining objectives |
| 2020–2021 | DETR and ViT | Object-set prediction and patch-based recognition |
| 2021 | DeiT, Swin, SegFormer, DINO | Training efficiency, hierarchy, dense outputs, self-supervision |
| 2021–2022 | MAE, Mask2Former, volumetric transformer segmentation | Reconstruction objectives, mask queries, 3D features |
| 2022 onward | Efficient exact attention | Memory movement and usable sequence lengths |
| 2023–2024 | SAM, later vision pretraining, video segmentation memory | Prompts, transferable representations, temporal state |
| 2024 | Trackastra and CBIA's ILP work | Learned cell associations and structured event decoding |
| 2025–2026 | Cell-TRACTR, Ultrack, CellSAM, Cellpose-SAM and later checkpoints | Joint tasks, candidate selection, cell-specific adaptation |

ViT was not the first use of attention in images. Read [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) and [Image Transformer](https://arxiv.org/abs/1802.05751) to understand that earlier line of work without turning every historical branch into a full implementation requirement.

## 14. Research paper study

For each paper below, prepare a one-page card using the template in section 18.

| Paper | Why it belongs | Question for paper discussion |
|---|---|---|
| [The Cell Tracking Challenge: 10 years of objective benchmarking, 2023](https://www.nature.com/articles/s41592-023-01879-y) | Field structure, generalization, annotations, and technical/biological evaluation | Which errors most limit useful complete tracks? |
| [DeepFuse, 2025](https://www.muni.cz/en/research/publications/2493619) | Full-resolution fusion/refinement of computer-generated instance annotations | How are correlated rater errors and independent evaluation handled? |
| [Gravitational cell detection and tracking, 2024](https://arxiv.org/abs/2312.03509) | An interpretable classical alternative with explicit imaging assumptions | Which failure cases would most benefit from learned features? |
| [ILP and probability scores, 2024](https://www.muni.cz/en/research/publications/2385779) | Event-cost interpretation in global cell tracking | How sensitive are selected lineages to score calibration and event costs? |
| [Attention networks with numerical integration, 2026](https://www.muni.cz/en/research/publications/2550668) | Piecewise boundary tracing with attention and numerical integration | How does contour tracing recover from local mistakes? |
| [Shape-Aware Cell Segmentation with Global Cues, 2026](https://www.muni.cz/en/research/publications/2584461) | Learned features combined with explicit shape/global structure | Which shape information improves boundaries beyond the backbone alone? |
| [Mammary epithelial organoid segmentation and tracking, 2023](https://cbia.fi.muni.cz/research/spatiotemporal/organoids.html) | Synthetic training data and a practical 2D brightfield workflow | How was synthetic-to-real transfer validated? |

Treat historical benchmark comparisons as results of their studied submissions, not a permanent ranking. Where only an institutional abstract is available, record that access level and do not invent architecture, loss, score, or ablation details.

Use these readings to discuss microscopy methods, annotation quality, assumptions, and evaluation. Their inclusion is an educational selection, rather than a claim that the methods share one task or establish a permanent benchmark ranking.

## 15. Data ladder and provenance

### Level A — Exact synthetic examples

Generate text tasks, colored shapes, touching disks, small ellipsoid volumes, and moving/dividing objects. Synthetic data provides exact geometric and lineage references and lets the project isolate one failure mechanism.

Vary noise, blur, crowding, shape, intensity, motion, division timing, and missed detections independently. Do not treat success on these simplified worlds as evidence of real microscopy generalization.

### Level B — Small public image examples

Use a small classification dataset for learning patchification and ViT training. Use a compact annotated microscopy subset for mask reconstruction. Check whether references mark nuclei, whole cells, or a larger structure.

Preserve masks with label-safe nearest-neighbor transformations. Do not interpolate integer instance IDs as continuous values. Apply identical geometric transformations to the image and its annotation, and document how temporal augmentation preserves correspondence.

### Level C — Real time-lapse microscopy

Select a compatible [CTC 2D dataset](https://celltrackingchallenge.net/2d-datasets/) before attempting [3D data](https://celltrackingchallenge.net/3d-datasets/). Nuclear fluorescence is a useful initial setting, but choose based on annotation availability and the intended task.

Inspect annotation coverage: segmentation masks may be sparse even when tracking markers have broader coverage. Unannotated cells must not automatically become background targets.

Public training data and hidden benchmark-test references serve different purposes. Locally score held-out annotated material; use official benchmark submission procedures only if a later task actually requests participation.

A dataset with only a few independent sequences supports limited generalization claims. Do not manufacture statistical independence by splitting adjacent frames or reporting thousands of cells as independent experiments.

### Manifest fields

Record:

- Dataset/version and source URL.
- File identifiers and checksums where practical.
- Object type, marker, microscopy modality, and channels.
- Pixel/voxel spacing and frame interval.
- Annotation origin, completeness, and any curation.
- Training/validation/test membership at the independent-sequence/specimen level.
- Preprocessing, crop selection, and label remapping.
- Documented pretrained-model training overlap; mark unknown overlap as unknown.

Record relevant dataset and checkpoint use terms when selecting resources for the eventual repository.

## 16. Experiment sequence and ablation matrix

These are planned experiments. Their outcomes must remain blank until measured.

| ID | Experiment | Controlled comparison | Evidence to save |
|---|---|---|---|
| E01 | Attention by hand | Manual calculations versus tensor implementation | Matrices, gradients, tolerances |
| E02 | Causality | Future token changed with/without correct mask | Earlier outputs and leakage explanation |
| E03 | Padding | Same valid sequence with different padded content | Valid predictions and loss |
| E04 | Block stability | One residual/norm choice changed | Activations, gradients, training behavior |
| E05 | Position | Content shuffled with/without position consistency | Predictions and symmetry explanation |
| E06 | Cached inference | Full-prefix versus incremental/chunked computation | Logit differences and position offsets |
| E07 | Patch size | Fixed image with different patch scales | Token count, memory, image reconstruction |
| E08 | Touching instances | Shared foreground with changed markers/boundaries | Merges/splits and overlap |
| E09 | Backbone | CNN versus small transformer with shared heads | Initialization, compute, instance metrics |
| E10 | Set prediction | Query capacity and duplicate-object cases | Assignment, no-object and mask/box losses |
| E11 | Prompt quality | Oracle, perturbed, automatic boxes | Detection and mask errors separately |
| E12 | Association | Geometry, pair MLP, contextual transformer | Fixed detections/decoder, association errors |
| E13 | Decoder | Shared scores with changed decoding | Validity, divisions, complete tracks |
| E14 | Temporal context | Context lengths and allowed future information | Accuracy, latency, information constraints |
| E15 | Detection corruption | Known misses/merges/splits introduced | Robustness curves and error propagation |
| E16 | Joint hypotheses | Fixed masks versus candidate selection | Candidate coverage and selected lineages |
| E17 | Limited labels | Several predeclared annotation budgets | Training examples, variation, correction effort |
| E18 | Domain shift | Held-out acquisition/cell conditions | Failure categories and uncertainty behavior |

For a clean ablation, specify what stays fixed before running it. Some practical comparisons cannot perfectly match all budgets; document the mismatch rather than calling them controlled.

Each learned-model run should record seed, parameters, training data, pretrained state, batch/crop/window sizes, optimization, checkpoint criterion, runtime, and peak memory. Use saved paired outputs for comparisons on the same held-out sequences.

## 17. Testing and debugging requirements for the future code

Prioritize mathematical and data invariants over tests that merely repeat an implementation.

Required checks:

- Attention agrees with a trusted reference under equivalent masks and disabled dropout.
- Analytical gradients agree with automatic differentiation on small examples.
- Future-token perturbations cannot affect earlier causal predictions.
- Valid outputs are invariant to padded values.
- Cached and uncached decoding agree, including one-token and chunked continuation.
- Patchify/unpatchify preserves the input before projection.
- Image/mask transforms preserve correspondence and discrete labels.
- Detection permutations preserve predictions up to the corresponding permutation.
- Label remapping does not alter instance geometry or evaluation.
- Candidate edges obey the configured temporal direction and spatial gating.
- Decoded graphs obey the selected lineage model.
- Known fabricated errors affect the intended evaluation measures.
- Reloading a checkpoint restores the claimed computation and configuration.

Debug in this order: inspect data and labels; verify shapes and masks; overfit a tiny example; inspect predictions; verify loss; then scale training. A larger model is not the first response to a broken pipeline.

## 18. Research notes and reporting templates

### Paper card

~~~text
Title / authors / year:
Primary URL / version / access level:
Problem:
Biological target and imaging modality:
Inputs:
Token or feature representation:
Backbone:
Prediction targets:
Reconstruction / association / decoder:
Loss and supervision:
Training data and pretraining:
Evaluation and baselines:
Main result, with scope:
Assumptions:
Observed limitations:
Questions not answered by the accessible text:
One proposed experiment:
Connection to this project's modules:
~~~

### Hypothesis card

~~~text
Question:
Mechanism proposed:
Expected observable change:
Baseline:
One changed factor:
Factors held fixed:
Data and split:
Primary endpoint:
Secondary endpoints / possible tradeoffs:
Failure that would falsify or narrow the claim:
Compute/run budget:
Measured result:
Limited conclusion:
~~~

### Final experiment report

Explain the problem and observed behavior before implementation details. Include a pipeline figure, data/split provenance, mathematical specification, configurations, appropriate metrics, variation across runs/sequences, and a representative failure gallery.

Keep educational adaptation, paper reproduction, and novel method development as different labels. A reproduction requires checking the paper's actual protocol; a new module name does not establish novelty.

## 19. Compute tiers and manageable scope

| Tier | Required work | Scale policy |
|---|---|---|
| CPU core | Numerical attention, gradients, masks, positions, patchification, synthetic matching | Tiny arrays and short sequences |
| Modest GPU core | Small language/vision models, U-Net, detection-token association | Profile a batch, then set explicit token/crop limits |
| Pretrained inspection | SAM/cell models, published tracking checkpoints | Start with inference and small adaptation |
| Advanced optional | Full query-mask models, large 3D/video training, broader domain studies | Require a justified budget and reliable earlier baselines |

Inspect actual hardware, available storage, libraries, and model compatibility before selecting a runtime. Do not assume that an example environment matches the environment where a new experiment will run.

Set bounded run counts and use early diagnostic experiments. Full SAM, DINO, or large language-model pretraining is not needed to satisfy the core learning objective. Selective from-scratch implementation plus careful use of reference models provides a comprehensive path at realistic scale.

## 20. Optional advanced branches

Add these after the relevant prerequisite gate, one branch at a time:

- **Parameter-efficient adaptation:** LoRA's low-rank weight updates, trainable parameter count, optimizer memory, and comparison with full fine-tuning. Read [LoRA](https://arxiv.org/abs/2106.09685).
- **Vision–language learning:** contrastive image/text alignment, positive/negative pairs, similarity temperature, and limitations of text-derived supervision for biological masks. Read [CLIP](https://arxiv.org/abs/2103.00020).
- **Unstructured geometry:** point/mesh tokens, neighborhood attention, relative coordinates, and order invariance. These representations extend the curriculum to unstructured geometry. Read [Point Transformer](https://arxiv.org/abs/2012.09164).
- **Sparse/approximate attention:** distinguish changing the interaction graph from approximating the attention operation. Compare actual memory/runtime and quality.
- **Mixture-of-experts FFNs:** routing, expert capacity, load balancing, active versus total parameters, and why sparsity does not make communication free.
- **Adapters and prompt tuning:** identify which parameters change and which information the prompt carries.
- **Representation probing:** test whether features contain useful biological information; distinguish probe performance from the model using that information causally.
- **Interpretability and intervention:** remove features, change context, randomize positions, or alter candidate graphs. Attention maps alone are insufficient causal evidence.
- **Active annotation:** select uncertain or representative examples, track expert correction time, and evaluate on an independent curated subset.
- **Synthetic-to-real learning:** perturb acquisition/appearance variables and measure which simulated assumptions transfer.
- **Multimodal microscopy:** separate channels, acquisitions, registration, missing modalities, and cross-attention choices.
- **Restoration before segmentation:** test whether denoising helps measurements or erases important structures.
- **Continual/domain adaptation:** study the stability–plasticity tradeoff while measuring retention and avoiding unsupported biological claims.

## 21. Readiness gates and pacing

Use evidence-based gates rather than calendar deadlines:

1. **Architecture gate:** derive attention, masks, gradients, positions, and a full block.
2. **NLP gate:** explain objectives, teacher forcing, generation, and cache correctness.
3. **Vision gate:** trace patch tokens to class/dense/set/mask outputs.
4. **Microscopy gate:** distinguish biological targets and instance reconstruction failures.
5. **Tracking gate:** separate scoring from constrained lineage decoding.
6. **Research gate:** defend data splits, controls, metrics, and a limited conclusion.

For a focused study route, prioritize Modules 04–06, 10, 12, 18–23 and the research paper cards, while referring back to earlier prerequisites as necessary. This route emphasizes the path from attention to microscopy evaluation; it does not imply that the whole curriculum can be mastered in a few days.

For the full project, complete the sequence with implementation and revision. Mark modules as planned, in progress, demonstrated, or independently explained. Do not mark a module complete solely because the assistant wrote its lesson.

## 22. Starting guide for learners

Use this public educational blueprint as a roadmap for an incremental transformer-to-cell-analysis project. Read the prerequisites and complete roadmap before selecting the first implementation task. Treat proposed code, datasets, and experiments as planned until their implementation and verification are available.

Begin with a brief prerequisite assessment and an inspection of the actual computing environment. Select a manageable increment, then work through intuition, equations, tensor shapes, a numerical example, and a small exercise. Adjust the pace to the concepts that can already be independently explained.

Implement central transformer operations explicitly before comparing them with optimized library calls. Explain why each operation exists. Keep a research ledger and progress record. Use primary papers and official documentation, verify software conventions, and distinguish educational adaptations from faithful reproductions.

Work one coherent module at a time. Keep later modules planned until built and checked. Do not invent results, assume unavailable data, or train large models before a small example works. Connect each lesson to vision, cell segmentation, or lineage tracking when the connection is meaningful.

The first implementation increment should normally contain an environment record, repository skeleton, one small tensor/attention notebook, reusable core functions, and the numerical checks necessary to trust that lesson. Save configuration, outputs, discrepancies, and an interpretation. Mark a lesson demonstrated or independently explained only when the corresponding evidence is available.
