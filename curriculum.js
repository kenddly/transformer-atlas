/* Transformer Atlas: public educational curriculum. */
window.CURRICULUM = [
  {
    "id": "00",
    "title": "From pixels to biological questions",
    "phase": "foundations",
    "duration": "6 min",
    "question": "What should a complete microscopy system produce, and what can it honestly tell us?",
    "intuition": "Imagine two bright nuclei moving through a movie. A frame tells you where bright material is. An instance mask separates the two objects. A track links each object to its later observations. A lineage adds a parent–daughter relationship when an object divides. Each step answers a different question, and an error early in the chain can change the final biological measurement.",
    "essential": [
      "Begin with the biological target. A fluorescent marker may reveal nuclei rather than whole cells. A bright region is an observation of a labelled structure, not automatically a complete physical cell. Images, instance masks, detections, tracks, and lineage graphs should remain separate outputs.",
      "An instance label image assigns one integer to every pixel belonging to an object in a frame. Those numbers are arbitrary: exchanging labels 1 and 2 does not change the segmentation. A persistent track identity instead represents a hypothesis that observations across time belong to the same object.",
      "A split mask can imitate division; a merged mask can imitate fusion. Appearance may mean entry into the field of view, and disappearance may mean a missed detection. Neither observation alone establishes biological birth or death."
    ],
    "technical": [
      "Represent the image as an intensity array, the segmentation as a discrete label array, and each detection as features derived from one labelled instance. Build a time-directed graph whose nodes are detections and whose selected edges represent continuation or division relationships.",
      "Declare units before deriving motion: a displacement of five pixels means different distances under different pixel spacings. A migration speed also requires the frame interval. Keep image geometry, object geometry, and time metadata together."
    ],
    "research": [
      "Define the biological target and the observation process before interpreting a model output. A fluorescence image may reveal nuclei rather than whole cells. Segmentation, association, lineage reconstruction, and biological measurement each introduce different assumptions.",
      "An intervention can improve one endpoint while harming another. Better foreground overlap may coexist with worse instance separation or lineage quality. Report several relevant endpoints, and distinguish a deterministic rerun from independent-seed replication."
    ],
    "equation": "image → instance masks → detections → association scores → valid lineages → biological measurements",
    "shapes": [
      "Image: \\([B, C, H, W]\\)",
      "Instance labels: \\([B, H, W]\\) integer IDs",
      "Detection table: \\([N, n_{\\mathrm{features}}]\\)",
      "Lineage: detection nodes + time-forward event edges"
    ],
    "numerical": "Two nuclei in frame 0 and two in frame 1 permit four possible pairwise links. Selecting two links requires an identity decision. If a mask merges the two nuclei, the next stage sees only one detection and cannot recover the missing object from that fixed detection table.",
    "exercise": "Write your current explanations of attention, instance segmentation, and tracking. Draw two moving objects over three frames and annotate masks, identities, and one possible division separately. Revisit these explanations after Modules 06, 12, and 21.",
    "failure": "Count two daughters after a segmentation split and infer a division without checking the image or temporal evidence. Explain which observations would distinguish the competing explanations.",
    "check": {
      "question": "Can two frames use different instance numbers for the same tracked nucleus?",
      "options": [
        "Yes; instance IDs are frame-local, while the track records correspondence.",
        "No; instance IDs are physical identities.",
        "Only if the nucleus has divided."
      ],
      "answer": 0,
      "explanation": "Frame-local labels encode object geometry. A separate mapping carries persistent identity through time."
    },
    "deliverable": "Baseline explanation, target definition, and a diagram of the full measurement pipeline.",
    "evidence": "Save the initial explanations and a labelled toy movie. Record whether each claim concerns image appearance, segmentation, tracking, or biology.",
    "lab": "tracking",
    "readings": [
      {
        "title": "Cell Tracking Challenge overview",
        "url": "https://celltrackingchallenge.net/"
      },
      {
        "title": "CBIA organoid workflow",
        "url": "https://cbia.fi.muni.cz/research/spatiotemporal/organoids.html"
      }
    ],
    "equationLatex": "\\begin{gathered}\\text{image}\\to\\text{instance masks}\\to\\text{detections}\\\\\\to\\text{association scores}\\to\\text{valid lineages}\\\\\\to\\text{biological measurements}\\end{gathered}"
  },
  {
    "id": "01",
    "title": "The mathematical toolkit",
    "phase": "foundations",
    "duration": "8 min",
    "question": "Which operations and numerical habits make later diagrams trustworthy?",
    "intuition": "A tensor is an array with named axes. Matrix multiplication combines compatible feature axes; reshape reorganizes existing values; a learned projection changes the representation. Tracing those distinctions is often more useful than memorizing a model’s name.",
    "essential": [
      "A dot product multiplies corresponding vector elements and sums them. Matrix multiplication repeats that calculation for rows and columns. If \\(X\\) has shape \\([B,T,D]\\) and \\(W\\) has shape \\([D,E]\\), \\(XW\\) has shape \\([B,T,E]\\): each token receives the same feature transformation.",
      "Logits are unrestricted scores. Softmax converts one row of scores into nonnegative probabilities that sum to one. Cross-entropy penalizes a low probability assigned to the correct class. Entropy describes how spread out the distribution is; it does not establish whether it is correct.",
      "The chain rule carries an output error backward through every contributing operation. Automatic differentiation computes this bookkeeping on the recorded computation graph. Gradients accumulate unless explicitly reset, so unintended accumulation changes training."
    ],
    "technical": [
      "Use stable softmax: subtract each row maximum before exponentiating. This leaves probabilities unchanged because the shared factor cancels. Prefer a stable log-softmax for log probabilities rather than computing \\(\\log\\!\\bigl(\\operatorname{softmax}(z)\\bigr)\\) after small probabilities underflow.",
      "Permutation changes axis order; reshape changes grouping. A view may share storage and may require compatible strides. Transposing an array does not guarantee contiguous storage. Broadcasting aligns compatible axes; a misplaced singleton can silently produce the wrong computation.",
      "Check a small analytic gradient against automatic differentiation and finite differences. A central difference \\(\\frac{L(w+\\varepsilon)-L(w-\\varepsilon)}{2\\varepsilon}\\) balances truncation and rounding errors; excessively small \\(\\varepsilon\\) can be unreliable in low precision. Use tolerances appropriate to the dtype.",
      "Dropout behaves differently in training and evaluation. Parameter count, stored activations, arithmetic operations, and elapsed runtime measure different resources; none can be inferred exactly from another."
    ],
    "research": [
      "Before scaling a model, verify a tiny forward/backward computation, inspect numerical ranges, and profile one step. Keep the resolved software versions and actual hardware with the configuration. Inspect the current environment directly."
    ],
    "equation": "softmax(z)ᵢ = exp(zᵢ − max(z)) / Σⱼ exp(zⱼ − max(z))",
    "shapes": [
      "\\(X\\): \\([B, T, D]\\)",
      "\\(W\\): \\([D, E]\\)",
      "\\(XW+\\mathrm{bias}\\): \\([B, T, E]\\)",
      "Row probabilities: \\([B, T, n_{\\mathrm{classes}}]\\)"
    ],
    "numerical": "For logits \\([1000,1001]\\), direct exponentiation may overflow. Subtracting 1001 gives \\([-1,0]\\), so the probabilities are approximately \\([0.269,0.731]\\). For \\(L=w^2\\) at \\(w=3\\), the derivative is 6.",
    "exercise": "Trace a two-layer network by hand. Record every axis, compute a stable softmax, and compare one parameter gradient with a central finite difference.",
    "failure": "Normalize softmax across the batch instead of classes. Predictions then depend on unrelated examples in the batch; the rows no longer have the intended probabilistic meaning.",
    "check": {
      "question": "What is the output shape of \\([2,5,3]\\) multiplied by \\([3,7]\\)?",
      "options": [
        "\\([2,5,7]\\)",
        "\\([2,3,7]\\)",
        "\\([5,7,3]\\)"
      ],
      "answer": 0,
      "explanation": "The shared feature dimension 3 is summed. Batch 2 and token count 5 are preserved, and the output feature width is 7."
    },
    "deliverable": "A tensor and gradient notebook with explicit numerical tolerances.",
    "evidence": "Save inputs, intermediate values, analytical and automatic gradients, finite-difference step sizes, and maximum discrepancies.",
    "lab": "shapes",
    "readings": [
      {
        "title": "PyTorch automatic differentiation",
        "url": "https://docs.pytorch.org/docs/stable/notes/autograd.html"
      },
      {
        "title": "PyTorch Linear",
        "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html"
      }
    ],
    "equationLatex": "\\operatorname{softmax}(z)_i=\\frac{\\exp(z_i-\\max_j z_j)}{\\sum_j\\exp(z_j-\\max_k z_k)}"
  },
  {
    "id": "02",
    "title": "Why attention emerged",
    "phase": "foundations",
    "duration": "7 min",
    "question": "Which design problems led from recurrence to attention-centered architectures?",
    "intuition": "A recurrent encoder reads a sentence one step at a time and carries a state. Asking its final state to support every future output creates pressure to retain many details. Recurrent attention lets the decoder revisit different encoder states at each step instead of depending on a single summary.",
    "essential": [
      "A recurrent neural network updates a hidden state from the current input and previous state. A long short-term memory network (LSTM) adds gates that regulate information entering, leaving, and persisting in its cell state. Gates help optimization; they do not guarantee perfect long-term memory.",
      "Sequence-to-sequence learning uses an encoder for a source and a decoder for a target. In translation, different target words may depend on different source positions. Bahdanau attention forms a weighted context from encoder states while retaining recurrent computation.",
      "The 2017 transformer organizes interactions around attention without recurrent layers in the proposed architecture. It still needs embeddings, positional information, feed-forward networks, residuals, normalization, and output projections. Attention predates transformers."
    ],
    "technical": [
      "An RNN update \\(h_t=\\phi(W_xx_t+W_hh_{t-1}+b)\\) has a sequential state dependency. In a recurrent attention decoder, compatibility between the decoder state and each source state produces weights, and a weighted source context feeds generation.",
      "A transformer encoder can process source positions in parallel within a layer. Teacher-forced decoder training can compute target positions together under a causal mask. Autoregressive generation still selects an output token before feeding it into the next step."
    ],
    "research": [
      "Compare copy or reversal tasks using recurrent sequence-to-sequence models with and without attention. Label these small educational demonstrations, rather than reproductions of translation benchmarks. Architecture, optimization, data, and task length all affect the result."
    ],
    "equation": "hₜ = φ(Wₓxₜ + Wₕhₜ₋₁ + b); contextₜ = Σⱼ αₜⱼ encoder_stateⱼ",
    "shapes": [
      "Source embeddings: \\([B, T_{\\mathrm{source}}, D]\\)",
      "Encoder states: \\([B, T_{\\mathrm{source}}, D_{\\mathrm{state}}]\\)",
      "Decoder-to-source weights: \\([B, T_{\\mathrm{target}}, T_{\\mathrm{source}}]\\)"
    ],
    "numerical": "Suppose the source states are \\([1,0]\\) and \\([0,2]\\). Weights \\([0.75,0.25]\\) produce context \\([0.75,0.5]\\). A later decoder step may use \\([0.1,0.9]\\), yielding \\([0.1,1.8]\\) from the same source.",
    "exercise": "Draw an RNN state chain, a recurrent attention decoder, and the original transformer encoder–decoder. Mark which computations depend on a previously generated output.",
    "failure": "Attribute a reversal-task improvement solely to architectural superiority when one model saw more data or used a larger training budget.",
    "check": {
      "question": "Which statement describes the history accurately?",
      "options": [
        "Attention was used in recurrent translation models before the transformer.",
        "Attention first appeared in ViT.",
        "Transformer generation removes all sequential dependencies."
      ],
      "answer": 0,
      "explanation": "Selective attention over encoder states preceded the transformer. Autoregressive generation remains sequential even when training is parallelized."
    },
    "deliverable": "An architecture comparison and a bounded recurrent copy/reversal experiment plan.",
    "evidence": "Save task lengths, training budget, seed, predictions, and failure examples before drawing a conclusion.",
    "readings": [
      {
        "title": "LSTM, 1997",
        "url": "https://doi.org/10.1162/neco.1997.9.8.1735"
      },
      {
        "title": "Sequence to Sequence Learning",
        "url": "https://arxiv.org/abs/1409.3215"
      },
      {
        "title": "Bahdanau attention",
        "url": "https://arxiv.org/abs/1409.0473"
      },
      {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762"
      }
    ],
    "equationLatex": "\\begin{aligned}h_t&=\\phi(W_xx_t+W_hh_{t-1}+b)\\\\c_t&=\\sum_j\\alpha_{tj}\\,h_j^{\\mathrm{enc}}\\end{aligned}"
  },
  {
    "id": "03",
    "title": "Tokens and learned embeddings",
    "phase": "foundations",
    "duration": "6 min",
    "question": "How does discrete text become a differentiable representation?",
    "intuition": "A token ID is a row address, not a measurement. The embedding table is a trainable dictionary of vectors. The same token begins with the same lookup vector in different sentences, but later attention layers can give it different contextual representations.",
    "essential": [
      "Character, byte, word, and subword tokenization trade vocabulary size against sequence length. A word vocabulary has many entries and struggles with unseen words; byte or character sequences use a smaller vocabulary but often require more positions. Normalization and special tokens are part of the tokenizer specification.",
      "Byte-pair encoding repeatedly merges frequent adjacent units learned from a training corpus. A tokenizer chooses discrete units; learning embeddings adjusts their vectors. SentencePiece is a toolkit supporting tokenization approaches, rather than one universal tokenization algorithm.",
      "Beginning, ending, padding, and unknown-token conventions affect both inputs and loss. Train a learned tokenizer only on training text so that evaluation data does not influence the preprocessing fit."
    ],
    "technical": [
      "An embedding table \\(E\\) has shape \\([V_{\\mathrm{vocab}},D]\\). Integer IDs \\([B,T]\\) select rows and produce \\([B,T,D]\\). ID 100 is not inherently more similar to ID 101 than ID 900. Similarity depends on learned vectors and the task.",
      "Repeated occurrences of a token contribute gradients to the same table row. A contextual hidden state after several transformer layers has the same feature width as the input embedding in a conventional block, but includes information gathered from permitted neighbors."
    ],
    "research": [
      "Inspect tokenization of English phrases, names from several languages, and microscopy terms. A domain word split into several pieces is not automatically unusable, but it changes sequence length and the units over which likelihood is reported. Perplexities from incompatible tokenizations should not be compared as if they measured the same events."
    ],
    "equation": "X[b,t,:] = E[token_id[b,t],:]",
    "shapes": [
      "IDs: \\([B, T]\\) integer values",
      "Embedding table: \\([V_{\\mathrm{vocab}}, D]\\)",
      "Input vectors: \\([B, T, D]\\)",
      "Contextual vectors: \\([B, T, D]\\)"
    ],
    "numerical": "With table rows \\(0\\mapsto[1,0]\\), \\(1\\mapsto[0,1]\\), and \\(2\\mapsto[1,1]\\), IDs \\([2,0,2]\\) produce \\(\\begin{bmatrix}1&1\\\\1&0\\\\1&1\\end{bmatrix}\\). Both uses of row 2 update the same parameters during learning.",
    "exercise": "Perform several merges on a tiny repeated-word corpus. Then implement row lookup and inspect how two occurrences of one ID accumulate gradients.",
    "failure": "Treat integer IDs as continuous numerical features or fit the vocabulary on held-out text. The former invents an arbitrary distance; the latter leaks evaluation information.",
    "check": {
      "question": "What gives a token its contextual representation?",
      "options": [
        "Its hidden state after information mixing in the model.",
        "Its integer ID becoming a larger integer.",
        "Its fixed embedding row alone."
      ],
      "answer": 0,
      "explanation": "The input lookup is context-independent. Attention and later transformations condition hidden states on permitted surrounding content."
    },
    "deliverable": "A toy tokenizer, embedding lookup, and vocabulary/sequence-length comparison.",
    "evidence": "Save normalization rules, training corpus membership, vocabulary, merges, special-token IDs, and example tokenizations.",
    "readings": [
      {
        "title": "Subword units",
        "url": "https://arxiv.org/abs/1508.07909"
      },
      {
        "title": "SentencePiece",
        "url": "https://arxiv.org/abs/1808.06226"
      },
      {
        "title": "PyTorch Embedding",
        "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html"
      }
    ],
    "equationLatex": "X_{b,t,:}=E_{\\operatorname{token\\_id}(b,t),:}"
  },
  {
    "id": "04",
    "title": "Attention, one operation at a time",
    "phase": "foundations",
    "duration": "12 min",
    "question": "How does one token gather information from other tokens?",
    "intuition": "Each receiving token creates a query, each possible source creates a key, and each source also provides a value. Query–key compatibility determines how much of each value is mixed into the output. The compatibility weights and the content being mixed play different roles.",
    "essential": [
      "Let \\(X_q\\) be receiving features and \\(X_k\\) be source features. Learned matrices produce \\(Q=X_qW_Q\\), \\(K=X_kW_K\\), and \\(V=X_kW_V\\). In self-attention the two input streams are the same; in cross-attention they can come from different sources and have different lengths.",
      "Compute every query’s dot product with every permitted key, divide by the square root of the compatibility width, add an access mask, and apply softmax across keys. Each resulting row sums to one over allowed sources. Multiply that row by \\(V\\) to form a weighted content mixture.",
      "\\(Q\\) describes compatibility features of a receiver, \\(K\\) describes compatibility features of a source, and \\(V\\) carries source content. These are learned projections, not literal linguistic questions and answers. A head has no guaranteed human-interpretable role.",
      "With several heads, project and reshape to \\([B,h,T,d_h]\\), run attention per head, concatenate head outputs, and apply an output projection. Each head can use every original input feature; it is not assigned a fixed semantic slice of the input."
    ],
    "technical": [
      "Scaling is motivated by an assumption: independent zero-mean query/key components with unit variance produce a dot-product variance growing with \\(d_h\\). Dividing by \\(\\sqrt{d_h}\\) moderates score magnitude. This is a useful initialization argument, not a claim that trained components always satisfy those assumptions.",
      "Given upstream output gradient \\(G\\), the value path receives \\(\\mathrm dV=A^{\\mathsf T}G\\), and the weights receive \\(\\mathrm dA=GV^{\\mathsf T}\\). For each row, \\(\\mathrm dS=A\\odot\\left(\\mathrm dA-\\sum_{\\mathrm{keys}}(A\\odot\\mathrm dA)\\right)\\). The compatibility paths then receive \\(\\mathrm dQ=\\frac{\\mathrm dS\\,K}{\\sqrt{d_h}}\\) and \\(\\mathrm dK=\\frac{\\mathrm dS^{\\mathsf T}Q}{\\sqrt{d_h}}\\).",
      "Projection gradients are \\(\\mathrm dW_Q=\\sum_{\\mathrm{batch}}X_q^{\\mathsf T}\\mathrm dQ\\), with corresponding \\(K\\)/\\(V\\) expressions. Inputs receive \\(\\mathrm dX_q=\\mathrm dQ\\,W_Q^{\\mathsf T}\\) and \\(\\mathrm dX_k=\\mathrm dK\\,W_K^{\\mathsf T}+\\mathrm dV\\,W_V^{\\mathsf T}\\). In self-attention one shared input receives all three contributions. A fixed blocking mask has no trainable gradient; a learned relative bias does.",
      "\\(Q\\) and \\(K\\) must share compatibility width. \\(V\\) may have a different content width, although equal widths are convenient in the educational example. Optional biases and a fused QKV projection change organization without changing this central computation."
    ],
    "research": [
      "Use an ordinary matrix implementation as the reference, then compare an optimized implementation with equivalent masks and dropout disabled. Attention maps describe internal weighting; they are not sufficient evidence for causal explanations of model predictions. Test interventions on values, features, or context."
    ],
    "equation": "S = QKᵀ / √d_h + M; A = softmax_keys(S); O = AV",
    "shapes": [
      "\\(X_q\\): \\([B, T_q, D]\\); \\(X_k\\): \\([B, T_k, D]\\)",
      "\\(W_Q\\), \\(W_K\\), \\(W_V\\): \\([D, d_h]\\) in the simple equal-width head",
      "\\(Q\\): \\([B, T_q, d_h]\\); \\(K,V\\): \\([B, T_k, d_h]\\)",
      "Scores and weights: \\([B, T_q, T_k]\\)",
      "Output: \\([B, T_q, d_h]\\)",
      "Multihead scores: \\([B, h, T_q, T_k]\\)"
    ],
    "numerical": "For one query \\(q=[1,0]\\), keys \\([1,0]\\) and \\([0,1]\\), and width 2, scores are \\([0.707,0]\\). Softmax gives about \\([0.670,0.330]\\). With values \\([2,0]\\) and \\([0,4]\\), the output is \\([1.340,1.321]\\). Changing only \\(V\\) changes the mixture content without changing those weights.",
    "exercise": "Calculate three tokens and two-dimensional projections by hand, including one complete softmax row. Derive the backward pass, compare with automatic differentiation, then inspect the maximum numerical discrepancy.",
    "failure": "Apply softmax over queries instead of keys, or interpret a large weight as evidence that a source feature caused the final decision. Diagnose each error separately.",
    "check": {
      "question": "If \\(V\\) changes while \\(Q\\) and \\(K\\) stay fixed, what changes?",
      "options": [
        "The output content changes; attention weights stay fixed.",
        "Only attention weights change.",
        "Nothing changes."
      ],
      "answer": 0,
      "explanation": "\\(Q\\) and \\(K\\) determine compatibility. \\(V\\) is mixed using those fixed weights, so output content changes independently."
    },
    "deliverable": "Hand-derived attention, backward pass, and a transparent reference implementation.",
    "evidence": "Save projection matrices, scores, masks, weights, output, all gradients, dtype, and agreement tolerances. This browser calculator is a numerical demonstration rather than a trained model.",
    "lab": "attention",
    "readings": [
      {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762"
      },
      {
        "title": "MultiheadAttention conventions",
        "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html"
      }
    ],
    "equationLatex": "\\begin{aligned}S&=\\frac{QK^{\\mathsf T}}{\\sqrt{d_h}}+M\\\\A&=\\operatorname{softmax}_{\\mathrm{keys}}(S)\\\\O&=AV\\end{aligned}"
  },
  {
    "id": "05",
    "title": "Masks define information access",
    "phase": "foundations",
    "duration": "8 min",
    "question": "Which observations is each prediction allowed to use?",
    "intuition": "A mask is an access rule. A causal mask blocks future information; padding masks block artificial inputs; neighborhood masks restrict candidates; loss masks choose which predictions are graded. These are different decisions even when they share a similar array shape.",
    "essential": [
      "For next-token training, input “BOS cells can divide” aligns with targets “cells can divide EOS”. The targets are known during teacher forcing, but the prediction at each position must still obey its permitted prefix. During generation, selected tokens are fed back one step at a time.",
      "Block forbidden attention logits before softmax. Setting an input value to zero is not an equivalent rule: that source can still receive weight and affect the normalization. Padding must not influence valid outputs, and padded labels must be excluded from the intended loss.",
      "A key-padding mask does not automatically zero padded query outputs. A loss mask does not prevent an input from influencing other predictions. Inspect information-access and supervision masks independently."
    ],
    "technical": [
      "An additive mask can use zero for allowed entries and negative infinity for forbidden entries. An entirely blocked row has no valid probability distribution under an ordinary softmax. Choose an explicit handling policy and check it instead of trusting accidental implementation behavior.",
      "API conventions differ. The README flags that PyTorch scaled-dot-product attention uses Boolean True for allowed positions, while MultiheadAttention Boolean masks use True for blocked positions. Verify conventions against the installed version. A direct scaled-dot-product attention call also needs an explicitly zero dropout probability during evaluation.",
      "Causal correctness is a behavioral invariant: with fixed parameters and deterministic evaluation, modifying a future input must leave earlier causal outputs unchanged. Padding correctness means changing only padded content cannot change valid predictions."
    ],
    "research": [
      "Information availability depends on the task. A language next-token task is causal; an offline microscopy window may legitimately use future frames. Declare the online/offline setting before selecting masks, and keep it fixed when comparing models."
    ],
    "equation": "Mᵢⱼ = 0 if source j is allowed for query i; Mᵢⱼ = −∞ otherwise",
    "shapes": [
      "Causal matrix: \\([T, T]\\)",
      "Key validity: \\([B, T_{\\mathrm{key}}]\\)",
      "Broadcast access mask: compatible with \\([B, h, T_{\\mathrm{query}}, T_{\\mathrm{key}}]\\)",
      "Loss validity: \\([B, T_{\\mathrm{target}}]\\)"
    ],
    "numerical": "For three causal positions, the allowed rows are \\([1,0,0]\\), \\([1,1,0]\\), and \\([1,1,1]\\). If the unmasked scores of row 0 are \\([0,9,9]\\), a correct mask still gives weights \\([1,0,0]\\).",
    "exercise": "Change the final token and compare earlier outputs with and without a causal mask. Repeat with altered padded content. Print all masks before executing attention.",
    "failure": "A model with leaked future tokens can obtain an impressive teacher-forced loss while failing at generation. The loss measured access to the answer rather than the intended predictive task.",
    "check": {
      "question": "Does excluding padded targets from the loss prevent padded input keys from affecting valid tokens?",
      "options": [
        "No; an attention padding mask is also required.",
        "Yes; loss masking controls all information flow.",
        "Only when sequence lengths match."
      ],
      "answer": 0,
      "explanation": "Loss selection controls which outputs are scored. Access masks control which inputs influence those outputs."
    },
    "deliverable": "Causality, padding-invariance, and deliberate leakage demonstrations.",
    "evidence": "Save before/after valid outputs, masks, loss selection, and the maximum earlier-output difference.",
    "lab": "attention",
    "readings": [
      {
        "title": "Scaled-dot-product attention documentation",
        "url": "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"
      }
    ],
    "equationLatex": "M_{ij}=\\begin{cases}0,&\\text{source }j\\text{ is allowed for query }i\\\\-\\infty,&\\text{otherwise}\\end{cases}"
  },
  {
    "id": "06",
    "title": "Build the full transformer block",
    "phase": "foundations",
    "duration": "10 min",
    "question": "What makes attention part of a trainable deep network?",
    "intuition": "Attention mixes information between positions. A feed-forward network changes features at each position. Residual paths let a representation retain its previous value while learning an adjustment. Normalization controls feature scale, with its location affecting how information and gradients move.",
    "essential": [
      "Use a pre-normalized educational block: \\(u=x+\\operatorname{Dropout}(\\operatorname{MHA}(\\operatorname{LN}_1(x)))\\); \\(y=u+\\operatorname{Dropout}(\\operatorname{FFN}(\\operatorname{LN}_2(u)))\\). \\(\\operatorname{LN}_1\\) and \\(\\operatorname{LN}_2\\) have separate learned parameters. A conventional pre-norm stack also documents its final normalization.",
      "Layer normalization computes mean and population variance over the chosen feature dimensions of each token, then applies a learned scale and offset. Batch normalization uses different statistical axes. RMS normalization divides by a root-mean-square scale without subtracting the mean.",
      "The positionwise feed-forward network projects \\(D\\) features into a hidden width, applies a nonlinear function, and projects back to \\(D\\). The same weights are applied at each position. It mixes channels without directly exchanging information between positions.",
      "Dropout on attention weights differs from dropout on sublayer outputs. Evaluation disables dropout modules, and direct attention APIs may require their own dropout setting. Residual addition requires compatible shapes."
    ],
    "technical": [
      "Pre-norm uses \\(x+F(\\operatorname{LN}(x))\\); post-norm uses \\(\\operatorname{LN}(x+F(x))\\). The original transformer used a post-norm arrangement. Moving normalization changes optimization and gradient paths, so compare it as an explicit design intervention.",
      "For a token vector \\(x\\), \\(\\operatorname{LN}(x)=\\gamma\\odot\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\varepsilon}}+\\beta\\), using the population variance over \\(D\\) features. Epsilon limits division by very small variance. RMSNorm’s denominator uses \\(\\operatorname{mean}(x^2)\\); omitting a bias from LayerNorm does not make it RMSNorm.",
      "An ordinary feed-forward network is \\(\\phi(xW_1+b_1)W_2+b_2\\). ReLU clips negative values, whereas GELU uses a smooth gating behavior. SwiGLU multiplies \\(\\operatorname{SiLU}(xW_g)\\) by \\(xW_u\\) and projects with \\(W_d\\). Match parameter counts when comparing gated and ordinary variants; equal expansion ratios can have unequal budgets.",
      "Count embeddings, \\(Q\\)/\\(K\\)/\\(V\\) and output projections, both normalizations, feed-forward weights, biases, and output heads. Plot activations and gradients across depth when changing residuals, width, initialization, or optional stochastic depth."
    ],
    "research": [
      "Ablate one factor at a time: remove residuals, move normalization, or change feed-forward width. Report observed instability with its settings rather than declaring that one choice always fails. A short successful loss curve does not establish a universally better architecture."
    ],
    "equation": "u = x + Dropout(MHA(LN₁(x))); y = u + Dropout(FFN(LN₂(u)))",
    "shapes": [
      "Input/residual stream: \\([B, T, D]\\)",
      "Head projections: \\([B, h, T, d_h]\\)",
      "Concatenated attention: \\([B, T, D]\\)",
      "FFN expansion: \\([B, T, D_{\\mathrm{ff}}]\\)",
      "FFN contraction/output: \\([B, T, D]\\)"
    ],
    "numerical": "For features \\([1,3]\\), the mean is 2 and population variance is 1. Ignoring epsilon and using \\(\\gamma=1\\), \\(\\beta=0\\), LayerNorm gives \\([-1,1]\\). RMS normalization instead divides \\([1,3]\\) by \\(\\sqrt{5}\\), giving approximately \\([0.447,1.342]\\).",
    "exercise": "Implement attention, normalization, residuals, and the feed-forward network with ordinary tensor operations. Reconstruct the original post-norm arrangement separately and compare activation statistics.",
    "failure": "Normalize across tokens when the intended LayerNorm axis is features, or add a sublayer output of width D_ff directly to a residual of width \\(D\\).",
    "check": {
      "question": "Which sublayer directly mixes information across different token positions?",
      "options": [
        "Attention.",
        "The positionwise FFN.",
        "A feature-axis LayerNorm alone."
      ],
      "answer": 0,
      "explanation": "Attention aggregates source values across positions. The positionwise FFN and feature-wise normalization operate independently on each token."
    },
    "deliverable": "An explicit transformer block, small stack, and activation/gradient inspection.",
    "evidence": "Save parameter count, normalization axes, dropout settings, initialization, activation ranges, and gradient ranges.",
    "lab": "shapes",
    "readings": [
      {
        "title": "Residual learning",
        "url": "https://arxiv.org/abs/1512.03385"
      },
      {
        "title": "Layer Normalization",
        "url": "https://arxiv.org/abs/1607.06450"
      },
      {
        "title": "Normalization placement",
        "url": "https://arxiv.org/abs/2002.04745"
      },
      {
        "title": "RMSNorm",
        "url": "https://arxiv.org/abs/1910.07467"
      },
      {
        "title": "GLU variants",
        "url": "https://arxiv.org/abs/2002.05202"
      }
    ],
    "equationLatex": "\\begin{aligned}u&=x+\\operatorname{Dropout}\\bigl(\\operatorname{MHA}(\\operatorname{LN}_1(x))\\bigr)\\\\y&=u+\\operatorname{Dropout}\\bigl(\\operatorname{FFN}(\\operatorname{LN}_2(u))\\bigr)\\end{aligned}"
  },
  {
    "id": "07",
    "title": "Architectures and learning objectives",
    "phase": "foundations",
    "duration": "8 min",
    "question": "Which information path and objective fit the prediction problem?",
    "intuition": "An architecture defines who can read whom. An objective defines which answer is rewarded. A bidirectional encoder, a causal decoder, and an encoder–decoder can share attention operations while solving different learning problems.",
    "essential": [
      "An encoder-only model usually allows bidirectional access within its permitted input. A decoder-only language model uses a causal prefix. An encoder–decoder reads source content in an encoder and lets a causal target decoder access that source through cross-attention.",
      "In cross-attention, queries originate from the receiving stream, while keys and values come from the source. Query and source lengths need not match. The source can be text, image features, or another observation stream.",
      "BERT’s original training included masked-language modeling and a next-sentence objective. The latter is a historical design choice, not a requirement for encoders. Autoregressive models predict successive tokens; T5 frames tasks as text-to-text and uses span corruption in pretraining.",
      "Pretraining supplies an initial representation; fine-tuning changes parameters for a later task; conditioning provides input context; teacher forcing supplies known target prefixes during training. These describe different aspects of a learning system."
    ],
    "technical": [
      "A vocabulary projection converts \\([B,T,D]\\) into logits \\([B,T,V_{\\mathrm{vocab}}]\\). Next-token negative log-likelihood averages \\(-\\log p(\\text{correct target})\\) over valid targets. Perplexity is \\(\\exp(\\text{mean negative log-likelihood})\\), with comparisons requiring compatible tokenization and evaluation conventions.",
      "Input/output weight tying can reuse the embedding table as the vocabulary projection when dimensions agree. The shared parameters then receive both lookup-path and output-projection gradients.",
      "For variable-length microbatches, accumulating independently averaged losses gives each microbatch equal weight. To reproduce a mean over all valid tokens, accumulate summed losses and valid-token counts instead."
    ],
    "research": [
      "Run tiny causal-token, masked-token, and conditional-generation experiments separately. Compare information access and supervision without claiming that a topology uniquely determines its objective. Contrastive image/text objectives and image reconstruction are later branches, not automatic substitutes for task labels."
    ],
    "equation": "L = −(1/N_valid) Σ_valid log pθ(yₜ | y<ₜ, source)",
    "shapes": [
      "Source features: \\([B, T_{\\mathrm{source}}, D_{\\mathrm{source}}]\\)",
      "Decoder queries: \\([B, T_{\\mathrm{target}}, d_h]\\)",
      "Cross-attention scores: \\([B, h, T_{\\mathrm{target}}, T_{\\mathrm{source}}]\\)",
      "Vocabulary logits: \\([B, T_{\\mathrm{target}}, V_{\\mathrm{vocab}}]\\)"
    ],
    "numerical": "One microbatch contains 2 valid tokens with mean loss 1; another contains 8 with mean loss 3. Averaging the two means gives 2. The token-weighted mean is \\(\\frac{2\\times1+8\\times3}{10}=2.6\\).",
    "exercise": "Trace \\(Q\\)/\\(K\\)/\\(V\\) origins in an encoder–decoder. Create shifted next-token targets, ignore padded targets, and explain your loss denominator.",
    "failure": "Compare perplexities across incompatible tokenizers or change microbatch composition without noticing that the loss weighting changed.",
    "check": {
      "question": "Where do keys and values come from in encoder–decoder cross-attention?",
      "options": [
        "Encoder/source features.",
        "Only the decoder query stream.",
        "The integer target labels."
      ],
      "answer": 0,
      "explanation": "The decoder receives source information by querying projected encoder features."
    },
    "deliverable": "Small language-objective experiments and an information-flow diagram.",
    "evidence": "Save target alignment, valid-token counts, tokenizer, masks, loss definition, and representative generations.",
    "readings": [
      {
        "title": "BERT",
        "url": "https://arxiv.org/abs/1810.04805"
      },
      {
        "title": "Language Models are Few-Shot Learners",
        "url": "https://arxiv.org/abs/2005.14165"
      },
      {
        "title": "T5",
        "url": "https://arxiv.org/abs/1910.10683"
      }
    ],
    "equationLatex": "\\mathcal L=-\\frac{1}{N_{\\mathrm{valid}}}\\sum_{t\\in\\mathcal T_{\\mathrm{valid}}}\\log p_\\theta(y_t\\mid y_{<t},\\mathrm{source})"
  },
  {
    "id": "08",
    "title": "Position, order, and physical geometry",
    "phase": "foundations",
    "duration": "8 min",
    "question": "How can content-based interactions know where an element belongs?",
    "intuition": "Shuffle a set of unpositioned tokens consistently, and an ordinary unmasked self-attention layer produces the correspondingly shuffled outputs. It recognizes relationships between content, not an intrinsic order. Position supplies additional structure, but an arbitrary detection-list index should not become a biological coordinate.",
    "essential": [
      "Learned absolute position vectors add a trainable location representation to token features. Sinusoidal positions use fixed frequencies. Relative representations or biases describe displacement between positions. Rotary positional embeddings (RoPE) rotate query/key feature pairs according to position.",
      "The permutation-equivariance demonstration assumes no positional signal and no order-dependent access mask. A causal mask itself introduces ordering, so state the conditions rather than claiming attention is always order-blind.",
      "For images, raster order is a storage convention; nearby raster indices are not always spatial neighbors. Use two- or three-dimensional geometry when appropriate. For tracking, coordinates should reflect physical space, and elapsed time matters when frame intervals vary."
    ],
    "technical": [
      "A two-dimensional rotation is \\(R(\\theta)=\\begin{bmatrix}\\cos\\theta&-\\sin\\theta\\\\\\sin\\theta&\\cos\\theta\\end{bmatrix}\\). Orthogonality preserves norms. The dot product of \\(R(m\\theta)q\\) and \\(R(n\\theta)k\\) depends on the relative offset \\(n-m\\) through \\(q^{\\mathsf T}R((n-m)\\theta)k\\).",
      "Being mathematically defined at a longer sequence position is different from reliable long-context behavior after training. Coordinate scaling, interpolation, positional frequencies, and the training distribution can change extrapolation behavior."
    ],
    "research": [
      "Compare no position, index position, and physical/relative position while holding data and model budget fixed. In cell association, randomly reorder detections within each frame and preserve the mappings; predictions should change only by the corresponding permutation."
    ],
    "equation": "(R(mθ)q)ᵀ(R(nθ)k) = qᵀR((n−m)θ)k",
    "shapes": [
      "Absolute text position: \\([T, D]\\)",
      "Image coordinates: \\([N, 2]\\)",
      "Volume coordinates: \\([N, 3]\\)",
      "Cell position/time features: \\([B, N, d_{\\mathrm{spatial}} + 1]\\)"
    ],
    "numerical": "Let \\(q=k=[1,0]\\) and \\(\\theta=90^\\circ\\). At positions 0 and 1 the rotated vectors are \\([1,0]\\) and \\([0,1]\\), so their dot product is 0. At positions 1 and 2 the dot product is also 0: the relative offset is unchanged.",
    "exercise": "Shuffle unpositioned tokens and verify output permutation. Then add inconsistent positions and explain the failure. Numerically verify the rotation identity.",
    "failure": "Use arbitrary within-frame object indices as physical positions. Reordering a detection table then changes correspondence predictions even though the movie is unchanged.",
    "check": {
      "question": "Does a positional formula defined at index 10000 guarantee reliable prediction there?",
      "options": [
        "No; reliable behavior also depends on training and other assumptions.",
        "Yes; definition guarantees generalization.",
        "Only if the embeddings use sine waves."
      ],
      "answer": 0,
      "explanation": "A valid mathematical representation does not establish a trained model’s accuracy outside its training distribution."
    },
    "deliverable": "A position/geometry laboratory and a documented coordinate convention.",
    "evidence": "Save permutations, coordinates, position settings, sequence lengths, and prediction discrepancies.",
    "readings": [
      {
        "title": "Relative position representations",
        "url": "https://arxiv.org/abs/1803.02155"
      },
      {
        "title": "RoFormer / RoPE",
        "url": "https://arxiv.org/abs/2104.09864"
      }
    ],
    "equationLatex": "\\bigl(R(m\\theta)q\\bigr)^{\\mathsf T}\\bigl(R(n\\theta)k\\bigr)=q^{\\mathsf T}R\\bigl((n-m)\\theta\\bigr)k"
  },
  {
    "id": "09",
    "title": "Training, caching, and real costs",
    "phase": "foundations",
    "duration": "10 min",
    "question": "What changes between learning a model and executing it efficiently?",
    "intuition": "Training stores information for a backward pass and updates weights. Inference keeps weights fixed. During causal generation, old keys and values can be cached because new tokens do not change earlier causal states. That cache saves repeated work, but it is neither learned memory nor a weight update.",
    "essential": [
      "Training choices include initialization, Adam or AdamW optimization, weight decay, learning-rate schedules, warm-up, clipping, effective batch size, and checkpoint selection. They must be recorded with the architecture and data split. Validation selects settings; test data estimates final performance.",
      "Inference converts logits into a selected token through greedy choice or a probability distribution adjusted by temperature. Stopping rules matter. Better held-out loss and better task-specific generation are related but not identical outcomes.",
      "Dense attention stores a score array proportional to \\(BhN^2\\) and performs pairwise work proportional to roughly \\(N^2D\\). Projection and feed-forward costs also matter. FlashAttention changes tiling and memory movement for exact dense attention; it does not make dense pairwise arithmetic linear."
    ],
    "technical": [
      "For a conventional cache, \\(\\text{bytes}=2LB\\,N_{\\mathrm{cached}}\\,h_{\\mathrm{KV}}\\,d_{\\mathrm{head}}\\times\\text{bytes per element}\\). The leading 2 counts keys and values. Multi-query and grouped-query attention alter the number of KV heads and therefore cache storage.",
      "Cached decoding uses unequal query/key lengths. A new query at absolute position offset+i may attend through that absolute position. A fresh square triangle is insufficient; check the library’s non-square alignment, one-token continuation, chunked continuation, and positional offsets.",
      "Cache reuse assumes fixed parameters, unchanged prefix representations, deterministic evaluation, and compatible access and positions. Changing a prefix token, weights, or positional conventions invalidates that assumption. RoPE positions continue from the existing offset.",
      "Mixed precision may require gradient scaling and different tolerances. Save weights, optimizer, scheduler, random state, configuration, and provenance to restore a claimed training computation. Reproducible shuffling is distinct from an independent training seed."
    ],
    "research": [
      "Profile reference attention versus an optimized exact kernel, local/sparse interaction patterns, and approximations as separate conditions. Record prefill and incremental decoding independently. Wall-clock speed depends on hardware, shapes, dtype, and implementation as well as asymptotic arithmetic."
    ],
    "equation": "cache bytes = 2 × L × B × N_cached × h_KV × d_head × bytes_per_element",
    "shapes": [
      "Dense scores: \\([B, h, N, N]\\)",
      "New-token queries: \\([B, h, N_{\\mathrm{new}}, d_h]\\)",
      "Cached keys/values: \\([B, h_{\\mathrm{KV}}, N_{\\mathrm{cached}}, d_h]\\)",
      "Continuation scores: \\([B, h, N_{\\mathrm{new}}, N_{\\mathrm{cached}} + N_{\\mathrm{new}}]\\)"
    ],
    "numerical": "For 2 layers, batch 1, 100 cached tokens, 2 KV heads, head width 4, and 4 bytes per element, the key/value cache stores \\(2\\times2\\times1\\times100\\times2\\times4\\times4=12{,}800\\) bytes. Doubling token count doubles this cache but quadruples a dense square score matrix.",
    "exercise": "Compare cached and full-prefix logits with dropout disabled for one-token and multi-token continuations. Profile one training step before increasing sequence length.",
    "failure": "Restart positional indices at zero after caching a prefix, or use a wrongly aligned causal mask. The cached computation then disagrees with full-prefix evaluation.",
    "check": {
      "question": "What does FlashAttention primarily change for dense attention?",
      "options": [
        "Memory movement and tiling while preserving the dense operation within numerical precision.",
        "The number of pairwise interactions becomes linear.",
        "It removes \\(Q\\)/\\(K\\)/\\(V\\) projections."
      ],
      "answer": 0,
      "explanation": "Efficient exact attention avoids some materialization and memory traffic; it retains dense pairwise arithmetic."
    },
    "deliverable": "A training/inference profile and cache-correctness checks.",
    "evidence": "Save hardware, runtime versions, shapes, dtype, peak memory, prefill/decoding timings, and cached/uncached maximum differences.",
    "lab": "shapes",
    "readings": [
      {
        "title": "AdamW",
        "url": "https://arxiv.org/abs/1711.05101"
      },
      {
        "title": "Grouped-query attention",
        "url": "https://arxiv.org/abs/2305.13245"
      },
      {
        "title": "FlashAttention",
        "url": "https://arxiv.org/abs/2205.14135"
      }
    ],
    "equationLatex": "\\begin{aligned}\\text{cache bytes}&=2LB\\,N_{\\mathrm{cached}}\\,h_{\\mathrm{KV}}\\,d_{\\mathrm{head}}\\\\&\\qquad\\times\\text{bytes per element}\\end{aligned}"
  },
  {
    "id": "10",
    "title": "An image becomes a token sequence",
    "phase": "vision",
    "duration": "9 min",
    "question": "How can the same attention block process pixels?",
    "intuition": "Cut an image into a grid of patches. Flatten each patch into a vector and project it into a shared feature width. Attention now sees a sequence of patch descriptions, plus position information. A patch is a sampling unit; it does not necessarily correspond to one object.",
    "essential": [
      "For a 2D image \\([B,C,H,W]\\) and square patch width \\(P\\), the token count is \\(\\frac HP\\frac WP\\) when both dimensions divide evenly. Each raw patch has \\(CP^2\\) values. Its learned projection creates \\(D\\) features, and \\(D\\) need not equal \\(CP^2\\).",
      "A Vision Transformer (ViT) uses patch features, positions, a transformer encoder, final normalization, and a task head. A learned class token can collect information for classification; mean pooling provides an alternative. Dense tasks retain spatial patch features rather than relying on one pooled vector.",
      "Convolution shares local filters and acquires larger receptive fields through depth. Attention allows content-dependent token interactions. Neither mechanism is automatically superior for microscopy; data, pretraining, scale, and output design matter.",
      "Choose a documented crop/pad policy for non-divisible sizes. Keep valid regions and reconstruction coordinates consistent. Reconstruct the exact image before any learned projection to verify patch and channel order."
    ],
    "technical": [
      "Patchification reorganizes \\([B,C,H,W]\\) into \\([B,N,CP^2]\\). A matrix \\(W_{\\mathrm{patch}}:[CP^2,D]\\) maps patches to \\([B,N,D]\\). A convolution with kernel and stride \\(P\\) can implement the same learned projection under matching weights and ordering.",
      "Halving patch width in 2D quadruples \\(N\\). A full score matrix then grows sixteenfold. Smaller patches retain finer spatial sampling but increase attention cost. Later upsampling cannot guarantee recovery of evidence discarded by a coarse representation."
    ],
    "research": [
      "Compare patch size, position information, class-token versus mean pooling, and data quantity. Inspect earlier vision attention work such as non-local networks and Image Transformer; ViT was not the first use of attention in images. A tiny ViT is an educational model rather than an ImageNet reproduction."
    ],
    "equation": "N = (H/P)(W/P); X_patch ∈ ℝ^[B,N,CP²]; E = X_patch W_patch",
    "shapes": [
      "Image: \\([B, 3, 224, 224]\\)",
      "\\(16\\times16\\) grid: \\([B, 196, 768]\\)",
      "Projected patches: \\([B, 196, D]\\)",
      "With class token: \\([B, 197, D]\\)",
      "Classifier: \\([B, n_{\\mathrm{classes}}]\\)"
    ],
    "numerical": "A \\(224\\times224\\) RGB image with \\(16\\times16\\) patches has \\(14\\times14=196\\) patches and \\(3\\times16^2=768\\) values per patch. At patch width 8, there are 784 patches; \\(\\frac{784^2}{196^2}=16\\) times as many dense attention score entries.",
    "exercise": "Patchify and unpatchify a small labelled checkerboard exactly. Inspect one patch’s pixel order, then add a learned projection and trace a tiny ViT forward pass.",
    "failure": "Restore patch tokens to the wrong grid order, creating a plausible but spatially scrambled dense output. Verify geometry before training.",
    "check": {
      "question": "Is the raw patch length 768 in the \\(16\\times16\\) RGB example necessarily the model width?",
      "options": [
        "No; \\(D\\) is independently chosen by the learned projection.",
        "Yes; every ViT must use \\(D=768\\).",
        "Only with a class token."
      ],
      "answer": 0,
      "explanation": "Raw patch length counts pixel values. Model width counts learned representation features."
    },
    "deliverable": "A patch inspector, exact reconstruction check, and tiny ViT plan.",
    "evidence": "Save image dimensions, patch size, crop/pad policy, patch order, reconstruction error, and token/score counts.",
    "lab": "patches",
    "readings": [
      {
        "title": "Vision Transformer",
        "url": "https://arxiv.org/abs/2010.11929"
      },
      {
        "title": "Non-local Neural Networks",
        "url": "https://arxiv.org/abs/1711.07971"
      },
      {
        "title": "Image Transformer",
        "url": "https://arxiv.org/abs/1802.05751"
      }
    ],
    "equationLatex": "\\begin{aligned}N&=\\frac HP\\frac WP\\\\X_{\\mathrm{patch}}&\\in\\mathbb R^{B\\times N\\times CP^2}\\\\E&=X_{\\mathrm{patch}}W_{\\mathrm{patch}}\\end{aligned}"
  },
  {
    "id": "11",
    "title": "Hierarchy, windows, and pretraining",
    "phase": "vision",
    "duration": "8 min",
    "question": "Which interventions improve practicality, and which change the learning problem?",
    "intuition": "A stronger result may come from different communication, better supervision, more data, or a better starting representation. Treat these as separate levers. A hierarchical model changes spatial scales; a self-supervised objective changes how its weights are learned.",
    "essential": [
      "DeiT studies data-efficient training and distillation, including a distillation token. Swin restricts attention to windows, shifts window boundaries between blocks, and merges patches to form multiscale features. DINO learns representations through teacher–student self-supervision. MAE learns masked-patch reconstruction with an asymmetric encoder–decoder.",
      "Shifted windows let neighboring regions exchange information across earlier window boundaries. Boundary masks prevent the shift from introducing unintended wraparound interactions. Patch merging reduces spatial resolution and changes channel width.",
      "In DINO, augmented views train a student against a teacher updated by a moving average, with design choices that avoid collapse. Original DINO, DINOv2, and DINOv3 are distinct versions. MAE encodes visible patches and reconstructs masked positions with a decoder; reconstructed pixels are not cell instance masks."
    ],
    "technical": [
      "If window size stays fixed, the attention interaction count scales approximately linearly with total image tokens, since each query sees a bounded local window. Projection, merging, feed-forward, and other model costs remain.",
      "A feature pyramid contains several resolutions. Restore the geometry of each feature map before fusion. When inspecting pretrained models, record exact checkpoint, image preprocessing, feature extraction stage, and whether weights are frozen, partly tuned, or fully tuned.",
      "In the equation, \\(n_{\\mathrm{window}}\\) is the number of tokens in each fixed-size window."
    ],
    "research": [
      "Separate architecture, objective, augmentation, data scale, initialization, and trainable budget. Compare frozen features with a small head, partial fine-tuning, and a randomly initialized small model as distinct conditions. Pretraining overlap with evaluation data should be documented, and unknown overlap should stay marked unknown."
    ],
    "equation": "global pairs ≈ N²; fixed-window pairs ≈ N × tokens_per_window",
    "shapes": [
      "Initial patch grid: \\([B, H_p, W_p, D]\\)",
      "Merged grid: \\([B, H_p/2, W_p/2, D_{\\mathrm{next}}]\\)",
      "Window attention: \\([B\\times n_{\\mathrm{windows}}, h, N_{\\mathrm{window}}, N_{\\mathrm{window}}]\\)",
      "MAE visible tokens: \\([B, N_{\\mathrm{visible}}, D]\\)"
    ],
    "numerical": "For 64 tokens in windows of 16, there are \\(4\\times16^2=1024\\) within-window score entries per head, compared with \\(64^2=4096\\) global entries. Shifting windows changes communication, not this basic fixed-window count.",
    "exercise": "Draw \\(4\\times4\\) windows on an \\(8\\times8\\) grid, shift the partition, and identify which tokens become connected. Compare frozen versus trainable feature extraction in a small downstream plan.",
    "failure": "Call a pretrained model’s improvement an architecture effect while its comparison starts randomly and sees much less data.",
    "check": {
      "question": "Which pairing is correct?",
      "options": [
        "Swin changes hierarchical/local communication; MAE changes the pretraining task.",
        "MAE directly produces instance labels.",
        "DeiT and DINO are the same objective."
      ],
      "answer": 0,
      "explanation": "These interventions act on different parts of the pipeline and should be evaluated separately."
    },
    "deliverable": "A window-attention exercise and a carefully labelled backbone comparison.",
    "evidence": "Save checkpoints, version, preprocessing, frozen/trainable parameters, feature map shapes, label budget, and runtime.",
    "lab": "patches",
    "readings": [
      {
        "title": "DeiT",
        "url": "https://arxiv.org/abs/2012.12877"
      },
      {
        "title": "Swin Transformer",
        "url": "https://arxiv.org/abs/2103.14030"
      },
      {
        "title": "DINO",
        "url": "https://arxiv.org/abs/2104.14294"
      },
      {
        "title": "DINOv2",
        "url": "https://arxiv.org/abs/2304.07193"
      },
      {
        "title": "DINOv3",
        "url": "https://arxiv.org/abs/2508.10104"
      },
      {
        "title": "Masked Autoencoders",
        "url": "https://arxiv.org/abs/2111.06377"
      }
    ],
    "equationLatex": "\\begin{aligned}\\text{global pairs}&\\approx N^2\\\\\\text{fixed-window pairs}&\\approx N\\,n_{\\mathrm{window}}\\end{aligned}"
  },
  {
    "id": "12",
    "title": "Separate touching cell instances",
    "phase": "vision",
    "duration": "10 min",
    "question": "What must segmentation preserve beyond foreground overlap?",
    "intuition": "Two touching disks can form one connected foreground region. A perfect foreground mask can therefore still give the wrong number of objects. Instance segmentation needs a separating representation and a reconstruction rule, not merely a foreground probability.",
    "essential": [
      "Build baselines in order: thresholding and connected components; distance-transform markers and watershed; a small U-Net predicting foreground and a separating target; then one compatible pretrained cell-instance model. Start with exact synthetic labels before inspecting real nuclei, cells, or organoids.",
      "Thresholding depends on intensity and background variation. Connected components group connected foreground. A distance transform measures distance to background; marker-controlled watershed grows regions from seeds. Missing markers merge objects, and excessive markers can split one object.",
      "Derive foreground, boundary, center, or per-instance distance targets from annotated masks. A distance inside each reference instance differs from distance inside the foreground union. Document the target because it changes what touching objects teach the model.",
      "StarDist predicts radial shape proposals and suppresses overlaps. Cellpose predicts within-object flows used to reconstruct masks. These flows are spatial reconstruction fields, not physical movement between frames. Shape assumptions can fail on branched or overlapping structures."
    ],
    "technical": [
      "Pixel cross-entropy supervises class probabilities; Dice-style losses supervise overlap; distances and flows use regression objectives. Final instance quality is evaluated after reconstruction, using detections, merges, splits, and boundary/overlap criteria.",
      "Instance IDs are categorical region identifiers. A relabelling from IDs \\([1,2]\\) to \\([7,99]\\) must leave geometry and metrics unchanged. Do not regress scalar label values or interpolate IDs with bilinear resizing. Apply identical geometric transforms to images and labels, using label-safe nearest-neighbor sampling for labels."
    ],
    "research": [
      "Hold foreground fixed while changing markers or boundaries to isolate reconstruction. Then hold reconstruction fixed while changing image features. A model can improve pixel overlap while worsening instance count; inspect an error gallery and distinguish those endpoints."
    ],
    "equation": "image → foreground + separating representation → instance reconstruction → labelled objects",
    "shapes": [
      "Image: \\([B, C, H, W]\\)",
      "Foreground/boundary targets: \\([B, 2, H, W]\\)",
      "Flow example: \\([B, 2, H, W]\\)",
      "Instance labels: \\([B, H, W]\\) discrete IDs"
    ],
    "numerical": "Two touching disks produce one connected component. Watershed with two correct markers can produce two instances; one marker produces one region, while three markers may create three. All three conditions can share exactly the same foreground.",
    "exercise": "Create touching disks, keep their union unchanged, and vary marker placement. Count merges and splits separately from foreground overlap.",
    "failure": "Report high Dice as proof of accurate cell counts, or interpret a reconstructed flow field as a cell velocity.",
    "check": {
      "question": "Can foreground overlap be perfect while the instance count is wrong?",
      "options": [
        "Yes; touching objects can remain merged.",
        "No; perfect overlap guarantees correct identities.",
        "Only for 3D images."
      ],
      "answer": 0,
      "explanation": "Foreground describes the union of objects. It does not uniquely determine how that union is partitioned into instances."
    },
    "deliverable": "Classical/U-Net baseline plan, common reconstruction interface, and labelled error gallery.",
    "evidence": "Save image/target definitions, marker settings, reconstruction parameters, relabelling invariance, and merge/split examples.",
    "lab": "segmentation",
    "readings": [
      {
        "title": "Watershed tutorial",
        "url": "https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html"
      },
      {
        "title": "U-Net",
        "url": "https://arxiv.org/abs/1505.04597"
      },
      {
        "title": "StarDist",
        "url": "https://arxiv.org/abs/1806.03535"
      },
      {
        "title": "Cellpose",
        "url": "https://doi.org/10.1038/s41592-020-01018-x"
      },
      {
        "title": "Omnipose",
        "url": "https://doi.org/10.1038/s41592-022-01639-4"
      }
    ],
    "equationLatex": "\\begin{gathered}\\text{image}\\to\\text{foreground + separating representation}\\\\\\to\\text{instance reconstruction}\\to\\text{labelled objects}\\end{gathered}"
  },
  {
    "id": "13",
    "title": "From patch features to dense labels",
    "phase": "vision",
    "duration": "7 min",
    "question": "How do transformer features become a decision at every pixel?",
    "intuition": "An image encoder produces features at patch locations. A dense decoder restores a spatial grid, combines scales, and predicts pixel-level targets. Upsampling makes a map larger; it does not independently recover boundary evidence that the representation failed to preserve.",
    "essential": [
      "Start with \\([B,N,D]\\), remove or separately handle non-spatial tokens, restore the patch grid, and apply a dense head. A linear head, interpolation, convolutional refinement, and multiscale fusion are different ways to turn coarse features into output maps.",
      "Keep backbone, prediction heads, and instance reconstruction separate. The backbone computes features. Heads predict foreground, boundaries, distances, masks, or other targets. Reconstruction converts those predictions into separate object instances.",
      "SegFormer combines a hierarchical transformer encoder with a lightweight multiscale MLP decoder. Semantic labels assign categories to pixels; they do not automatically separate touching cells into individual identities."
    ],
    "technical": [
      "For a patch grid \\(H_p\\times W_p\\), \\(N\\) must match \\(H_pW_p\\) after non-spatial tokens are handled. Permute features into \\([B,D,H_p,W_p]\\) for conventional dense operations, and track each resolution change through the decoder.",
      "Bilinear interpolation uses nearby values to populate a finer grid. It cannot uniquely reconstruct high-frequency details absent from the coarse map. Skip features or multiscale evidence can supply information that plain resizing lacks."
    ],
    "research": [
      "Compare U-Net and a small transformer using the same targets, reconstruction, labels, and evaluation. Report pretraining, trainable parameters, crop size, augmentations, runtime, and memory. A pretrained transformer versus a randomly initialized CNN is a practical comparison, but it does not isolate the backbone architecture."
    ],
    "equation": "[B,N,D] → [B,D,H_p,W_p] → dense target maps → instance reconstruction",
    "shapes": [
      "Patch features: \\([B, N, D]\\)",
      "Spatial grid: \\([B, D, H_p, W_p]\\)",
      "Foreground/boundary logits: \\([B, 2, H, W]\\)",
      "Reconstructed labels: \\([B, H, W]\\)"
    ],
    "numerical": "An \\(8\\times8\\) patch grid has only 64 spatial feature locations. Resizing its output to \\(64\\times64\\) creates 4096 values by interpolation, but does not create 4096 independent observations of the boundary.",
    "exercise": "Trace every resolution from a patch encoder to full-size foreground and boundary maps. Use the same watershed reconstruction as the U-Net baseline.",
    "failure": "Feed a class token into a spatial reshape, or compare different instance reconstruction settings and attribute the resulting difference to the encoder.",
    "check": {
      "question": "Which stage explicitly creates separate instances in a foreground/boundary pipeline?",
      "options": [
        "The reconstruction procedure.",
        "The name of the backbone.",
        "Bilinear resizing alone."
      ],
      "answer": 0,
      "explanation": "Foreground and boundary maps are intermediate predictions. The instance reconstruction converts them into individual object regions."
    },
    "deliverable": "A shared-head dense segmentation comparison plan with explicit resolution traces.",
    "evidence": "Save head outputs, reconstruction settings, feature maps, pretraining condition, and instance metrics.",
    "lab": "patches",
    "readings": [
      {
        "title": "SegFormer",
        "url": "https://arxiv.org/abs/2105.15203"
      }
    ],
    "equationLatex": "\\begin{gathered}[B,N,D]\\to[B,D,H_p,W_p]\\\\\\to\\text{dense target maps}\\to\\text{instance reconstruction}\\end{gathered}"
  },
  {
    "id": "14",
    "title": "Object queries and set prediction",
    "phase": "vision",
    "duration": "9 min",
    "question": "How can a fixed number of slots predict an unordered, variable-sized object set?",
    "intuition": "An object query is a learned slot that reads image features. Several slots can propose objects; unused slots learn “no object.” Since the reference objects have no meaningful list order, training must find a sensible correspondence before applying per-object losses.",
    "essential": [
      "DETR uses learned object queries and cross-attention to image features. Each slot predicts a class and box. Extra slots receive no-object supervision. The number of query slots sets a capacity limit on independently represented objects.",
      "A bipartite matching step pairs predictions with reference objects using a chosen cost. Matched predictions receive class and localization losses; unmatched predictions receive no-object supervision. Matching cost and final training loss are related design choices, not the same object.",
      "DETR’s within-image training assignment differs from tracking assignment across frames. A fixed query index is not automatically a persistent biological identity. The original DETR avoids a conventional non-maximum suppression stage, but this does not describe every transformer detector."
    ],
    "technical": [
      "Matching searches one-to-one assignments minimizing a combined class/box cost. Box losses can include coordinate distance and generalized intersection over union. The discrete matching selects correspondences, after which differentiable losses train the matched predictions.",
      "Deformable DETR samples around reference locations and uses multiscale features. It changes the attention sampling pattern; it is not merely the original dense operation applied to a smaller image.",
      "In the matching equation, \\(r_i\\) is reference object \\(i\\), \\(\\hat y_j\\) is prediction \\(j\\), \\(c\\) is their matching cost, and \\(\\mathcal A\\) is the set of valid one-to-one assignments."
    ],
    "research": [
      "Use synthetic shapes to inspect duplicate predictions, insufficient query capacity, and missing objects. Vary one factor while preserving matching/loss definitions. A variable number of predicted objects does not imply an unlimited-capacity output set."
    ],
    "equation": "σ* = argmin_assignment Σᵢ matching_cost(referenceᵢ, prediction_σ(i))",
    "shapes": [
      "Image features: \\([B, N_{\\mathrm{image}}, D]\\)",
      "Object queries: \\([B, K_{\\mathrm{queries}}, D]\\)",
      "Cross-attention scores: \\([B, h, K_{\\mathrm{queries}}, N_{\\mathrm{image}}]\\)",
      "Class outputs: \\([B, K_{\\mathrm{queries}}, n_{\\mathrm{classes}}+1]\\)",
      "Boxes: \\([B, K_{\\mathrm{queries}}, 4]\\)"
    ],
    "numerical": "For two references A,B and three predictions, let their costs be \\(A:[1,4,5]\\) and \\(B:[3,1,2]\\). Matching \\(A\\to1\\) and \\(B\\to2\\) costs 2. Prediction 3 is unmatched and receives no-object supervision, even though it was a possible match.",
    "exercise": "Enumerate assignments for three predictions and two references. Compute matching, then separately calculate matched class/box and unmatched no-object losses.",
    "failure": "Use only two slots for a scene with three distinct objects and blame the decoder for the unavoidable missing object.",
    "check": {
      "question": "Does a DETR slot index automatically identify the same cell in the next frame?",
      "options": [
        "No; persistent identity requires a temporal mechanism or association.",
        "Yes; the query is learned.",
        "Only when no-object slots are present."
      ],
      "answer": 0,
      "explanation": "Within-frame set assignment does not establish temporal identity."
    },
    "deliverable": "A tiny set-assignment calculation and a synthetic query-capacity experiment.",
    "evidence": "Save cost matrices, selected matches, unmatched slots, losses, query count, and duplicates/misses.",
    "readings": [
      {
        "title": "DETR",
        "url": "https://arxiv.org/abs/2005.12872"
      },
      {
        "title": "Deformable DETR",
        "url": "https://arxiv.org/abs/2010.04159"
      }
    ],
    "equationLatex": "\\sigma^*=\\underset{\\sigma\\in\\mathcal A}{\\operatorname{arg\\,min}}\\sum_i c\\bigl(r_i,\\hat y_{\\sigma(i)}\\bigr)"
  },
  {
    "id": "15",
    "title": "Mask queries, prompts, and cell models",
    "phase": "vision",
    "duration": "10 min",
    "question": "How can a query specify an object mask, and what does a prompt contribute?",
    "intuition": "A mask query combines an object hypothesis with spatial features. A prompt supplies additional information about the desired region. If a perfect reference box tells the system where an object is, the resulting score answers a different question from fully automatic detection and segmentation.",
    "essential": [
      "Mask classification predicts a class and spatial mask for each query. Semantic segmentation assigns categories, instance segmentation separates objects, and panoptic segmentation combines instance-aware and category-level regions. Mask2Former uses multiscale features and cross-attention restricted by predicted mask regions.",
      "Set matching can use class and mask costs, followed by matched class and mask losses such as binary cross-entropy and Dice. Point sampling for efficient matching/loss computation is separate from mask-restricted attention.",
      "SAM has an image encoder, prompt encoder, and mask decoder. Points or boxes condition mask candidates and quality predictions. Automatic mask generation wraps prompting with proposal and filtering procedures.",
      "Keep families distinct: SAM supplies general prompt-conditioned masks; SAM 2 introduces temporal memory for image/video segmentation; CellSAM combines CellFinder-generated prompts with SAM-based masks; Cellpose-SAM adapts a SAM backbone within Cellpose’s prediction/reconstruction framework. DINO-based Cellpose variants use a different backbone family. None of these names alone establishes a division-aware lineage model."
    ],
    "technical": [
      "Prompt coordinates must match image resizing, padding, and encoder geometry. Compare reference-derived boxes, deliberately perturbed boxes, and automatically detected boxes as separate conditions. Measure detection quality and conditional mask quality independently.",
      "Record exact checkpoint and software version. The supplied blueprint identifies CellSAM’s December 2025 Nature Methods publication and Cellpose-SAM’s 2025 preprint/released software, with later 2026 checkpoints listed in the ecosystem. Treat that as dated reading context and verify versions when running an implementation."
    ],
    "research": [
      "Start with compatible pretrained inference rather than full model training. Ablate prompt quality, backbone adaptation, annotation budget, and postprocessing separately. A reference-derived prompt is an oracle condition, and excellent oracle masks do not establish a reliable automatic cell pipeline."
    ],
    "equation": "image features + encoded prompt → mask candidates + predicted quality",
    "shapes": [
      "Image features: \\([B, D, H_f, W_f]\\)",
      "Prompt/query tokens: \\([B, K_{\\mathrm{prompt}}, D]\\)",
      "Mask candidates: \\([B, K_{\\mathrm{masks}}, H_{\\mathrm{mask}}, W_{\\mathrm{mask}}]\\)",
      "Quality predictions: \\([B, K_{\\mathrm{masks}}]\\)"
    ],
    "numerical": "If automatic detection finds 8 of 10 cells and produces perfect masks for those 8, conditional mask quality can be excellent while the end-to-end pipeline still misses 20% of the objects.",
    "exercise": "Create a prompt audit with exact, shifted, oversized, missing, and automatically proposed boxes. Inspect which errors originate before the mask decoder.",
    "failure": "Present reference-derived boxes as an automatic pipeline, or interpret stable video masks as a correct parent–daughter lineage.",
    "check": {
      "question": "What does an oracle-box mask experiment isolate?",
      "options": [
        "Mask prediction given privileged localization information.",
        "The complete automatic detection pipeline.",
        "Division-aware tracking performance."
      ],
      "answer": 0,
      "explanation": "The reference box supplies localization that an automatic system would have to infer."
    },
    "deliverable": "A prompt/mask interpretation lab and exact model-inventory record.",
    "evidence": "Save checkpoint, prompt source, coordinates, preprocessing, detection misses, mask errors, and thresholds.",
    "readings": [
      {
        "title": "Mask2Former",
        "url": "https://arxiv.org/abs/2112.01527"
      },
      {
        "title": "SAM",
        "url": "https://arxiv.org/abs/2304.02643"
      },
      {
        "title": "SAM 2",
        "url": "https://arxiv.org/abs/2408.00714"
      },
      {
        "title": "CellSAM",
        "url": "https://doi.org/10.1038/s41592-025-02879-w"
      },
      {
        "title": "Cellpose-SAM",
        "url": "https://doi.org/10.1101/2025.04.28.651001"
      },
      {
        "title": "Cellpose model inventory",
        "url": "https://cellpose.readthedocs.io/en/latest/models.html"
      }
    ],
    "equationLatex": "\\begin{gathered}\\text{image features}+\\text{encoded prompt}\\\\\\to\\text{mask candidates}+\\text{predicted quality}\\end{gathered}"
  },
  {
    "id": "16",
    "title": "Volumes and anisotropic microscopy",
    "phase": "vision",
    "duration": "8 min",
    "question": "What changes when pixels become voxels with unequal physical spacing?",
    "intuition": "A cube in voxel coordinates need not be a cube in physical space. If depth samples are four times farther apart than lateral samples, equal numbers of voxels span very different physical distances. Patch shape, motion features, overlap, and reconstruction should respect that geometry.",
    "essential": [
      "Use \\([B,C,Z,H,W]\\) and patch dimensions \\(P_z,P_y,P_x\\). With divisible axes, token count is \\(\\frac{Z}{P_z}\\frac{H}{P_y}\\frac{W}{P_x}\\). Non-cubic patches can better match physical support or memory constraints.",
      "Sampling spacing differs from optical resolution. Depth-dependent signal and anisotropic blur affect image evidence. Isotropic resampling reorganizes observed data; it cannot recreate detail that was never measured.",
      "Compare slice-wise 2D, neighboring-slice input, and true 3D processing. A maximum-intensity projection can collapse distinct objects along depth. Three-dimensional positions and normalized physical coordinates should preserve the chosen unit convention.",
      "Large volumes require crops or overlapping sliding windows. Reconcile instance predictions using geometry and instance matching/stitching. Averaging arbitrary integer IDs is meaningless; only suitable continuous prediction maps can be blended before reconstruction."
    ],
    "technical": [
      "A patch’s nominal support is patch voxel count multiplied by spacing along each axis. At spacing \\([4,1,1]\\) micrometers, a \\([4,16,16]\\) patch spans approximately \\([16,16,16]\\) micrometers. State whether coordinates refer to voxel centers or another geometry convention.",
      "UNETR and Swin UNETR provide examples of volumetric dense prediction. Their medical semantic segmentation outputs need suitable targets and instance reconstruction to become cell-instance pipelines. Measure memory for a small crop before increasing size."
    ],
    "research": [
      "Begin with exact synthetic ellipsoids and then a small annotated real crop. Evaluate depth continuity, border effects, and stitching errors separately. A successful 2D method does not establish 3D performance, and a semantic volume score does not establish correct cell identities."
    ],
    "equation": "N = (Z/P_z)(H/P_y)(W/P_x); physical support = patch sizes ⊙ voxel spacing",
    "shapes": [
      "Volume: \\([B, C, Z, H, W]\\)",
      "Raw patches: \\([B, N, C\\times P_z\\times P_y\\times P_x]\\)",
      "Tokens: \\([B, N, D]\\)",
      "3D labels: \\([B, Z, H, W]\\)"
    ],
    "numerical": "A \\(128^3\\) volume with \\(16^3\\) patches has \\(8^3=512\\) tokens. With \\(8^3\\) patches it has \\(16^3=4096\\) tokens. Token count rises eightfold, while the dense square score matrix rises \\(8^2=64\\)-fold.",
    "exercise": "Calculate token counts and physical patch supports for an anisotropic volume. Design a crop overlap and instance-stitching policy before running a model.",
    "failure": "Compute physical displacement using voxel indices without spacing, or average label ID 2 with ID 8 and interpret the resulting ID 5 as an object.",
    "check": {
      "question": "Does isotropic resampling recover unobserved axial detail?",
      "options": [
        "No; it interpolates existing observations.",
        "Yes; equal spacing guarantees equal resolution.",
        "Only for transformer inputs."
      ],
      "answer": 0,
      "explanation": "Sampling geometry can be changed computationally, but missing image evidence cannot be recreated by interpolation alone."
    },
    "deliverable": "A volumetric token/memory study and a physically explicit reconstruction plan.",
    "evidence": "Save voxel spacing, patch dimensions, crop/pad policy, overlap, memory, and stitched-instance failures.",
    "lab": "patches",
    "readings": [
      {
        "title": "UNETR",
        "url": "https://arxiv.org/abs/2103.10504"
      },
      {
        "title": "Swin UNETR",
        "url": "https://arxiv.org/abs/2201.01266"
      },
      {
        "title": "CTC 3D datasets",
        "url": "https://celltrackingchallenge.net/3d-datasets/"
      }
    ],
    "equationLatex": "\\begin{aligned}N&=\\frac{Z}{P_z}\\frac{H}{P_y}\\frac{W}{P_x}\\\\\\text{physical support}&=\\text{patch sizes}\\odot\\text{voxel spacing}\\end{aligned}"
  },
  {
    "id": "17",
    "title": "Time as patches, objects, or queries",
    "phase": "vision",
    "duration": "8 min",
    "question": "Which temporal representation and information constraints fit the task?",
    "intuition": "A movie can be represented as all its image patches, as compact detected objects, or as persistent hypotheses carried between frames. Patches retain image evidence, objects reduce token counts, and persistent queries carry state. Each choice loses or retains different information.",
    "essential": [
      "Patch-level video tokens preserve ambiguous boundaries but can be numerous. Detection tokens describe already segmented objects and inherit their errors. Persistent track queries represent ongoing hypotheses, with identity determined by the complete architecture rather than the query name alone.",
      "Tubelets combine space and time into one token. Joint space–time attention considers all tokens together. Divided spatial/temporal attention and local spatiotemporal windows change communication and cost. TimeSformer and Video Swin are useful architectural examples; their original recognition outputs do not directly supply cell masks or lineages.",
      "Online decisions use observations available by decision time. Offline windows may use later frames to resolve ambiguity. Causal masking follows the task’s information constraints; it is not a universal microscopy requirement. Latency can also determine what information is available."
    ],
    "technical": [
      "Choose a video layout \\([B,F,C,H,W]\\) and explicitly state whether an operation moves time into the token axis, batch axis, or leaves it separate. With \\(N\\) spatial patches per frame and \\(F\\) frames, joint dense attention has roughly \\((FN)^2\\) pairs.",
      "Frame index is not elapsed biological time. Irregular sampling and changes in frame rate alter the plausible displacement distribution. Represent time intervals or timestamps where those differences matter."
    ],
    "research": [
      "Compare temporal context lengths under the same online/offline rule and record accuracy, latency, and token budget. A method using future frames should not be presented as a zero-latency online method. Full video-model training is an optional advanced branch."
    ],
    "equation": "joint space–time token pairs ≈ (F × N_spatial)²",
    "shapes": [
      "Video: \\([B, F, C, H, W]\\)",
      "Frame patches: \\([B, F, N_{\\mathrm{spatial}}, D]\\)",
      "Joint tokens: \\([B, F\\times N_{\\mathrm{spatial}}, D]\\)",
      "Detection tokens: \\([B, N_{\\mathrm{window}}, D]\\)"
    ],
    "numerical": "Four frames with 64 patches each give 256 tokens and 65,536 dense pairs per head. Doubling to eight frames gives 512 tokens and 262,144 pairs: four times the pair count.",
    "exercise": "Draw information flow for patch tokens, detection tokens, and persistent queries. Mark the latest frame each decision may read, then calculate the token budget.",
    "failure": "Compare an offline model using future context with an online baseline and attribute every improvement to attention.",
    "check": {
      "question": "When may an offline tracking window use future frames?",
      "options": [
        "When the stated task allows them and the comparison declares that information.",
        "Never; microscopy is always causal.",
        "Whenever a model has a transformer backbone."
      ],
      "answer": 0,
      "explanation": "Permitted information is a task constraint. It must be declared and respected in the comparison."
    },
    "deliverable": "A spatiotemporal representation comparison with explicit information availability.",
    "evidence": "Save frame interval, context window, decision latency, access rule, token counts, and memory.",
    "lab": "tracking",
    "readings": [
      {
        "title": "TimeSformer",
        "url": "https://arxiv.org/abs/2102.05095"
      },
      {
        "title": "Video Swin Transformer",
        "url": "https://arxiv.org/abs/2106.13230"
      }
    ],
    "equationLatex": "\\text{joint space–time pairs}\\approx(FN_{\\mathrm{spatial}})^2"
  },
  {
    "id": "18",
    "title": "Tracking starts with geometry",
    "phase": "microscopy",
    "duration": "9 min",
    "question": "What can explicit candidate rules and assignment solve before adding attention?",
    "intuition": "Two objects can both prefer the same next detection. Greedily choosing one object’s nearest neighbor can block a much better joint assignment. First inspect candidate edges and costs, then choose a decoding rule. A learned score is only one possible ingredient.",
    "essential": [
      "Freeze segmentation to isolate tracking. Begin with exact synthetic detections, then introduce controlled missing, false, merged, and split detections. Build nearest-neighbor linking and one-to-one linear assignment from the same observations.",
      "A cost can combine normalized displacement, mask overlap, and appearance difference. State units, scaling, temporal direction, and spatial gating. An optional motion predictor needs an explicit dynamical assumption rather than an unexplained extrapolation.",
      "Appearance and disappearance require explicit handling. Ordinary one-to-one assignment cannot directly model one mother producing two daughters. Add division events or an event-aware extension instead of hiding them in a nearest-neighbor heuristic.",
      "Jointly optimizing one frame pair differs from globally optimizing a whole movie. Local assignment can be globally wrong when later observations resolve an ambiguity."
    ],
    "technical": [
      "A linear assignment minimizes a sum of selected pair costs under one-to-one constraints. Dummy start/end choices or a declared unmatched policy handle differing object counts. Candidate gating reduces the search space but can exclude the correct link.",
      "Debug in order: is the correct observation present; is its edge eligible; is its score reasonable; does the decoder choose a consistent combination? A better optimizer cannot select an edge absent from the candidate graph.",
      "In the cost equation, \\(d_{\\mathrm{norm}}\\) is normalized displacement and \\(d_{\\mathrm{appearance}}\\) is appearance difference. The coefficients \\(\\alpha,\\beta,\\gamma\\) set the relative contributions."
    ],
    "research": [
      "Use crossing, entry/exit, gap, and division examples. Compare geometry, pairwise learned scores, and contextual scores with detections and decoder fixed. Then reuse scores to compare decoders separately. This identifies which stage creates the observed improvement."
    ],
    "equation": "cost(i,j) = α·normalized_displacement + β·(1−IoU) + γ·appearance_difference",
    "shapes": [
      "Source centroids: \\([N_{\\mathrm{source}}, d_{\\mathrm{spatial}}]\\)",
      "Target centroids: \\([N_{\\mathrm{target}}, d_{\\mathrm{spatial}}]\\)",
      "Pair cost matrix: \\([N_{\\mathrm{source}}, N_{\\mathrm{target}}]\\)",
      "Eligible-pair mask: \\([N_{\\mathrm{source}}, N_{\\mathrm{target}}]\\)"
    ],
    "numerical": "Costs are \\(c(A,1)=1\\), \\(c(A,2)=2\\), \\(c(B,1)=1.1\\), \\(c(B,2)=10\\). Greedy A-first selects \\(A\\to1\\) and \\(B\\to2\\) for total 11. One-to-one assignment selects \\(A\\to2\\) and \\(B\\to1\\) for total 3.1.",
    "exercise": "Calculate the greedy and optimal assignments above. Then add an unmatched cost, a missing detection, and a division, explaining which assumptions must change.",
    "failure": "Use one-to-one linking for every event and interpret an unmatched daughter as evidence against division.",
    "check": {
      "question": "Can a lower-cost assignment recover an excluded correct candidate edge?",
      "options": [
        "No; candidate generation bounds what the decoder can select.",
        "Yes; global optimization invents it.",
        "Only with a pairwise MLP."
      ],
      "answer": 0,
      "explanation": "Optimization chooses among available hypotheses. Missing candidates require changes upstream."
    },
    "deliverable": "Geometry-based association baselines and saved candidate/cost diagnostics.",
    "evidence": "Save detections, physical units, gates, full cost matrices, selected links, unmatched rules, and event errors.",
    "lab": "assignment",
    "readings": [
      {
        "title": "Linear assignment documentation",
        "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html"
      },
      {
        "title": "Conservation Tracking",
        "url": "https://www.cv-foundation.org/openaccess/content_iccv_2013/html/Schiegg_Conservation_Tracking_2013_ICCV_paper.html"
      }
    ],
    "equationLatex": "\\begin{aligned}c(i,j)&=\\alpha\\,d_{\\mathrm{norm}}(i,j)\\\\&\\quad+\\beta\\bigl(1-\\operatorname{IoU}(i,j)\\bigr)\\\\&\\quad+\\gamma\\,d_{\\mathrm{appearance}}(i,j)\\end{aligned}"
  },
  {
    "id": "19",
    "title": "A cell detection becomes a token",
    "phase": "microscopy",
    "duration": "7 min",
    "question": "Which measured features should represent an object to an association model?",
    "intuition": "A cell token is a compact description of a detection: where it is, when it was observed, how large it is, what it looks like, and possibly features from its image crop. Attention operates on these descriptions; it does not automatically see information omitted from them.",
    "essential": [
      "Candidate features include elapsed time, centroid in physical units, area or volume, intensity summaries, shape descriptors, and optional image-encoder features. Select features appropriate to the labelled structure and imaging modality.",
      "Normalize using training data only. Keep intensity preprocessing, coordinate scaling, and feature statistics documented. A feature derived from future frames or unavailable labels can leak information into an online task.",
      "Variable object counts require batching with padding and valid-token masks. Attention masks control information access, eligible-pair masks select plausible associations, and supervision masks identify labelled relationships. Maintain all three separately.",
      "Absolute and relative coordinates, temporal offsets, neighborhood gating, and crop embeddings are design options. Arbitrary ordering within a frame should not imply biological meaning."
    ],
    "technical": [
      "Project the measured feature width F_in to the model width \\(D\\). The object-count axis is distinct from feature width. Preserve mappings from padded token positions to frame-local labels and reference correspondence indices.",
      "A consistent within-frame permutation must permute features, validity masks, labels, and predicted scores together. The physical result should remain equivalent. Padding values must not influence valid predictions."
    ],
    "research": [
      "Ablate coordinates only; coordinates plus shape; and coordinates plus image features. Hold candidate generation, data split, and decoder fixed. Check whether improvements persist when detections are corrupted or imaging intensity changes."
    ],
    "equation": "tokenᵢ = projection([time, physical_centroid, size, intensity, shape, optional_appearance]ᵢ)",
    "shapes": [
      "Measured features: \\([B, N_{\\mathrm{max}}, F_{\\mathrm{in}}]\\)",
      "Projected tokens: \\([B, N_{\\mathrm{max}}, D]\\)",
      "Valid-token mask: \\([B, N_{\\mathrm{max}}]\\)",
      "Reference pair labels: \\([B, N_{\\mathrm{source}}, N_{\\mathrm{target}}]\\)"
    ],
    "numerical": "Two images can report centroids five pixels apart but represent different distances: at \\(0.5\\,\\mathrm{\\mu m/pixel}\\) the displacement is \\(2.5\\,\\mathrm{\\mu m}\\); at \\(2\\,\\mathrm{\\mu m/pixel}\\) it is \\(10\\,\\mathrm{\\mu m}\\). A model using raw pixels may confuse acquisition scale with motion.",
    "exercise": "Build a small detection table and pad two frames of different sizes. Record token-to-instance mappings, fit feature normalization on training examples, and reorder detections to test equivalence.",
    "failure": "Use test-set mean and variance for feature scaling, or lose the instance mapping during sorting and train on incorrect correspondence labels.",
    "check": {
      "question": "What should happen after a consistent within-frame detection permutation?",
      "options": [
        "Outputs should permute correspondingly without inventing biological changes.",
        "The best lineage should change because index is identity.",
        "The feature width should increase."
      ],
      "answer": 0,
      "explanation": "The detection list is a storage order. Physical coordinates and correspondence mappings carry meaning."
    },
    "deliverable": "A detection-token data specification with full feature and mapping traces.",
    "evidence": "Save training-only normalization statistics, feature units, label mappings, masks, and permutation/padding checks.",
    "lab": "tracking",
    "readings": [
      {
        "title": "Trackastra",
        "url": "https://arxiv.org/html/2405.15700v2"
      },
      {
        "title": "Trackastra implementation",
        "url": "https://github.com/weigertlab/trackastra"
      }
    ],
    "equationLatex": "\\begin{aligned}\\mathrm{token}_i&=\\operatorname{projection}(f_i)\\\\f_i&=[\\text{time},\\text{physical centroid},\\text{size},\\\\&\\qquad\\text{intensity},\\text{shape},\\text{optional appearance}]_i\\end{aligned}"
  },
  {
    "id": "20",
    "title": "Learn associations from context",
    "phase": "microscopy",
    "duration": "9 min",
    "question": "Can nearby observations resolve ambiguities that an independent pair model cannot?",
    "intuition": "A pairwise model judges one possible link in isolation. A contextual model can inspect neighboring objects and their motion. That extra information may distinguish a plausible collective movement from an identity swap, but only if the features, labels, and supervision make that distinction learnable.",
    "essential": [
      "Compare a small contextual transformer with geometry and an independent pairwise multilayer perceptron (MLP), using identical detections, splits, and decoder. Begin by overfitting a tiny clean example, then evaluate held-out synthetic movies before using a real sequence.",
      "Define positives before selecting a loss. Can one parent have two children? Can one child have multiple parents? Are missed observations represented? Is the objective binary pair classification, parent selection, or a structured normalization? State negative sampling and class weighting.",
      "Trackastra is a concrete detection-token tracking example with spatial/temporal context and a normalization accommodating division. Its original within-window supervision includes ancestor–descendant relationships, while published linking uses adjacent-frame candidate edges. Gap handling in this learning project is a separate extension.",
      "Internal attention weights are not automatically association probabilities. Inspect the final correspondence head, its normalization, and its target definition separately from attention."
    ],
    "technical": [
      "The contextual encoder produces hidden features for each detection, and a score head converts source/target features into candidate association scores. Pair masks, loss masks, and decoder eligibility must remain aligned with the detection mapping.",
      "Independent binary probabilities can permit multiple positive children; a row-wise single-parent softmax imposes a different competition. Neither alone guarantees all graph constraints. Select a supervision structure consistent with the declared events."
    ],
    "research": [
      "Ablate context length, positional information, appearance features, and controlled detection noise. Keep the decoder fixed when judging association. Pin the Trackastra version/checkpoint because later implementations include variants. Call a simplified model an educational adaptation unless architecture, training, data, and evaluation match the paper protocol."
    ],
    "equation": "contextual features = Transformer(cell tokens); association scores = correspondence_head(source, target)",
    "shapes": [
      "Window tokens: \\([B, N_{\\mathrm{window}}, D]\\)",
      "Contextual features: \\([B, N_{\\mathrm{window}}, D]\\)",
      "Association scores: \\([B, N_{\\mathrm{source}}, N_{\\mathrm{target}}]\\)",
      "Supervision/eligibility masks: \\([B, N_{\\mathrm{source}}, N_{\\mathrm{target}}]\\)"
    ],
    "numerical": "If a mother has two true daughters with target labels \\([1,1]\\), a single-choice row target cannot express both relationships. Independent binary supervision or an explicitly division-aware normalization changes what the model is allowed to learn.",
    "exercise": "Write the positive/negative association definition for continuation and binary division. Compare one pairwise score with a contextual prediction on a controlled crossing movie.",
    "failure": "Use a single-choice target while expecting two daughters, or report high attention weight as a calibrated continuation probability.",
    "check": {
      "question": "What should stay fixed to isolate an association model’s contribution?",
      "options": [
        "Detections, splits, candidate rules, and decoding.",
        "Only the model name.",
        "The random seed, while changing all other stages."
      ],
      "answer": 0,
      "explanation": "Changing upstream observations or downstream decoding confounds which stage caused the result."
    },
    "deliverable": "A small contextual association model plan and supervision specification.",
    "evidence": "Save labels, negative sampling, masks, candidate scores, fixed decoder, clean-overfit outputs, and held-out errors.",
    "lab": "tracking",
    "readings": [
      {
        "title": "Trackastra, ECCV 2024",
        "url": "https://arxiv.org/html/2405.15700v2"
      },
      {
        "title": "Official Trackastra repository",
        "url": "https://github.com/weigertlab/trackastra"
      }
    ],
    "equationLatex": "\\begin{aligned}H&=\\operatorname{Transformer}(\\text{cell tokens})\\\\S&=\\operatorname{correspondence\\_head}(H_{\\mathrm{source}},H_{\\mathrm{target}})\\end{aligned}"
  },
  {
    "id": "21",
    "title": "Scores become valid lineages",
    "phase": "microscopy",
    "duration": "10 min",
    "question": "Why are high pair scores insufficient for a biologically consistent graph?",
    "intuition": "Several attractive links can contradict one another. A child may be claimed by two parents, or a segmentation split may resemble a division. A decoder selects a compatible combination of events under a declared lineage model. An optimum is the best permitted explanation under its costs, not proof of biological truth.",
    "essential": [
      "Create detection nodes and time-forward candidates for continuation, division, appearance, disappearance, and optional gaps. In an ordinary non-fusing lineage, a selected detection has at most one parent. A continuation has one successor, while a permitted binary division has two daughters.",
      "Tie event choices to detection selection. Every selected detection needs an incoming continuation/gap or appearance, and an outgoing termination, continuation/gap, or permitted division. Explicit incompatible-hypothesis constraints prevent contradictory selections.",
      "Linear assignment handles a restricted pairwise one-to-one problem. Flow formulations encode compatible conservation structures. An integer linear program (ILP) can express richer discrete event and hypothesis constraints. These are different modeling choices, not interchangeable optimizer names.",
      "Compare constrained greedy, continuation-only assignment, and event-aware global decoding using the same scores. This isolates the contribution of event constraints and optimization."
    ],
    "technical": [
      "Use binary selection variables and minimize a sum of event costs subject to consistency constraints. Binary division is an initial modeling assumption, not a universal biological limit. Gap edges also require explicit temporal and export conventions.",
      "A network score is not automatically a calibrated probability. A negative-log probability cost requires a stated probabilistic interpretation or factorization. Independently calibrated pair scores do not automatically define a coherent joint distribution over lineages.",
      "Candidate generation sets the decoder’s hypothesis space. Missing correct edges, merged detections, or absent separate-mask hypotheses cannot be repaired by minimizing the same graph more precisely. Convert internal gaps and segments into the selected evaluator’s conventions before scoring."
    ],
    "research": [
      "Inspect two possible parents, false split versus division, a missing frame, an excluded correct candidate, and conflicting event selections. Vary calibration and event penalties with association scores frozen. Report sensitivity and failure cases, not merely the objective value."
    ],
    "equation": "minimize Σₑ cₑ xₑ, with xₑ ∈ {0,1} and declared lineage-consistency constraints",
    "shapes": [
      "Candidate graph: \\(N_{\\mathrm{detection}}\\) nodes, \\(E_{\\mathrm{event}}\\) hypotheses",
      "Event costs: \\([E_{\\mathrm{event}}]\\)",
      "Selection variables: \\([E_{\\mathrm{event}}]\\) binary",
      "Export: track segments + explicit parent relationships"
    ],
    "numerical": "Suppose edges \\(A\\to C\\) and \\(B\\to C\\) score 0.9 and 0.8. Selecting both violates the one-parent rule even though both scores are high. A consistent decoder must reject at least one or select another event explanation.",
    "exercise": "Write incoming/outgoing constraints for a three-frame toy graph. Add a binary division and a missing-frame gap, then inspect which event combinations are valid.",
    "failure": "Interpret two predicted children as a true division without testing a false segmentation split, or treat the lowest objective as biological ground truth.",
    "check": {
      "question": "What does a globally optimal decoder guarantee?",
      "options": [
        "The best allowed solution under the specified costs and constraints.",
        "The true biological lineage regardless of missing candidates.",
        "Calibrated probabilities for every event."
      ],
      "answer": 0,
      "explanation": "Optimality is conditional on the model, costs, and candidate set. Biological correctness requires independent evidence and evaluation."
    },
    "deliverable": "A division-aware graph specification and graph-invariant checks.",
    "evidence": "Save candidates, scores, event penalties, constraints, selected graph, excluded alternatives, and export conventions.",
    "lab": "tracking",
    "readings": [
      {
        "title": "CBIA: ILP and probability scores",
        "url": "https://www.muni.cz/en/research/publications/2385779"
      },
      {
        "title": "Graph-based tracking evaluation",
        "url": "https://doi.org/10.1371/journal.pone.0144959"
      }
    ],
    "equationLatex": "\\begin{gathered}\\underset{x}{\\operatorname{minimize}}\\quad\\sum_e c_e x_e\\\\x_e\\in\\{0,1\\},\\quad\\text{subject to declared lineage constraints}\\end{gathered}"
  },
  {
    "id": "22",
    "title": "Let temporal evidence choose masks",
    "phase": "microscopy",
    "duration": "8 min",
    "question": "Must every segmentation ambiguity be resolved before tracking begins?",
    "intuition": "An ambiguous bright region may be one object or two touching objects. Rather than committing immediately, retain both interpretations. Later motion can help choose among them—provided the correct interpretation exists in the hypothesis set.",
    "essential": [
      "Joint hypothesis selection supplies alternative segmentations, including merged and separated candidates, then chooses compatible masks and associations together. Ultrack uses optimization over segmentation hypotheses and associations; it is not itself a transformer backbone.",
      "Joint learned segmentation/tracking uses image features and temporal state within a learned architecture. Cell-TRACTR studies persistent track queries and joint outputs. This differs from linking tokens extracted from already finalized masks.",
      "Keep these two directions separate. An optimizer selecting candidate masks and a network carrying persistent queries may solve related problems with different inputs, costs, representations, and failure modes.",
      "The supplied Cell-TRACTR reading concerns 2D bacterial and mammalian imaging. Broader 3D applicability requires evidence rather than extrapolation from its name."
    ],
    "technical": [
      "Alternative masks may overlap or represent incompatible partitions. Add exclusion constraints so the selected combination is a valid interpretation. Association evidence then changes the total cost of each mask hypothesis.",
      "Temporal evidence cannot recover an absent interpretation unless the system can generate new candidates. Candidate coverage is therefore a separate endpoint from score quality and decoder quality. Persistent queries also require an explicit policy for births, terminations, and divisions."
    ],
    "research": [
      "Begin with a tiny candidate-selection example and optionally inspect a compatible published joint model. Vary candidate coverage, association scores, and constraints separately. Full joint-model training is an advanced extension, after reliable baseline segmentation and tracking."
    ],
    "equation": "alternative masks + association/event hypotheses → compatible selected segmentation and lineage",
    "shapes": [
      "Candidate masks: \\([K_{\\mathrm{candidates}}, H, W]\\)",
      "Incompatibility pairs: \\([K_{\\mathrm{conflicts}}, 2]\\)",
      "Candidate association/event costs: \\([E]\\)",
      "Selected mask/event variables: binary arrays"
    ],
    "numerical": "Let the one-object interpretation cost 1 and the two-object interpretation cost 2 from image evidence. If temporal costs are 5 and 1, totals are 6 and 3, so the joint system selects two objects. If the two-object candidate is absent, that conclusion is unavailable.",
    "exercise": "Create two incompatible mask explanations for a touching region. Add later-frame links and calculate when temporal evidence changes the selected interpretation.",
    "failure": "Attribute failure to the optimizer when every supplied candidate merges the true objects, or assume temporal query persistence automatically handles division.",
    "check": {
      "question": "Which limitation persists even with perfect candidate selection?",
      "options": [
        "The true interpretation may be absent from the candidate set.",
        "All masks become physically correct.",
        "Every hypothesis must receive equal cost."
      ],
      "answer": 0,
      "explanation": "Selection chooses available hypotheses; absent geometry requires better candidate generation or new predictions."
    },
    "deliverable": "A toy joint-hypothesis experiment and a comparison with persistent-query tracking.",
    "evidence": "Save alternative masks, exclusions, image/temporal costs, selected hypotheses, and candidate-coverage failures.",
    "lab": "tracking",
    "readings": [
      {
        "title": "Ultrack",
        "url": "https://doi.org/10.1038/s41592-025-02778-0"
      },
      {
        "title": "Cell-TRACTR",
        "url": "https://doi.org/10.1371/journal.pcbi.1013071"
      }
    ],
    "equationLatex": "\\begin{gathered}\\text{alternative masks}+\\text{association/event hypotheses}\\\\\\to\\text{compatible selected segmentation and lineage}\\end{gathered}"
  },
  {
    "id": "23",
    "title": "Evaluate the biological task",
    "phase": "research",
    "duration": "11 min",
    "question": "Which evidence makes a pipeline useful and trustworthy?",
    "intuition": "Correct-looking masks can still swap identities. A high average overlap can hide missed divisions. Evaluation should ask whether geometry, instances, associations, lineages, and biological measurements are each reliable enough for their intended use.",
    "essential": [
      "Assess foreground overlap and boundaries; instance misses, false objects, merges, and splits; association errors; complete tracks and track fractions; branching events; cell-cycle lengths; and practical correction effort, runtime, and memory. No single endpoint covers them all.",
      "Use official Cell Tracking Challenge implementations for named metrics. SEG concerns reference-instance mask agreement using Jaccard/IoU and CTC matching. DET is based on normalized object-detection graph-edit cost. TRA includes detection and temporal graph edits. Do not rename a simplified browser overlap or link count as an official CTC score.",
      "Declare division matching rules and temporal tolerance before reporting precision/recall. A shifted division time may have different consequences for lineage topology and cycle-length measurement. Cell-HOTA is an optional complementary diagnostic with its own conventions.",
      "Gold human-derived references, silver algorithm-derived annotations, synthetic truth, and expert disagreement provide different evidence. Annotation fusion can improve supervision without creating independent ground truth."
    ],
    "technical": [
      "\\(\\operatorname{TRA}=1-\\frac{\\min(\\operatorname{AOGM},\\operatorname{AOGM}_0)}{\\operatorname{AOGM}_0}\\), where \\(\\operatorname{AOGM}\\) is a weighted graph-edit cost and \\(\\operatorname{AOGM}_0\\) is the cost of constructing the reference graph from scratch. Handle degenerate cases according to the official evaluator.",
      "Separate boundary ambiguity, association ambiguity, model uncertainty, and domain shift. Attention entropy is not a calibrated error probability. For meaningful probabilities, inspect reliability diagrams and selective-review curves on an independent curated subset.",
      "Split at independent sequence/specimen level. Adjacent frames share objects and acquisition conditions. Thousands of cells from one movie do not become thousands of independent biological replicates. Sparse annotation coverage must not silently turn unannotated cells into background."
    ],
    "research": [
      "Fabricate correct masks with swapped identities, merged objects, false divisions, missing observations, and shifted division timing. Check which metrics respond and which conceal the error. Report paired held-out sequences, actual independent seeds, annotation provenance, domain shift, and representative failures."
    ],
    "equation": "TRA = 1 − min(AOGM, AOGM₀) / AOGM₀",
    "shapes": [
      "Predicted/reference labels: \\([F, H, W]\\) or \\([F, Z, H, W]\\)",
      "Predicted/reference lineage graphs: nodes + edges/events",
      "Sequence-level endpoint table: \\([n_{\\mathrm{independent\\ sequences}}, n_{\\mathrm{metrics}}]\\)"
    ],
    "numerical": "If \\(\\operatorname{AOGM}=20\\) and \\(\\operatorname{AOGM}_0=100\\), \\(\\operatorname{TRA}=1-\\frac{20}{100}=0.8\\). This value depends on the official graph conventions and costs; a generic count of correctly linked pairs is a different metric.",
    "exercise": "Construct one example with perfect per-frame masks but swapped identities. Predict which geometry and tracking metrics change before evaluating it.",
    "failure": "Report pixel accuracy on mostly background images as proof of useful cell tracking, or treat frames from the same movie as independent test specimens.",
    "check": {
      "question": "Which example can preserve perfect masks while damaging tracking?",
      "options": [
        "Swapping two persistent identities across frames.",
        "Changing arbitrary frame-local label numbers consistently.",
        "Changing the display color palette only."
      ],
      "answer": 0,
      "explanation": "Identity switches affect temporal relationships without necessarily changing any per-frame geometry."
    },
    "deliverable": "An evaluation, uncertainty, and data-provenance audit tied to biological measurements.",
    "evidence": "Save official evaluator version, conventions, curated references, per-sequence metrics, fabricated-error checks, runtime, and correction effort.",
    "lab": "tracking",
    "readings": [
      {
        "title": "CTC evaluation methodology",
        "url": "https://celltrackingchallenge.net/evaluation-methodology/"
      },
      {
        "title": "CTC annotations",
        "url": "https://celltrackingchallenge.net/annotations/"
      },
      {
        "title": "Metrics Reloaded",
        "url": "https://doi.org/10.1038/s41592-023-02151-z"
      },
      {
        "title": "Metric pitfalls",
        "url": "https://doi.org/10.1038/s41592-023-02150-0"
      }
    ],
    "equationLatex": "\\operatorname{TRA}=1-\\frac{\\min(\\operatorname{AOGM},\\operatorname{AOGM}_0)}{\\operatorname{AOGM}_0}"
  },
  {
    "id": "24",
    "title": "A bounded research claim",
    "phase": "research",
    "duration": "9 min",
    "question": "What can a complete chain of evidence actually support?",
    "intuition": "A research result is a comparison with a defined question, controls, and limits. The goal is to find what survives testing, including a well-explained negative outcome. A new model name or an appealing visualization does not establish novelty.",
    "essential": [
      "A manageable first question is whether contextual attention improves cell association over geometry and independent-pair baselines when detections become ambiguous or imperfect. Start in 2D and keep the scope bounded.",
      "Separate reference-derived detections, which isolate association, from predicted segmentation, which evaluates the complete pipeline. Compare nearest-neighbor linking, linear assignment, a pairwise MLP, and a small contextual transformer. A published pretrained method is a separate practical condition when compatible.",
      "Keep the decoder fixed to compare association models. Freeze association scores to compare decoders. Only then explore joint mask selection or 3D. Choose a primary endpoint and record secondary endpoints that might reveal tradeoffs.",
      "When resources permit, use at least three independent training seeds for the main learned comparison. Report the actual number and use narrower claims when evidence is limited. Pair outputs on the same held-out independent sequences."
    ],
    "technical": [
      "Predeclare baseline, changed factor, fixed factors, split, annotation budget, checkpoint selection, run budget, and failure that would falsify or narrow the hypothesis. Leave results blank until measured.",
      "The final report should connect a pipeline diagram, mathematical specification, numerical checks, configurations, data provenance, per-sequence/run variation, runtime/memory, and representative errors. Educational adaptation, paper reproduction, and novel method development are different labels.",
      "A reproduction requires the paper’s actual protocol and accessible implementation details. Where only an institutional abstract is available, record that access limit instead of inventing losses, architectures, numerical scores, or ablations."
    ],
    "research": [
      "Carry a narrow, evidence-backed research question: which failure most limits useful complete tracks, which error stems from detection versus association, and which controlled intervention is feasible? The project may conclude that attention showed no benefit under the tested conditions. That can still demonstrate sound experimental reasoning."
    ],
    "equation": "claim = measured comparison + declared controls + uncertainty + stated scope",
    "shapes": [
      "Run registry: \\([\\text{experiment}, \\text{seed}, \\text{configuration}, \\text{checkpoint}]\\)",
      "Paired outputs: \\([\\text{held-out sequence}, \\text{method}, \\text{endpoint}]\\)",
      "Evidence package: configs + manifests + checks + outputs + failure gallery"
    ],
    "numerical": "If a transformer improves mean link accuracy on one movie but worsens complete-track recovery on another, neither endpoint alone supports “better tracking.” Report both, the sequence count, and the conditions that explain the tradeoff.",
    "exercise": "Fill a hypothesis card: question, proposed mechanism, expected observable change, baseline, one changed factor, fixed factors, split, primary/secondary endpoints, falsifying failure, and compute budget. Leave measured result and conclusion empty.",
    "failure": "Treat rerunning one saved seed as a robustness study, or report thousands of dependent cells as independent experimental replication.",
    "check": {
      "question": "Can a carefully controlled negative experiment satisfy the capstone?",
      "options": [
        "Yes; the outcome is a defensible conclusion with evidence and limits.",
        "No; a transformer must beat every baseline.",
        "Only if the result is renamed as a novel method."
      ],
      "answer": 0,
      "explanation": "The milestone is an auditable comparison and clear understanding, not a predetermined positive outcome."
    },
    "deliverable": "A reproducible capstone report and an independently explained research proposal.",
    "evidence": "Save configuration, data manifests, checks, actual seeds, paired outputs, evaluator version, failure gallery, compute measurements, and a limited conclusion.",
    "readings": [
      {
        "title": "CTC: ten years of benchmarking",
        "url": "https://www.nature.com/articles/s41592-023-01879-y"
      },
      {
        "title": "Trackastra",
        "url": "https://arxiv.org/html/2405.15700v2"
      }
    ],
    "equationLatex": "\\begin{gathered}\\text{claim}=\\text{measured comparison}+\\text{declared controls}\\\\+\\text{uncertainty}+\\text{stated scope}\\end{gathered}"
  }
];

window.GLOSSARY = [
  {
    "term": "Activation",
    "definition": "An intermediate value produced during a forward pass. Stored activations can dominate training memory even when parameter count is modest."
  },
  {
    "term": "Association",
    "definition": "A proposed relationship between detections at different times. Its score and the selected lineage edge are separate objects."
  },
  {
    "term": "Attention",
    "definition": "A content-dependent weighted mixture of source values, with weights computed from query–key compatibility and permitted information access."
  },
  {
    "term": "Attention head",
    "definition": "One learned query/key/value projection and attention computation. Several heads are concatenated and projected; no semantic role is guaranteed."
  },
  {
    "term": "Autoregressive",
    "definition": "Predicting an output from an allowed prefix, then feeding selected outputs into subsequent predictions."
  },
  {
    "term": "Backbone",
    "definition": "The feature-extracting part of a model, separate from task heads and postprocessing or graph decoding."
  },
  {
    "term": "Batch",
    "definition": "Examples processed together. It is an independent axis from sequence length, image geometry, and feature width."
  },
  {
    "term": "Bipartite matching",
    "definition": "Selecting one-to-one correspondences between two sets according to a cost. Within-image set matching differs from tracking across time."
  },
  {
    "term": "Broadcasting",
    "definition": "Applying operations across compatible array axes by implicit expansion of singleton dimensions."
  },
  {
    "term": "Causal mask",
    "definition": "An information-access rule preventing a prediction from using future positions in a declared order."
  },
  {
    "term": "Checkpoint",
    "definition": "Saved model state and, for resumable training, optimizer/scheduler/random states, configuration, and provenance."
  },
  {
    "term": "Class token",
    "definition": "A learned token used to gather an image-level representation for classification; it is not a spatial patch or a cell instance."
  },
  {
    "term": "Contextual feature",
    "definition": "A representation updated using permitted surrounding information, distinct from a context-independent embedding lookup."
  },
  {
    "term": "Cross-attention",
    "definition": "Attention whose queries come from a receiving stream while keys and values come from another source stream."
  },
  {
    "term": "Decoder",
    "definition": "An overloaded term: language generation, dense upsampling, query-mask prediction, and graph event selection are different decoding mechanisms."
  },
  {
    "term": "Detection",
    "definition": "An observed object candidate at a particular time, often derived from an instance mask and summarized by measured features."
  },
  {
    "term": "Dice",
    "definition": "An overlap measure related to twice the intersection divided by the summed region sizes; a foreground score does not guarantee correct instance partitioning. For regions \\(A\\) and \\(B\\), \\(\\operatorname{Dice}(A,B)=\\frac{2|A\\cap B|}{|A|+|B|}\\) when the denominator is nonzero."
  },
  {
    "term": "Embedding",
    "definition": "A learned feature vector assigned to an input unit. A token ID selects an embedding row rather than encoding a numerical semantic distance."
  },
  {
    "term": "Entropy",
    "definition": "A measure of distribution spread. High or low attention entropy does not by itself establish prediction error probability. For a discrete distribution, \\(H(p)=-\\sum_i p_i\\log p_i\\), using the convention \\(0\\log0=0\\)."
  },
  {
    "term": "FFN",
    "definition": "Feed-forward network: in a transformer, usually a shared positionwise nonlinear feature transformation, without direct token-to-token mixing."
  },
  {
    "term": "Foreground",
    "definition": "The union of pixels or voxels belonging to target structures. It does not uniquely specify individual objects."
  },
  {
    "term": "Gradient",
    "definition": "A derivative describing how a small change to a variable affects an objective; shared variables receive contributions from all computation paths."
  },
  {
    "term": "Head",
    "definition": "May mean an attention head or a task prediction head. A prediction head converts features into task-specific outputs."
  },
  {
    "term": "ILP",
    "definition": "Integer linear programming: optimization with linear costs/constraints and integer variables, useful for explicit event and hypothesis selection."
  },
  {
    "term": "Instance segmentation",
    "definition": "Assigning distinct regions to individual objects, rather than only labelling foreground or semantic categories."
  },
  {
    "term": "IoU / Jaccard",
    "definition": "Intersection size divided by union size. Named benchmark measures may add specific matching and aggregation conventions. For regions \\(A\\) and \\(B\\), \\(\\operatorname{IoU}(A,B)=\\frac{|A\\cap B|}{|A\\cup B|}\\) when the union is nonempty."
  },
  {
    "term": "KV cache",
    "definition": "Stored keys and values reused during fixed-parameter causal inference. It is an execution optimization rather than learned durable memory."
  },
  {
    "term": "LayerNorm",
    "definition": "Normalization over chosen feature axes of each example/token, using mean, population variance, epsilon, and learned scale/offset."
  },
  {
    "term": "Lineage",
    "definition": "A time-directed object history including continuation and parent–daughter relationships under a declared biological/event model."
  },
  {
    "term": "Logit",
    "definition": "An unrestricted score before probability normalization. A high raw score is not automatically a calibrated probability."
  },
  {
    "term": "Loss mask",
    "definition": "A rule selecting which predictions contribute to training loss. It does not control input information access."
  },
  {
    "term": "Object query",
    "definition": "A learned slot that gathers image information and predicts an object or no-object result. Its index is not automatically a persistent identity."
  },
  {
    "term": "Oracle prompt",
    "definition": "A reference-derived prompt providing privileged localization. It measures conditional mask performance rather than a fully automatic pipeline."
  },
  {
    "term": "Padding mask",
    "definition": "An access rule that prevents artificial padding from contributing to valid predictions. Query output and loss handling may still require separate policies."
  },
  {
    "term": "Patch",
    "definition": "A sampled spatial or spatiotemporal input region projected into features. A patch is not necessarily one semantic object."
  },
  {
    "term": "Permutation equivariance",
    "definition": "Consistent input reordering leads to correspondingly reordered outputs, under assumptions such as absent order-dependent positions/masks."
  },
  {
    "term": "Physical spacing",
    "definition": "The distance between image samples along each axis. It differs from optical resolution and is required for meaningful physical geometry."
  },
  {
    "term": "Pretraining",
    "definition": "Learning initial model parameters on an earlier objective/data source before a downstream task; exact checkpoint provenance matters."
  },
  {
    "term": "Projection",
    "definition": "A learned linear feature transformation. Unlike reshape, it can change representation width and information content."
  },
  {
    "term": "Q / K / V",
    "definition": "Queries and keys define compatibility; values supply mixed content. They are learned projections, with \\(Q\\)/\\(K\\) sharing compatibility width."
  },
  {
    "term": "Residual",
    "definition": "An identity pathway added to a learned transformation, requiring compatible shapes and affecting gradient flow."
  },
  {
    "term": "RMSNorm",
    "definition": "Feature normalization using a root-mean-square scale without mean subtraction. It is not merely LayerNorm with its bias removed."
  },
  {
    "term": "RoPE",
    "definition": "Rotary positional embeddings: position-dependent rotations of \\(Q\\)/\\(K\\) feature pairs that encode relative offsets in their interactions."
  },
  {
    "term": "Self-attention",
    "definition": "Attention where queries, keys, and values derive from the same input stream, subject to access rules."
  },
  {
    "term": "Silver annotation",
    "definition": "Algorithm-derived reference material. Fusion or refinement can be useful but does not guarantee independent ground truth."
  },
  {
    "term": "Softmax",
    "definition": "Exponentiate and normalize scores along a chosen axis to form a distribution. Subtracting the row maximum improves numerical stability."
  },
  {
    "term": "Teacher forcing",
    "definition": "Training conditional predictions with known preceding target tokens, while retaining the intended causal access rule."
  },
  {
    "term": "Token",
    "definition": "A unit represented by a feature vector: a subword, image patch, voxel patch, or cell detection can all be tokens."
  },
  {
    "term": "TRA",
    "definition": "The official Cell Tracking Challenge normalized graph-edit tracking measure, requiring its evaluator and conventions."
  },
  {
    "term": "Watershed",
    "definition": "A region-growing/flooding reconstruction method often driven by markers and distance or boundary maps; marker quality changes instance partitioning."
  }
];

window.ROADMAP_SECTIONS = [
  {
    "id": "foundations",
    "title": "The foundations",
    "range": "00–09",
    "description": "From numerical tools and sequence history to attention, blocks, objectives, position, and efficient execution."
  },
  {
    "id": "vision",
    "title": "From tokens to vision",
    "range": "10–11, 13–15",
    "description": "Patches, multiscale features, dense outputs, object sets, masks, and prompts."
  },
  {
    "id": "microscopy",
    "title": "Cells through space and time",
    "range": "12, 16–22",
    "description": "Separate instances, respect physical geometry, learn associations, and enforce valid events."
  },
  {
    "id": "research",
    "title": "Evidence and research",
    "range": "23–24",
    "description": "Choose appropriate metrics, audit uncertainty, and defend a bounded experimental claim."
  }
];

window.READINESS_GATES = [
  {
    "title": "Architecture",
    "modules": "04–06, 08",
    "evidence": "Derive attention, gradients, masks, positions, and a complete block."
  },
  {
    "title": "Language",
    "modules": "03, 07, 09",
    "evidence": "Explain objectives, teacher forcing, generation, and cached/uncached agreement."
  },
  {
    "title": "Vision",
    "modules": "10–15",
    "evidence": "Trace patch features to class, dense, set, and mask outputs."
  },
  {
    "title": "Microscopy",
    "modules": "00, 12, 16",
    "evidence": "Distinguish biological targets, physical units, and instance reconstruction failures."
  },
  {
    "title": "Tracking",
    "modules": "18–22",
    "evidence": "Separate candidate generation, association scoring, and constrained lineage decoding."
  },
  {
    "title": "Research",
    "modules": "23–24",
    "evidence": "Defend splits, controls, metrics, independent variation, and a limited conclusion."
  }
];

window.CROSS_DOMAIN = [
  {
    "component": "Input unit",
    "language": "Subword or byte",
    "vision": "Spatial or volumetric patch",
    "cells": "Detected cell at a time"
  },
  {
    "component": "Features",
    "language": "Embedding lookup",
    "vision": "Flattened/projected patch",
    "cells": "Geometry, intensity, shape, appearance"
  },
  {
    "component": "Position",
    "language": "Token index",
    "vision": "2D or 3D coordinates",
    "cells": "Physical location and elapsed time"
  },
  {
    "component": "Self-attention",
    "language": "Context among words",
    "vision": "Context among patches",
    "cells": "Context among nearby observations"
  },
  {
    "component": "Cross-attention",
    "language": "Target reads source",
    "vision": "Object queries read image features",
    "cells": "Track/detection queries read another observation stream"
  },
  {
    "component": "Mask",
    "language": "Causality, padding, document access",
    "vision": "Valid patches, windows, regions",
    "cells": "Padding, time access, spatial candidates"
  },
  {
    "component": "Prediction head",
    "language": "Vocabulary or class logits",
    "vision": "Classes, boxes, distances, masks",
    "cells": "Association or event scores"
  },
  {
    "component": "Loss",
    "language": "Token likelihood or masked prediction",
    "vision": "Pixel, mask, box, set supervision",
    "cells": "Pair, parent, event supervision"
  },
  {
    "component": "Decoding",
    "language": "Greedy or sampled tokens",
    "vision": "Masks or object sets",
    "cells": "Consistent tracks and lineages"
  },
  {
    "component": "Main failures",
    "language": "Leakage and wrong predictions",
    "vision": "Coarse boundaries, merges, splits",
    "cells": "Identity switches, false divisions, gaps"
  }
];

window.EXPERIMENTS = [
  {
    "id": "E01",
    "title": "Attention by hand",
    "comparison": "Manual calculations vs tensor operations",
    "evidence": "Matrices, gradients, tolerances",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E02",
    "title": "Causality",
    "comparison": "Future token changed with/without correct mask",
    "evidence": "Earlier outputs and leakage explanation",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E03",
    "title": "Padding",
    "comparison": "Same valid sequence, different padded content",
    "evidence": "Valid predictions and loss",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E04",
    "title": "Block stability",
    "comparison": "One residual/normalization choice changed",
    "evidence": "Activations, gradients, behavior",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E05",
    "title": "Position",
    "comparison": "Content shuffled with/without position consistency",
    "evidence": "Predictions and symmetry",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E06",
    "title": "Cached inference",
    "comparison": "Full-prefix vs incremental/chunked execution",
    "evidence": "Logit differences and position offsets",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E07",
    "title": "Patch size",
    "comparison": "Fixed image, different patch scales",
    "evidence": "Tokens, memory, reconstruction",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E08",
    "title": "Touching instances",
    "comparison": "Shared foreground, changed markers/boundaries",
    "evidence": "Merges, splits, overlap",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E09",
    "title": "Backbone",
    "comparison": "CNN vs small transformer with shared heads",
    "evidence": "Initialization, compute, instance metrics",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E10",
    "title": "Set prediction",
    "comparison": "Query capacity and duplicate cases",
    "evidence": "Assignment and losses",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E11",
    "title": "Prompt quality",
    "comparison": "Oracle vs perturbed vs automatic boxes",
    "evidence": "Detection and mask errors separately",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E12",
    "title": "Association",
    "comparison": "Geometry vs pair MLP vs contextual model",
    "evidence": "Fixed detections/decoder and errors",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E13",
    "title": "Decoder",
    "comparison": "Shared scores, changed decoding",
    "evidence": "Validity, divisions, complete tracks",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E14",
    "title": "Temporal context",
    "comparison": "Context lengths and allowed future information",
    "evidence": "Accuracy, latency, information rules",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E15",
    "title": "Detection corruption",
    "comparison": "Known misses, merges, and splits introduced",
    "evidence": "Robustness and error propagation",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E16",
    "title": "Joint hypotheses",
    "comparison": "Fixed masks vs candidate selection",
    "evidence": "Candidate coverage and lineages",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E17",
    "title": "Limited labels",
    "comparison": "Predeclared annotation budgets",
    "evidence": "Examples, variation, correction effort",
    "status": "Planned",
    "result": ""
  },
  {
    "id": "E18",
    "title": "Domain shift",
    "comparison": "Held-out acquisition/cell conditions",
    "evidence": "Failures and uncertainty behavior",
    "status": "Planned",
    "result": ""
  }
];

window.OPTIONAL_BRANCHES = [
  {
    "title": "Parameter-efficient adaptation",
    "detail": "Compare LoRA low-rank updates with full fine-tuning, counting trainable parameters and optimizer memory.",
    "url": "https://arxiv.org/abs/2106.09685"
  },
  {
    "title": "Vision–language learning",
    "detail": "Study CLIP positive/negative pairs and similarity temperature; text alignment does not directly supply biological instance labels.",
    "url": "https://arxiv.org/abs/2103.00020"
  },
  {
    "title": "Point and mesh tokens",
    "detail": "Study neighborhoods, relative geometry, and order invariance for unstructured point and mesh data.",
    "url": "https://arxiv.org/abs/2012.09164"
  },
  {
    "title": "Sparse and approximate attention",
    "detail": "Distinguish changing the interaction graph from approximating attention; measure runtime, memory, and quality."
  },
  {
    "title": "Mixture of experts",
    "detail": "Inspect routing, capacity, load balancing, and active versus total parameters. Sparse activation still has communication costs."
  },
  {
    "title": "Adapters and prompt tuning",
    "detail": "State exactly which parameters change and what information the prompt carries."
  },
  {
    "title": "Probing and intervention",
    "detail": "Probe useful features and intervene on context; attention maps alone are insufficient causal evidence."
  },
  {
    "title": "Active annotation",
    "detail": "Select uncertain or representative examples; measure expert correction time on an independent curated subset."
  },
  {
    "title": "Synthetic-to-real learning",
    "detail": "Perturb acquisition and appearance assumptions and test which effects transfer."
  },
  {
    "title": "Multimodal microscopy",
    "detail": "Track channels, acquisition, registration, missing modalities, and cross-attention design."
  },
  {
    "title": "Restoration before segmentation",
    "detail": "Test whether denoising improves biological measurements or erases useful structures."
  },
  {
    "title": "Continual/domain adaptation",
    "detail": "Connect to stability–plasticity while measuring retention and avoiding unsupported biological claims."
  }
];
