# The first attention increment

This is an educational CPU example for Modules 01, 04, and 05. It includes ordinary tensor-based attention, explicit multihead projection/reshape/concatenation, analytical Q/K/V and projection gradients, and small numerical invariants. It does not train a model, download data, reproduce a paper benchmark, or implement the full future microscopy repository.

## Files and use

Keep `attention.py` beside `attention-by-hand.ipynb`. With a compatible PyTorch environment, run `python attention.py` for the numerical checks, or open the notebook and execute its cells in order. No installation was performed to create this package. The notebook has no stored execution outputs: run it to create your own evidence.

`requirements.txt` states a compatibility range, not a resolved dependency lock. Before building larger experiments, create an isolated environment and record its resolved dependencies, actual hardware, runtime, storage, and checkpoint/data provenance. Verify the current environment rather than assuming a previous runtime still applies.

## Numerical contract

- Shapes are `[batch, tokens, features]` or `[batch, heads, tokens, features]`. Q and K share compatibility width; V may use a different content width.
- Boolean `True` means **allowed** throughout this example. This agrees with the tested PyTorch functional scaled-dot-product API, and differs from some other attention APIs. Recheck installed documentation before substituting another API.
- Masks apply to scores before softmax. Softmax normalizes across keys. The row maximum is subtracted for numerical stability.
- An entirely blocked row returns zero weights and zero output. This is an explicit policy, not an ordinary softmax distribution. Padded query outputs and padded loss targets still require separate handling in a training application.
- Inputs must be finite. The example detects score overflow rather than silently accepting invalid arithmetic. CPU float64 is used for derivative comparisons. No dropout is applied.
- All backward expressions assume a fixed mask and no dropout. A learned additive bias would require its own parameter gradient.
- Multihead projections are explicit matrices, not a high-level transformer layer. The library attention call appears only as a trusted numerical comparison, with dropout explicitly zero.

## Understanding before execution

Predict the softmax axis, each output shape, and what changes when only V changes. Then predict what should happen when a future token or padded key changes. Distinguish attention weights from task association probabilities. Save your configuration, outputs, discrepancies, and short interpretation; passing the checks does not by itself establish mastery.

## Verification scope

The host inspection found Python 3.13.9, PyTorch 2.13.0+cpu, and no available CUDA device. Verification results are saved in `verification.json`. All six notebook code cells were also executed sequentially in a standard Python namespace; no Jupyter kernel was launched, and the evidence-file path was redirected for validation. This is one deterministic numerical run, not a training result, independent-seed robustness study, or performance benchmark. Hardware outside the selected CPU runtime was not profiled.

The starter does not implement KV caching, positional embeddings, normalization, a complete transformer block, segmentation, or lineage decoding. Those remain subsequent increments. The supplied roadmap and website lessons describe them without claiming they were executed.
