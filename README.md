# Transformer Atlas

An interactive educational journey from transformer fundamentals to microscopy segmentation, cell association, and valid lineages.

**Website:** [kenddly.github.io/transformer-atlas](https://kenddly.github.io/transformer-atlas/)

## Explore

- 25 lessons with Essential, Technical, and Research explanations, equations, tensor traces, numerical examples, exercises, failure cases, and understanding checks.
- Six interactive laboratories: attention, image patches, tensor dimensions, cell tracking, instance segmentation, and one-to-one assignment.
- A 50-concept glossary, primary reading library, microscopy paper route, readiness gates, and an 18-experiment research ledger.
- Browser-local progress and notes, with notebook export.
- A public educational blueprint and a downloadable CPU attention starter.

The browser demonstrations use small synthetic examples. The PyTorch attention starter includes recorded numerical verification. Larger model training, real-data studies, and benchmark reproductions remain planned exercises.

## Run locally

Serve this directory with any static web server:

```sh
python -m http.server 8000
```

Open `http://localhost:8000/`. No application build or backend is required. Notes and progress are stored separately for each browser and site origin.

## GitHub Pages

GitHub Pages publishes the root directory of the `main` branch. `.nojekyll` preserves the plain HTML, CSS, JavaScript, and downloads. Push changes to `main` to update the website.

Assets use relative paths and lesson navigation uses URL fragments, so the project works under `/transformer-atlas/`.

## Learning materials

- [Public educational blueprint](blueprint.md)
- [Attention starter ZIP](downloads/attention-starter.zip)
- [Attention notebook](downloads/attention-by-hand.ipynb)
- [Implementation notes and verification scope](downloads/implementation-notes.md)

Interaction inspiration includes [CNN Explainer](https://poloclub.github.io/cnn-explainer/), [Transformer Explainer](https://poloclub.github.io/transformer-explainer/), and [Distill’s Attention and Augmented RNNs](https://distill.pub/2016/augmented-rnns/). Lesson explanations and visuals are original adaptations of the learning blueprint.
