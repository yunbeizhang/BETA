# Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models

<p align="center">
  🏆 <b>ICLR 2026 TTU Workshop &middot; Oral</b>
</p>


<p align="center">
  <a href="https://arxiv.org/pdf/2604.15609"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square" alt="Paper PDF"></a>
  <a href="https://openreview.net/forum?id=v56b8I1tua"><img src="https://img.shields.io/badge/ICLR%202026%20TTU-Oral-AF5FFF?style=flat-square" alt="ICLR 2026 TTU Oral"></a>
  <a href="https://yunbeizhang.github.io/BETA/"><img src="https://img.shields.io/badge/Project%20Page-GitHub%20Pages-brightgreen?style=flat-square" alt="Project Page"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square" alt="License"></a>
</p>
<p align="center">
  <b>Yunbei Zhang&nbsp;&nbsp;&emsp;&emsp;Shuaicheng Niu&nbsp;&nbsp;&emsp;&emsp;Chengyi Cai&nbsp;&nbsp;&emsp;&emsp;Feng Liu&nbsp;&nbsp;&emsp;&emsp;Jihun Hamm</b>
</p>


## News

- **Apr 16, 2026.** Code released.
- 🏆 **Mar 07, 2026.** Selected as **Oral Presentation** at the Third Workshop on Test-Time Updates (TTU).
- **Mar 01, 2026.** Accepted to the Third Workshop on Test-Time Updates (TTU), ICLR 2026 Workshop.

## TODO

- [x] Release code
- [x] [arXiv preprint](https://arxiv.org/pdf/2604.15609)

## Abstract

<p align="center">
  <img src="assets/settings.png" alt="Black-box TTA setting" width="820">
</p>

Test-Time Adaptation (TTA) for black-box models accessible only via APIs remains a largely unexplored challenge. Existing approaches such as post-hoc output refinement offer limited adaptive capacity, while Zeroth-Order Optimization (ZOO) enables input-space adaptation but faces high query costs and optimization challenges in the unsupervised TTA setting. We introduce **BETA** (Black-box Efficient Test-time Adaptation), a framework that addresses these limitations by employing a lightweight, local white-box *steering model* to create a tractable gradient pathway. Through a *prediction harmonization* technique combined with *consistency regularization* and *prompt learning-oriented filtering*, BETA enables stable adaptation with no additional API calls and negligible latency beyond standard inference. On ImageNet-C, BETA achieves a **+7.1% accuracy gain on ViT-B/16** and **+3.4% on CLIP**, surpassing strong white-box and gray-box methods including TENT and TPT. On a commercial API, BETA achieves comparable performance to ZOO at **250&times; lower cost** while maintaining real-time inference speed.

## Method Overview

<p align="center">
  <img src="assets/workflow.png" alt="BETA Workflow" width="850">
</p>

BETA operates with two models, a powerful frozen black-box target `f_B` and a lightweight local *steering model* `f_S`, and learns an additive visual prompt `δ` that is optimized locally through `f_S`. Because direct gradient transfer between architectures is ineffective, BETA uses **prediction harmonization** to fuse the two outputs into a shared objective. Two stabilizers, namely **consistency regularization** between clean and prompted predictions, and **prompt-learning-oriented filtering**, keep the unsupervised adaptation stable.

## Installation

```bash
git clone https://github.com/yunbeizhang/BETA.git
cd BETA
conda create -n beta python=3.10 -y
conda activate beta
pip install -r requirements.txt
```

## Data Preparation

BETA evaluates on ImageNet-C (Hendrycks & Dietterich, 2019). Download it and set the data roots via the `DATA_DIR` environment variable (see `main.sh`):

```
DATA_DIR/
├── ImageNet/            # original validation set
└── ImageNet-C/          # 15 corruptions × 5 severities
```

Optional domain-shift benchmarks (ImageNet-R / V2 / Sketch / -A) can be placed alongside and passed via `--data_rendition`, `--data_v2`, etc.

## Quick Start

Run BETA with the reference ViT-B/16 configuration:

```bash
bash main.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhang2026adapting,
  title={Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models},
  author={Yunbei Zhang and Shuaicheng Niu and Chengyi Cai and Feng Liu and Jihun Hamm},
  booktitle={Third Workshop on Test-Time Updates (Main Track)},
  year={2026},
  url={https://openreview.net/forum?id=v56b8I1tua}
}
```

## Acknowledgements

This repository builds on [FOA](https://github.com/mr-eggplant/FOA), [BayesianLM](https://github.com/tmlr-group/BayesianLM), and [AReS](https://github.com/yunbeizhang/AReS). Thanks to the authors for open-sourcing their code.

## License

This project is released under the [Apache 2.0 License](LICENSE).
