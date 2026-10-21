# BETA local review — 20 October 2026

This update was prepared and reviewed locally on 20 October 2026. The user subsequently approved publication. GitHub About has been updated, and the reviewed site and media are included in the publication commit.

## Conference and award

The project-page title, header, News, README and citation now identify **NeurIPS 2026**. The **ICLR 2026 TTU Workshop Best Paper Award** remains separately identified in the header and News. The acceptance news uses the year only because no exact acceptance date was supplied.

Two existing technical details were aligned with the saved camera-ready source: the entropy/Jensen–Shannon decomposition now uses its correct equality and positive JS term, and LAME is labeled gray-box in the access tables.

## Motion gallery

1. **Same budget. More images.** The previously approved BETA (Ours) budget movie is included unchanged, with the slower representative gallery and API packets.
2. **Model families.** A new video shows the common interface for a ViT vision classifier, CLIP and BiomedCLIP, with real ImageNet-C, EuroSAT and Derm7pt examples and reported final accuracies.
3. **Stable adaptation.** A new video continuously reveals the original paper’s Contrast ablation curves for full BETA, removing consistency and removing filtering.

The project page uses silent `autoplay loop muted playsinline` MP4s with no native pause/play controls. The README embeds locally stored infinite-loop GIFs. Browser and GitHub account-level accessibility preferences may govern animation playback.

## Model coverage

The new benchmark selector and videos explicitly distinguish vision classifiers from vision-language models. Reported accuracies are 55.5 → 62.6 on ImageNet-C (ViT-B/16), 42.0 → 53.3 on EuroSAT (CLIP ViT-B/16), 55.9 → 58.6 on Derm7pt (CLIP), and 60.9 → 62.1 on Derm7pt (BiomedCLIP).

## GitHub About

GitHub’s About description is server-side repository metadata. Following publication approval, the description was updated to the text saved in `github-about.txt` and `repository-metadata-preview.json`:

> [NeurIPS 2026 | ICLR 2026 TTU Best Paper Award] Adapting in the Dark: Efficient and Stable Test-Time Adaptation for Black-Box Models

The repository homepage remains https://yunbeizhang.github.io/BETA/.
