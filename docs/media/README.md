# BETA media sources

The project page uses looping H.264 MP4s; the README uses infinite-loop GIFs of the same three videos.

- `beta-budget.mp4` is the previously approved dedicated ImageNet-C Contrast replay, preserved byte for byte. Its source implementation is the BETA homepage-mp4-ours production in the local paper workspace. It illustrates the complete shared $50 query budget. Dollar values convert local query counts at $0.0032/image query; no commercial API charges were incurred. See `assets/budget-provenance.json`. Representative gallery sampling reduces visual frequency; all original curve points and image counts remain included.
- `beta-model-families.mp4` animates a common mechanism over representative real inputs. The four numerical pairs are reported final accuracies, not intermediate training measurements or per-image predictions. See `assets/reported-results.json`.
- `beta-stability.mp4` redraws original paper Figure `batch_acc_contrast.pdf` vector paths, with no synthetic perturbations or new measurements. See `assets/paper-chart-data.json`. Display interpolation reveals the existing paths continuously.

Image provenance and hashes are saved in `assets/provenance.json`. Derm7pt source filenames are in `assets/derm7pt/sources.json`; the six photos were resized to at most 640 px for presentation. The representative galleries do not assign diagnoses or before/after predictions to individual inputs.

Both new videos are 1280×720, 30 FPS, 20 seconds of motion plus a 0.5 second final hold, H.264 / yuv420p / faststart, without audio. The existing movie retains 960×560, 60 FPS and 21 seconds.

To regenerate the new MP4s from these saved inputs on macOS, install Pillow and NumPy and make ffmpeg available, then run from the repository root:

```sh
python3 docs/media/render_videos.py --output assets/video
```

The renderer uses the system Arial regular and bold fonts. The visual feedback packets illustrate the method; their speed is not measured API latency. The website’s separate GIF copies use 800 px width, 12 FPS, a 128-color palette and `loop=0`.
