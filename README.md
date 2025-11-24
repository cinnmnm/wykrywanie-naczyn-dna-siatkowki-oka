Project: Retinal vessel segmentation

Quick start

1. Install dependencies
```powershell
pip install -r requirements.txt
```

2. Train DL Unet model (run from the repo root so relative paths in `config.yaml` stay valid)
```powershell
python -m models.dlunet.Core.train --config config.yaml
```

3. Train Random Forest model
```powershell
python scripts/train_rf.py --config config.yaml --log-dir logs
```

4. Run inference for DL Unet
```powershell
python scripts/inference.py --config config.yaml --model checkpoints/unet/unet_best.pt --model-type dlunet --output output
```

5. Launch interactive app notebook
```powershell
jupyter notebook app.ipynb
```
- Open `app.ipynb`, run Cell 1 to load notebook extensions, then run Cell 2 to create the controller and GUI window.
- The notebook uses the same virtual environment, so ensure it is activated before starting Jupyter.

Trainer tips
- Enable `use_augmentation: true` in `config.yaml` (DLUnet section) to apply geometric augmentations
- Enable `balance_patches: true` to sample patches so the model sees more vessel-containing patches
- Set `num_workers` to your CPU count for faster data loading

Logging
- Logs are written into `logs/<timestamp>/project.log` by default.
- Use `--log-dir` to set log directory for scripts.
- You can request JSON logs by adding `--log-format json` if the `python-json-logger` package is installed.

