# ImageNet-1K: Batch-Size Scaling at a Fixed Sample Budget

This experiment reproduces Section 7.2 and Appendix E.2.1 of the paper, comparing SGD, Polyak, and Nesterov across batch sizes to examine the predicted qualitative behavior: Polyak enlarges the critical batch size, while Nesterov further reduces risk in the large-batch regime.
Train ResNet-18 from scratch with GroupNorm replacing BatchNorm. Fix the sample budget at D = 3 × 1,281,167 = 3,843,501 (three epochs), use B ∈ {1, 2, 4, ..., 4096}, and perform floor(D/B) updates. For B > 128, accumulate gradients over micro-batches of size 128.
All runs use constant learning rates and momentum factors, BF16, and seed 0, without warmup, weight decay, or label smoothing. Evaluate each final model on all 50,000 validation images and report the lowest validation cross-entropy among completed stable configurations for each method and batch size.
The recorded search contains 396 configurations: 360 in the main grid, 31 additional small-batch runs, and 5 additional Nesterov runs at higher learning rates. The main momentum grid is ρ ∈ {0.8, 0.9, 0.95, 0.98}; the effective learning rate is γ = η for SGD and γ = η/(1−ρ) for momentum methods.
The scripts handle individual training runs and aggregation (`imagenet_momentum_experiment.py`), hyperparameter searches (`imagenet_pilot_driver.py`), and plotting (`plot_imagenet_cbs.py`). Restore these original filenames and prepare an ImageNet directory containing `train/` and `val/` before running.
The Bash commands below reconstruct the recorded configurations. Adjust the code and data paths as needed. Configurations with an existing `summary.json` are skipped. The final commands produce `best_by_batch.csv` and the validation-loss-versus-batch-size figure.

```bash
cd /root/cbs_exp
common=(--script "$PWD/imagenet_momentum_experiment.py" --data-root /root/autodl-tmp/imagenet --runs-root "$PWD/results/pilot_momentum_3epochs" --epochs 3 --execute --no-summary)
for rho in 0.8 0.9 0.95 0.98; do python imagenet_pilot_driver.py "${common[@]}" --batches 32,64,128,256,512,1024,2048,4096 --momentum "$rho" --multipliers 0.5,1,2,4,8; done
python imagenet_pilot_driver.py "${common[@]}" --batches 1 --momentum 0.8 --reference-effective-lr 0.4 --multipliers 2
python imagenet_pilot_driver.py "${common[@]}" --batches 2,4,8,16 --momentum 0.8 --reference-effective-lr 0.4 --multipliers 1,2
python imagenet_pilot_driver.py "${common[@]}" --algorithms polyak,nesterov --batches 16 --momentum 0.9 --reference-effective-lr 0.4 --multipliers 1,2
for spec in "64 0.98 0.4" "128 0.95 0.8" "128 0.98 0.8" "128 0.98 1.6" "256 0.95 1.6"; do read -r b rho gamma <<< "$spec"; python imagenet_pilot_driver.py "${common[@]}" --algorithms nesterov --batches "$b" --momentum "$rho" --reference-batch "$b" --reference-effective-lr "$gamma" --effective-lr-cap "$gamma" --multipliers 1; done
python imagenet_momentum_experiment.py summarize --runs-root results/pilot_momentum_3epochs --required-runs 1
python plot_imagenet_cbs.py --data-dir results/pilot_momentum_3epochs
```
