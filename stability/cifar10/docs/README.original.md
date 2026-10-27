# CIFAR-10 Stability Experiment

Compare SGD, Polyak, and Nesterov stability boundaries (Appendix E.2.2).
Train ResNet-18 with GroupNorm for 2,000 updates, using constant learning rates and momentum, without warmup or weight decay.
Scan B = 1, 2, ..., 256 at 1−β = 1/64, and nine logarithmically spaced damping values from 1/64 to 1/4 at B = 4.
Evaluate a fixed 1,024-image training subset every 50 updates. Stability requires the final ten-loss average to be at most twice the initial loss and 1.25 times the preceding ten-loss average; numerical/safety stops count as failures.
Install `requirements.txt` and set the CIFAR-10 data path. Example fresh scan (historical transition reruns use the separate launcher scripts):
```bash
python3 cifar_stability_revised.py --preset full --devices cuda:0,cuda:1 --data-root /data/public/CIFAR-10/cifar-10-python --batches 1,2,4,8,16,32,64,128,256 --gaps 0.015625,0.02209708691207961,0.03125,0.04419417382415922,0.0625,0.08838834764831845,0.125,0.1767766952966369,0.25 --fixed-batch 4 --fixed-gap 0.015625 --lr-min 0.00390625 --lr-max 8 --lr-count 23 --seeds 0,1,2,3,4 --steps 2000 --eval-every 50 --tail-evals 10 --probe-size 1024 --output output/new_scan
python3 plot_cifar_all.py --followup-dir output/nesterov_followup_20260921 --output output/summary_nesterov_followup --individual
```
The second command redraws the final figure from existing follow-up results and their saved baseline. It does not read `output/new_scan`.
Keep manifests, trial configs/summaries, probe histories, baseline tables, and saved source code. The final figure excludes gap = 1/128 and all 4,000-update runs.
Root-level launcher logs, caches, and backup files can be removed. Keep the experiment data under `output/`.
