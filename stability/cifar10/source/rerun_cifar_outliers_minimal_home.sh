#!/usr/bin/env bash
set -euo pipefail

# Targeted follow-up for the CIFAR stability scans.
# This intentionally does NOT rerun the full experiment grid.
# It adds seeds 5,6,7 and reruns only old seeds that were cut off by
# parameter_safety_limit near the suspicious/boundary regions.

CODE=${CODE:-/home/jnwang/momentum/cifar10_stability/cifar_stability_revised.py}
OUT=${OUT:-/home/jnwang/momentum/cifar10_stability/results/cifar_stability_targeted_reruns_20260920}
DEVICES=${DEVICES:-cuda:0,cuda:1}

COMMON=(
  --preset pilot
  --devices "$DEVICES"
  --eval-every 50
  --tail-evals 10
  --parameter-safety-mode record
  --no-plot
)

mkdir -p "$OUT"

echo "Code:    $CODE"
echo "Output:  $OUT"
echo "Devices: $DEVICES"
echo

# -----------------------------------------------------------------------------
# 1) SGD, fixed-rho plot, B=1.
# Existing results: transition is between ~1.2 and 2, but several old seeds
# were stopped by the parameter-norm guard.  Densify only this transition.
# 8 seeds x 5 LRs = 40 trials.
# -----------------------------------------------------------------------------
python3 "$CODE" "${COMMON[@]}" \
  --axis batch --algorithms sgd \
  --batches 1 --fixed-gap 0.015625 \
  --seeds 0,1,2,3,4,5,6,7 \
  --steps 2000 \
  --lrs 1.189207,1.414214,1.6,1.8,2 \
  --output "$OUT/01_sgd_B1_transition"

# -----------------------------------------------------------------------------
# 2) Polyak, fixed rho=1-1/64, B=32.
# Old safety-stop seeds in this region are mainly 0 and 2; add seeds 5,6,7.
# We start at 0.8409 because lower completed runs are already enough to show
# the stable side; 2.378 is kept as the upper-side check.
# 5 seeds x 7 LRs = 35 trials.
# -----------------------------------------------------------------------------
python3 "$CODE" "${COMMON[@]}" \
  --axis batch --algorithms polyak \
  --batches 32 --fixed-gap 0.015625 \
  --seeds 0,2,5,6,7 \
  --steps 2000 \
  --lrs 0.840897,1,1.189207,1.414214,1.681793,2,2.378414 \
  --output "$OUT/02_polyak_B32_transition"

# -----------------------------------------------------------------------------
# 3) Polyak, fixed rho=1-1/64, B=128.
# Old safety-stop seeds near the transition are 0,1,4; add 5,6,7.
# 6 seeds x 4 LRs = 24 trials.
# -----------------------------------------------------------------------------
python3 "$CODE" "${COMMON[@]}" \
  --axis batch --algorithms polyak \
  --batches 128 --fixed-gap 0.015625 \
  --seeds 0,1,4,5,6,7 \
  --steps 2000 \
  --lrs 1.414214,1.681793,2,2.378414 \
  --output "$OUT/03_polyak_B128_transition"

# -----------------------------------------------------------------------------
# 4) Polyak, fixed rho=1-1/64, B=256.
# Old safety-stop seeds near the upper transition are 2,3,4; add 5,6,7.
# 6 seeds x 3 LRs = 18 trials.
# -----------------------------------------------------------------------------
python3 "$CODE" "${COMMON[@]}" \
  --axis batch --algorithms polyak \
  --batches 256 --fixed-gap 0.015625 \
  --seeds 2,3,4,5,6,7 \
  --steps 2000 \
  --lrs 2,2.378414,2.828427 \
  --output "$OUT/04_polyak_B256_transition"

# -----------------------------------------------------------------------------
# 5) Polyak, fixed B=4, 1-rho=0.125.
# This is the nonmonotone row singled out in the fixed-B plot.
# Old suspicious seeds are 0,1,2,4; add 5,6,7.
# 7 seeds x 5 LRs = 35 trials.
# -----------------------------------------------------------------------------
python3 "$CODE" "${COMMON[@]}" \
  --axis gap --algorithms polyak \
  --fixed-batch 4 --gaps 0.125 \
  --seeds 0,1,2,4,5,6,7 \
  --steps 2000 \
  --lrs 1.414214,1.681793,2,2.378414,2.828427 \
  --output "$OUT/05_polyak_B4_gap0125"

# -----------------------------------------------------------------------------
# 6a) Nesterov, fixed B=4: add one genuinely smaller damping gap.
# This gives another point in the unsaturated regime.  We use K=4000 so that
# K(1-rho)=31.25 at gap=1/128, comparable to K(1-rho)=31.25 for the old
# gap=1/64 runs with K=2000.
# 5 seeds x 6 LRs = 30 trials.
# -----------------------------------------------------------------------------
python3 "$CODE" "${COMMON[@]}" \
  --axis gap --algorithms nesterov \
  --fixed-batch 4 --gaps 0.0078125 \
  --seeds 0,1,2,3,4 \
  --steps 4000 \
  --lrs 0.25,0.353553,0.420448,0.5,0.594604,0.707107 \
  --output "$OUT/06a_nesterov_B4_gap_1over128"

# -----------------------------------------------------------------------------
# 6b-e) Nesterov: rerun ONLY old seeds that hit parameter_safety_limit near
# the useful transition for the existing small-gap points.
# Total = 2 + 2 + 8 + 6 = 18 trials.
# -----------------------------------------------------------------------------
python3 "$CODE" "${COMMON[@]}" \
  --axis gap --algorithms nesterov \
  --fixed-batch 4 --gaps 0.015625 \
  --seeds 0,2 \
  --steps 2000 \
  --lrs 0.707107 \
  --output "$OUT/06b_nesterov_B4_gap_1over64_safety"

python3 "$CODE" "${COMMON[@]}" \
  --axis gap --algorithms nesterov \
  --fixed-batch 4 --gaps 0.03125 \
  --seeds 0,3 \
  --steps 2000 \
  --lrs 1 \
  --output "$OUT/06c_nesterov_B4_gap_1over32_safety"

python3 "$CODE" "${COMMON[@]}" \
  --axis gap --algorithms nesterov \
  --fixed-batch 4 --gaps 0.0625 \
  --seeds 0,2,3,4 \
  --steps 2000 \
  --lrs 1.189207,1.414214 \
  --output "$OUT/06d_nesterov_B4_gap_1over16_safety"

python3 "$CODE" "${COMMON[@]}" \
  --axis gap --algorithms nesterov \
  --fixed-batch 4 --gaps 0.125 \
  --seeds 0,1,2 \
  --steps 2000 \
  --lrs 1.414214,1.681793 \
  --output "$OUT/06e_nesterov_B4_gap_1over8_safety"

echo
echo "Finished targeted reruns."
echo "All outputs are under: $OUT"
echo "Planned new trials: 200"
