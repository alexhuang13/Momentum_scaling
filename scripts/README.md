# Plotting and maintenance scripts

These scripts regenerate figures from formulas or already saved data. They do
not run the four numerical experiment categories.

- `plot_stability_schematic.py`: theoretical stability-boundary schematic;
- `plot_rate_diagram.py`: data-scaling phase diagram and legend;
- `plot_critical_batch_size.py`: plot a committed critical-batch-size CSV;
- `plot_stability_reference.py`: render the eight saved reference grids.

Defaults write newly generated figures under `results/figures/`, except the
reference-grid script, which intentionally refreshes curated files in
`assets/figures/`.
