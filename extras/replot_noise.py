"""
Regenerate noise_robustness.png from the values in the LaTeX table.
Run from project root:  python extras/replot_noise.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOISE_LEVELS = [1, 5, 10]

# Values matching the LaTeX table (tab:noise) in main.tex
data = {
    'MV':       {'r': [2.48, 3.89, 6.30], 'd': [5.59, 6.70, 11.81]},
    'MD':       {'r': [1.95, 2.21, 3.32], 'd': [9.36, 10.55, 16.03]},
    'MVD':      {'r': [2.07, 2.87, 5.93], 'd': [8.16, 12.50, 15.33]},
}

colors = {'MV': '#ff7f0e', 'MD': '#2ca02c', 'MVD': '#d62728'}

fig, ax = plt.subplots(figsize=(10, 7))

for name in ['MV', 'MD', 'MVD']:
    ax.plot(NOISE_LEVELS, data[name]['r'], 'o-', color=colors[name],
            linewidth=2.5, markersize=9,
            label=f'{name} reaction ($\\lambda,\\beta$)')
    ax.plot(NOISE_LEVELS, data[name]['d'], 's--', color=colors[name],
            linewidth=2.5, markersize=9,
            label=f'{name} diffusion ($w$)')

ax.set_xlabel('Observation noise $\\sigma$ (% of field std)', fontsize=13)
ax.set_ylabel('Mean relative error (%)', fontsize=13)
ax.set_xlim(0, 11)
ax.set_ylim(0, None)
ax.set_xticks(NOISE_LEVELS)
ax.set_xticklabels([f'{s}%' for s in NOISE_LEVELS], fontsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=10, ncol=2, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_title('Inverse parameter recovery under observation noise (5-run average)',
             fontsize=14)

plt.tight_layout()
plt.savefig('Latex/results/noise_robustness.png', dpi=200)
plt.savefig('extras/noise_robustness.png', dpi=200)
print("Saved to Latex/results/noise_robustness.png")
print("Saved to extras/noise_robustness.png")
