"""
Run all four inverse PINN formulations under observation noise (sigma = 0%, 1%, 5%, 10%).
sigma=0% uses the original clean CSV so all noise levels go through the SAME pipeline.

Generates noisy CSVs if they don't exist, runs each formulation, collects results,
saves them to noise_results.json, and produces the summary table + plot.

Run from project root:  python extras/run_noise_study.py
"""
import subprocess, sys, re, os, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOISE_LEVELS = [0, 1, 5, 10]  # 0% = clean data, actually run through the pipeline
SEED = 1234  # same seed for all runs

FORMULATIONS = [
    {
        'name': 'Baseline',
        'script': 'Fire_PINN_2D/pinn_inverse_2D.py',
        'orig_csv': 'Fire_PINN_2D/data/fire_dataset_columns.csv',
        'params': textwrap.dedent("""\
            _fp = fire_params
            _pairs = [
                ('lambda', _fp['lamb'], model.lambs[-1]),
                ('beta',   _fp['beta'], model.betas[-1]),
                ('w1',     _fp['w1'],   model.w1s[-1]),
                ('w2',     _fp['w2'],   model.w2s[-1]),
            ]
        """),
    },
    {
        'name': 'MV',
        'script': 'Fire_PINN_Multi_variable/pinn_inverse_multi_variable.py',
        'orig_csv': 'Fire_PINN_Multi_variable/data/fire_dataset_columns.csv',
        'params': textwrap.dedent("""\
            _fp = fire_params
            _pairs = [
                ('lambda1', _fp['lamb1'], model.lamb1_history[-1]),
                ('beta1',   _fp['beta1'], model.beta1_history[-1]),
                ('lambda2', _fp['lamb2'], model.lamb2_history[-1]),
                ('beta2',   _fp['beta2'], model.beta2_history[-1]),
                ('lambda3', _fp['lamb3'], model.lamb3_history[-1]),
                ('beta3',   _fp['beta3'], model.beta3_history[-1]),
                ('lambda4', _fp['lamb4'], model.lamb4_history[-1]),
                ('beta4',   _fp['beta4'], model.beta4_history[-1]),
                ('w1',      _fp['w1'],    model.w1s[-1]),
                ('w2',      _fp['w2'],    model.w2s[-1]),
            ]
        """),
    },
    {
        'name': 'MD',
        'script': 'Fire_PINN_Multi_diffusion/pinn_inverse_multi_diffusion.py',
        'orig_csv': 'Fire_PINN_Multi_diffusion/data/fire_dataset_columns.csv',
        'params': textwrap.dedent("""\
            _fp = fire_params
            _pairs = [
                ('lambda',  _fp['lamb'],   model.lambs[-1]),
                ('beta',    _fp['beta'],   model.betas[-1]),
                ('w1_q1',   _fp['w1_q1'],  model.w1_q1s[-1]),
                ('w2_q1',   _fp['w2_q1'],  model.w2_q1s[-1]),
                ('w1_q2',   _fp['w1_q2'],  model.w1_q2s[-1]),
                ('w2_q2',   _fp['w2_q2'],  model.w2_q2s[-1]),
                ('w1_q3',   _fp['w1_q3'],  model.w1_q3s[-1]),
                ('w2_q3',   _fp['w2_q3'],  model.w2_q3s[-1]),
                ('w1_q4',   _fp['w1_q4'],  model.w1_q4s[-1]),
                ('w2_q4',   _fp['w2_q4'],  model.w2_q4s[-1]),
            ]
        """),
    },
    {
        'name': 'MVD',
        'script': 'Fire_PINN_Multi_variable_diffusion/pinn_inverse_multi_var_dif.py',
        'orig_csv': 'Fire_PINN_Multi_variable_diffusion/data/fire_dataset_columns.csv',
        'params': textwrap.dedent("""\
            _fp = fire_params
            _pairs = [
                ('lambda1', _fp['lamb1'], model.lamb1_history[-1]),
                ('beta1',   _fp['beta1'], model.beta1_history[-1]),
                ('lambda2', _fp['lamb2'], model.lamb2_history[-1]),
                ('beta2',   _fp['beta2'], model.beta2_history[-1]),
                ('lambda3', _fp['lamb3'], model.lamb3_history[-1]),
                ('beta3',   _fp['beta3'], model.beta3_history[-1]),
                ('lambda4', _fp['lamb4'], model.lamb4_history[-1]),
                ('beta4',   _fp['beta4'], model.beta4_history[-1]),
                ('w1_q1',   _fp['w1_q1'], model.w1_q1s[-1]),
                ('w2_q1',   _fp['w2_q1'], model.w2_q1s[-1]),
                ('w1_q2',   _fp['w1_q2'], model.w1_q2s[-1]),
                ('w2_q2',   _fp['w2_q2'], model.w2_q2s[-1]),
                ('w1_q3',   _fp['w1_q3'], model.w1_q3s[-1]),
                ('w2_q3',   _fp['w2_q3'], model.w2_q3s[-1]),
                ('w1_q4',   _fp['w1_q4'], model.w1_q4s[-1]),
                ('w2_q4',   _fp['w2_q4'], model.w2_q4s[-1]),
            ]
        """),
    },
]

RESULT_TAIL = textwrap.dedent("""\
    import sys as _s
    _reaction  = [abs(e-t)/t*100 for n,t,e in _pairs if n.startswith(('lambda','beta'))]
    _diffusion = [abs(e-t)/t*100 for n,t,e in _pairs if n.startswith('w')]
    for _n,_t,_e in _pairs:
        print(f"##PARAM## {_n}: true={_t:.6f}  estimated={_e:.6f}  error={abs(_e-_t)/_t*100:.2f}%")
    print(f"##RESULT## r_mean={sum(_reaction)/len(_reaction):.4f} "
          f"d_mean={sum(_diffusion)/len(_diffusion):.4f}")
    _s.exit(0)
""")


# ── Generate noisy datasets if they don't exist ───────────────────
def generate_noisy_data():
    """Create noisy CSVs for sigma = 1%, 5%, 10% using seed 42."""
    np.random.seed(42)
    datasets = [f['orig_csv'] for f in FORMULATIONS]
    for ds_path in datasets:
        folder = os.path.dirname(ds_path)
        for sigma_pct in [1, 5, 10]:
            out_path = os.path.join(folder, f'fire_dataset_noise_{sigma_pct}.csv')
            print(f"  Generating {out_path} ...")
            df = pd.read_csv(ds_path)
            std_T = df['T'].std()
            std_F = df['F'].std()
            n = len(df)
            df['T'] = df['T'] + np.random.normal(0, std_T * sigma_pct / 100.0, n)
            df['F'] = df['F'] + np.random.normal(0, std_F * sigma_pct / 100.0, n)
            df.to_csv(out_path, index=False)


def build_wrapper(script_path, csv_to_use, orig_csv, param_snippet):
    """Patch an inverse script to use a given CSV, suppress plots, and extract results."""
    with open(script_path, 'r', encoding='utf-8') as f:
        src = f.read()
    src = f"import matplotlib; matplotlib.use('Agg')\n" + src
    # Point to the correct dataset (noisy or clean)
    if csv_to_use != orig_csv:
        src = src.replace(orig_csv, csv_to_use)
    src = src.replace("plt.show()", "pass")
    # Replace Greek Unicode chars to avoid cp1252 crash on Windows
    src = src.replace('\u03bb', 'lambda').replace('\u03b2', 'beta')
    marker = "#------------------------ Visualization -----------------------------"
    src = src.replace(marker, param_snippet + RESULT_TAIL + marker, 1)
    return src


# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Checking for noisy datasets...")
    generate_noisy_data()

    # results[name][sigma] = {'r_mean': ..., 'd_mean': ...}
    results = {f['name']: {} for f in FORMULATIONS}

    for form in FORMULATIONS:
        name = form['name']
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)

        for sigma in NOISE_LEVELS:
            # sigma=0 uses the original clean CSV
            if sigma == 0:
                csv_to_use = form['orig_csv']
            else:
                csv_to_use = form['orig_csv'].replace(
                    'fire_dataset_columns.csv',
                    f'fire_dataset_noise_{sigma}.csv'
                )

            print(f"\n  --- sigma = {sigma}% ---")
            print(f"  Data: {csv_to_use}")

            if not os.path.exists(csv_to_use):
                print(f"  ERROR: {csv_to_use} not found, skipping")
                continue

            wrapper_src = build_wrapper(
                form['script'], csv_to_use, form['orig_csv'], form['params']
            )
            tmp = f"_tmp_noise_{name}_{sigma}.py"
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(wrapper_src)
            try:
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                proc = subprocess.run(
                    [sys.executable, tmp],
                    capture_output=True, text=True, encoding='utf-8',
                    errors='replace', timeout=7200, env=env
                )
                for line in proc.stdout.splitlines():
                    if '##PARAM##' in line:
                        print(f"    {line.split('##PARAM##')[1].strip()}")

                match = re.search(r'##RESULT##\s*(.+)', proc.stdout)
                if match:
                    d = dict(tok.split('=', 1) for tok in match.group(1).strip().split())
                    r = {
                        'r_mean': round(float(d['r_mean']), 2),
                        'd_mean': round(float(d['d_mean']), 2),
                    }
                    results[name][sigma] = r
                    print(f"  => reaction={r['r_mean']:.2f}%  diffusion={r['d_mean']:.2f}%")
                else:
                    print("  ERROR: could not parse results")
                    print(proc.stdout[-500:])
                    print(proc.stderr[-300:])
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

    # ── Save results to JSON ──────────────────────────────────────
    json_path = 'extras/noise_results.json'
    # Convert int keys to strings for JSON
    json_results = {}
    for name, sigmas in results.items():
        json_results[name] = {str(s): v for s, v in sigmas.items()}
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Print summary table ───────────────────────────────────────
    names = [f['name'] for f in FORMULATIONS]
    print("\n\n" + "="*70)
    print("SUMMARY - mean relative error (%) vs noise level")
    print("="*70)
    print(f"{'sigma':>6}", end='')
    for n in names:
        print(f"  {n+' r':>10}  {n+' d':>10}", end='')
    print()
    print("-"*90)

    for sigma in NOISE_LEVELS:
        print(f"{str(sigma)+'%':>6}", end='')
        for n in names:
            if sigma in results[n]:
                r = results[n][sigma]
                print(f"  {r['r_mean']:>10.2f}  {r['d_mean']:>10.2f}", end='')
            else:
                print(f"  {'---':>10}  {'---':>10}", end='')
        print()

    # ── Print LaTeX table ─────────────────────────────────────────
    print("\n\nLaTeX table rows (copy-paste into main.tex):")
    print("-"*70)
    for sigma in NOISE_LEVELS:
        row = f"{sigma}\\%"
        for n in names:
            if sigma in results[n]:
                r = results[n][sigma]
                row += f"  & {r['r_mean']:.2f} & {r['d_mean']:.2f}"
            else:
                row += "  & --- & ---"
        row += " \\\\"
        print(row)

    # ── Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'Baseline': '#1f77b4', 'MV': '#ff7f0e', 'MD': '#2ca02c', 'MVD': '#d62728'}

    for n in names:
        available = sorted(results[n].keys())
        if not available:
            continue
        r_vals = [results[n][s]['r_mean'] for s in available]
        d_vals = [results[n][s]['d_mean'] for s in available]
        ax.plot(available, r_vals, 'o-',  color=colors[n], linewidth=2.5, markersize=9,
                label=f'{n} reaction ($\\lambda,\\beta$)')
        ax.plot(available, d_vals, 's--', color=colors[n], linewidth=2.5, markersize=9,
                label=f'{n} diffusion ($w$)')

    ax.set_xlabel('Observation noise $\\sigma$ (% of field std)', fontsize=13)
    ax.set_ylabel('Mean relative error (%)', fontsize=13)
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(0, None)
    ax.set_xticks(NOISE_LEVELS)
    ax.set_xticklabels([f'{s}%' for s in NOISE_LEVELS], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=10, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title('Inverse parameter recovery under observation noise', fontsize=14)

    plt.tight_layout()
    plot_path = 'Latex/results/noise_robustness.png'
    plt.savefig(plot_path, dpi=200)
    print(f"\nPlot saved to {plot_path}")

    # Also save a copy in extras
    plt.savefig('extras/noise_robustness.png', dpi=200)
    print(f"Plot also saved to extras/noise_robustness.png")
