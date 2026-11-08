"""
Run all four inverse PINN formulations under observation noise (sigma = 0%, 1%, 5%, 10%).
For each non-zero sigma, run 5 different noise seeds and report the average error.
sigma=0% is run once (no noise, so no seed variation).

Generates noisy CSVs, runs each formulation, collects results,
saves them to noise_results_avg.json, and produces the summary table + plot.

Run from project root:  python extras/run_noise_study_avg.py
"""
import subprocess, sys, re, os, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NOISE_LEVELS = [0, 1, 5, 10]
NOISE_SEEDS = [42, 123, 456, 789, 1010]  # 5 seeds for noisy data
TRAIN_SEED = 1234  # same training seed for all runs

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


def generate_noisy_csv(orig_csv, sigma_pct, seed):
    """Create a noisy CSV for a given sigma and seed. Returns the output path."""
    folder = os.path.dirname(orig_csv)
    out_name = 'fire_dataset_noise_' + str(sigma_pct) + '_seed' + str(seed) + '.csv'
    out_path = os.path.join(folder, out_name)
    rng = np.random.RandomState(seed)
    df = pd.read_csv(orig_csv)
    n = len(df)
    std_T = df['T'].std()
    std_F = df['F'].std()
    df['T'] = df['T'] + rng.normal(0, std_T * sigma_pct / 100.0, n)
    df['F'] = df['F'] + rng.normal(0, std_F * sigma_pct / 100.0, n)
    df.to_csv(out_path, index=False)
    return out_path.replace('\\', '/')


def build_wrapper(script_path, csv_to_use, orig_csv, param_snippet):
    """Patch an inverse script to use a given CSV, suppress plots, and extract results."""
    with open(script_path, 'r', encoding='utf-8') as f:
        src = f.read()
    src = f"import matplotlib; matplotlib.use('Agg')\n" + src
    if csv_to_use != orig_csv:
        src = src.replace(orig_csv, csv_to_use)
    src = src.replace("plt.show()", "pass")
    src = src.replace('\u03bb', 'lambda').replace('\u03b2', 'beta')
    marker = "#------------------------ Visualization -----------------------------"
    src = src.replace(marker, param_snippet + RESULT_TAIL + marker, 1)
    return src


def run_single(form, csv_to_use, tag):
    """Run one inverse PINN and return {'r_mean': ..., 'd_mean': ...} or None."""
    wrapper_src = build_wrapper(
        form['script'], csv_to_use, form['orig_csv'], form['params']
    )
    tmp = f"_tmp_noise_{form['name']}_{tag}.py"
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
                print(f"      {line.split('##PARAM##')[1].strip()}")

        match = re.search(r'##RESULT##\s*(.+)', proc.stdout)
        if match:
            d = dict(tok.split('=', 1) for tok in match.group(1).strip().split())
            return {
                'r_mean': float(d['r_mean']),
                'd_mean': float(d['d_mean']),
            }
        else:
            print("    ERROR: could not parse results")
            print(proc.stdout[-500:])
            print(proc.stderr[-300:])
            return None
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    # raw_results[name][sigma] = list of {r_mean, d_mean} dicts (one per seed)
    raw_results = {f['name']: {s: [] for s in NOISE_LEVELS} for f in FORMULATIONS}

    for form in FORMULATIONS:
        name = form['name']
        print(f"\n{'='*60}")
        print(f"  {name}")
        print('='*60)

        for sigma in NOISE_LEVELS:
            if sigma == 0:
                # Clean data, run once
                print(f"\n  --- sigma = 0% (clean, single run) ---")
                r = run_single(form, form['orig_csv'], f"s0")
                if r:
                    raw_results[name][0].append(r)
                    print(f"    => reaction={r['r_mean']:.2f}%  diffusion={r['d_mean']:.2f}%")
            else:
                # Noisy data, run once per seed
                for seed in NOISE_SEEDS:
                    print(f"\n  --- sigma = {sigma}%, seed = {seed} ---")
                    csv_path = generate_noisy_csv(form['orig_csv'], sigma, seed)
                    print(f"    Data: {csv_path}")
                    r = run_single(form, csv_path, f"s{sigma}_sd{seed}")
                    if r:
                        raw_results[name][sigma].append(r)
                        print(f"    => reaction={r['r_mean']:.2f}%  diffusion={r['d_mean']:.2f}%")
                    # Clean up the noisy CSV
                    if os.path.exists(csv_path):
                        os.remove(csv_path)

    # ── Compute averages ─────────────────────────────────────────
    # avg_results[name][sigma] = {'r_mean': avg, 'd_mean': avg, 'r_std': std, 'd_std': std, 'n_runs': n}
    avg_results = {f['name']: {} for f in FORMULATIONS}
    for name in avg_results:
        for sigma in NOISE_LEVELS:
            runs = raw_results[name][sigma]
            if not runs:
                continue
            r_vals = [r['r_mean'] for r in runs]
            d_vals = [r['d_mean'] for r in runs]
            avg_results[name][sigma] = {
                'r_mean': round(np.mean(r_vals), 2),
                'd_mean': round(np.mean(d_vals), 2),
                'r_std':  round(np.std(r_vals), 2),
                'd_std':  round(np.std(d_vals), 2),
                'n_runs': len(runs),
            }

    # ── Save results to JSON ─────────────────────────────────────
    json_path = 'extras/noise_results_avg.json'
    json_out = {}
    for name in avg_results:
        json_out[name] = {str(s): v for s, v in avg_results[name].items()}
        # Also save raw per-seed results
        json_out[name + '_raw'] = {
            str(s): [{'r_mean': round(r['r_mean'], 4), 'd_mean': round(r['d_mean'], 4)} for r in runs_list]
            for s, runs_list in raw_results[name].items()
        }
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Print summary table ──────────────────────────────────────
    names = [f['name'] for f in FORMULATIONS]
    print("\n\n" + "="*70)
    print("SUMMARY - mean relative error (%) vs noise level  [averaged over seeds]")
    print("="*70)
    print(f"{'sigma':>6}", end='')
    for n in names:
        print(f"  {n+' r':>10}  {n+' d':>10}", end='')
    print()
    print("-"*90)

    for sigma in NOISE_LEVELS:
        print(f"{str(sigma)+'%':>6}", end='')
        for n in names:
            if sigma in avg_results[n]:
                r = avg_results[n][sigma]
                r_str = f"{r['r_mean']:.2f}"
                d_str = f"{r['d_mean']:.2f}"
                if r['n_runs'] > 1:
                    r_str += f"\u00b1{r['r_std']:.1f}"
                    d_str += f"\u00b1{r['d_std']:.1f}"
                print(f"  {r_str:>10}  {d_str:>10}", end='')
            else:
                print(f"  {'---':>10}  {'---':>10}", end='')
        print()

    # ── Print per-seed breakdown ─────────────────────────────────
    print("\n\n" + "="*70)
    print("PER-SEED BREAKDOWN")
    print("="*70)
    for name in names:
        print(f"\n  {name}")
        print(f"  {'sigma':>6}  {'seed':>6}  {'reaction':>10}  {'diffusion':>10}")
        print("  " + "-"*40)
        for sigma in NOISE_LEVELS:
            runs = raw_results[name][sigma]
            if sigma == 0:
                for r in runs:
                    print(f"  {'0%':>6}  {'clean':>6}  {r['r_mean']:>9.2f}%  {r['d_mean']:>9.2f}%")
            else:
                for i, r in enumerate(runs):
                    seed = NOISE_SEEDS[i] if i < len(NOISE_SEEDS) else '?'
                    print(f"  {str(sigma)+'%':>6}  {seed:>6}  {r['r_mean']:>9.2f}%  {r['d_mean']:>9.2f}%")
                if runs:
                    avg = avg_results[name][sigma]
                    print(f"  {'':>6}  {'AVG':>6}  {avg['r_mean']:>9.2f}%  {avg['d_mean']:>9.2f}%")

    # ── Print LaTeX table ────────────────────────────────────────
    print("\n\nLaTeX table rows (copy-paste into main.tex):")
    print("-"*70)
    for sigma in NOISE_LEVELS:
        row = f"{sigma}\\%"
        for n in names:
            if sigma in avg_results[n]:
                r = avg_results[n][sigma]
                row += f"  & {r['r_mean']:.2f} & {r['d_mean']:.2f}"
            else:
                row += "  & --- & ---"
        row += " \\\\"
        print(row)

    # ── Total runs ───────────────────────────────────────────────
    total = sum(r['n_runs'] for n in avg_results for s, r in avg_results[n].items())
    print(f"\nTotal training runs completed: {total}")
    print(f"  0%: 4 runs (1 per formulation)")
    print(f"  1%, 5%, 10%: {len(NOISE_SEEDS)} seeds x 4 formulations x 3 levels = {len(NOISE_SEEDS)*4*3} runs")

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'Baseline': '#1f77b4', 'MV': '#ff7f0e', 'MD': '#2ca02c', 'MVD': '#d62728'}

    for n in names:
        available = sorted(avg_results[n].keys())
        if not available:
            continue
        r_vals = [avg_results[n][s]['r_mean'] for s in available]
        d_vals = [avg_results[n][s]['d_mean'] for s in available]

        # Error bars for seeds > 1
        r_err = [avg_results[n][s].get('r_std', 0) for s in available]
        d_err = [avg_results[n][s].get('d_std', 0) for s in available]

        ax.errorbar(available, r_vals, yerr=r_err, fmt='o-', color=colors[n],
                    linewidth=2.5, markersize=9, capsize=4,
                    label=f'{n} reaction ($\\lambda,\\beta$)')
        ax.errorbar(available, d_vals, yerr=d_err, fmt='s--', color=colors[n],
                    linewidth=2.5, markersize=9, capsize=4,
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
    ax.set_title('Inverse parameter recovery under observation noise (5-seed average)',
                 fontsize=14)

    plt.tight_layout()
    plot_path = 'Latex/results/noise_robustness.png'
    plt.savefig(plot_path, dpi=200)
    plt.savefig('extras/noise_robustness.png', dpi=200)
    print(f"\nPlot saved to {plot_path}")
    print(f"Plot also saved to extras/noise_robustness.png")
