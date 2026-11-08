"""
Run all four PINN formulations (forward + inverse) over 5 random seeds
and produce the data for Table 7 (tab:summary_form).

Usage (from project root):
    python extras/run_summary_table.py              # run everything
    python extras/run_summary_table.py --inverse     # inverse only (skip forward)
    python extras/run_summary_table.py --forward     # forward only (skip inverse)

Results are saved to extras/summary_table_results.json after each formulation.
"""
import subprocess, sys, re, os, json, argparse, textwrap, time
import numpy as np

SEEDS = [1234, 5678, 0, 6767, 9999]
SEED_LABELS = ['1234', '5678', '0000', '6767', '9999']

FORMULATIONS = [
    {
        'name': 'Baseline',
        'unknowns': 4,
        'forward_script': 'Fire_PINN_2D/pinn_2D.py',
        'inverse_script': 'Fire_PINN_2D/pinn_inverse_2D.py',
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
        'unknowns': 10,
        'forward_script': 'Fire_PINN_Multi_variable/pinn_multi_variable.py',
        'inverse_script': 'Fire_PINN_Multi_variable/pinn_inverse_multi_variable.py',
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
        'unknowns': 10,
        'forward_script': 'Fire_PINN_Multi_diffusion/pinn_multi_diffusion.py',
        'inverse_script': 'Fire_PINN_Multi_diffusion/pinn_inverse_multi_diffusion.py',
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
        'unknowns': 16,
        'forward_script': 'Fire_PINN_Multi_variable_diffusion/pinn__multi_var_dif.py',
        'inverse_script': 'Fire_PINN_Multi_variable_diffusion/pinn_inverse_multi_var_dif.py',
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

# Injected into inverse scripts after _pairs: prints per-param details + summary, then exits
INVERSE_TAIL = textwrap.dedent("""\
    import sys as _s
    _reaction  = [abs(e-t)/t*100 for n,t,e in _pairs if n.startswith(('lambda','beta'))]
    _diffusion = [abs(e-t)/t*100 for n,t,e in _pairs if n.startswith('w')]
    for _n,_t,_e in _pairs:
        _err = abs(_e-_t)/_t*100
        print(f"##PARAM## {_n}: true={_t:.6f}  estimated={_e:.6f}  error={_err:.4f}")
    print(f"##RESULT## r_mean={sum(_reaction)/len(_reaction):.4f} "
          f"r_max={max(_reaction):.4f} "
          f"d_mean={sum(_diffusion)/len(_diffusion):.4f} "
          f"d_max={max(_diffusion):.4f}")
    _s.exit(0)
""")

# Injected into forward scripts: just exit before visualization
FORWARD_TAIL = textwrap.dedent("""\
    import sys as _s
    _s.exit(0)
""")

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(PROJ_ROOT, 'extras', 'summary_table_results.json')


def patch_source(src, seed):
    """Patch seed, disable GUI matplotlib, suppress plt.show."""
    src = f"import matplotlib; matplotlib.use('Agg')\n" + src
    src = re.sub(r"np\.random\.seed\(\d+\)", f"np.random.seed({seed})", src)
    src = re.sub(r"torch\.manual_seed\(\d+\)", f"torch.manual_seed({seed})", src)
    src = src.replace("plt.show()", "pass")
    src = src.replace('\u03bb', 'lambda').replace('\u03b2', 'beta')
    return src


def build_forward_wrapper(script_path, seed):
    with open(script_path, 'r', encoding='utf-8') as f:
        src = f.read()
    src = patch_source(src, seed)
    marker = "#------------------------ Visualization -----------------------------"
    src = src.replace(marker, FORWARD_TAIL + marker, 1)
    return src


def build_inverse_wrapper(script_path, seed, param_snippet):
    with open(script_path, 'r', encoding='utf-8') as f:
        src = f.read()
    src = patch_source(src, seed)
    marker = "#------------------------ Visualization -----------------------------"
    src = src.replace(marker, param_snippet + INVERSE_TAIL + marker, 1)
    return src


def run_script(wrapper_src, tmp_name):
    """Write wrapper to temp file, run it, return stdout/stderr."""
    tmp = os.path.join(PROJ_ROOT, tmp_name)
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(wrapper_src)
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, encoding='utf-8',
            errors='replace', timeout=7200, env=env,
            cwd=PROJ_ROOT,
        )
        return proc.stdout, proc.stderr, proc.returncode
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def parse_rel_l2(stdout):
    """Parse global Relative L2 errors for T and S from stdout."""
    t_err, s_err = None, None
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if 'Relative L2 errors' in line:
            for j in range(i+1, min(i+4, len(lines))):
                m_t = re.search(r'Temperature:\s+([\d.eE+\-]+)', lines[j])
                m_s = re.search(r'Fuel:\s+([\d.eE+\-]+)', lines[j])
                if m_t:
                    t_err = float(m_t.group(1))
                if m_s:
                    s_err = float(m_s.group(1))
            break
    return t_err, s_err


def parse_inverse_results(stdout):
    """Parse per-parameter errors and summary from inverse stdout."""
    params = []
    for line in stdout.splitlines():
        if '##PARAM##' in line:
            payload = line.split('##PARAM##')[1].strip()
            m = re.match(r'(\S+):\s+true=([\d.eE+\-]+)\s+estimated=([\d.eE+\-]+)\s+error=([\d.eE+\-]+)', payload)
            if m:
                params.append({
                    'name': m.group(1),
                    'true': float(m.group(2)),
                    'estimated': float(m.group(3)),
                    'error_pct': float(m.group(4)),
                })

    summary = {}
    m = re.search(r'##RESULT##\s*(.+)', stdout)
    if m:
        tokens = dict(tok.split('=', 1) for tok in m.group(1).strip().split())
        summary = {k: float(v) for k, v in tokens.items()}

    return params, summary


def save_results(all_results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def print_summary_table(all_results):
    """Print summary matching Table 7 format, plus LaTeX snippet."""
    print("\n\n" + "=" * 110)
    print("TABLE 7 -- Summary across formulations (5-seed averages)")
    print("=" * 110)

    header = (f"{'Formulation':<14} {'Unk':>3}"
              f"  {'React mean+/-std':>16} {'max':>6}"
              f"  {'Diff mean+/-std':>16} {'max':>6}"
              f"  {'Fwd T~':>10} {'Fwd S~':>10}"
              f"  {'Inv T~':>10} {'Inv S~':>10}")
    print(header)
    print("-" * 110)

    latex_rows = []

    for form in FORMULATIONS:
        name = form['name']
        unk = form['unknowns']
        data = all_results.get(name, {})

        inv_data = data.get('inverse', {})
        fwd_data = data.get('forward', {})

        # Reaction and diffusion from inverse seeds
        r_means = [s['r_mean'] for s in inv_data.get('seeds', []) if 'r_mean' in s]
        r_maxes = [s['r_max'] for s in inv_data.get('seeds', []) if 'r_max' in s]
        d_means = [s['d_mean'] for s in inv_data.get('seeds', []) if 'd_mean' in s]
        d_maxes = [s['d_max'] for s in inv_data.get('seeds', []) if 'd_max' in s]

        # Forward field errors
        fwd_T = [s['rel_l2_T'] for s in fwd_data.get('seeds', []) if s.get('rel_l2_T') is not None]
        fwd_S = [s['rel_l2_S'] for s in fwd_data.get('seeds', []) if s.get('rel_l2_S') is not None]

        # Inverse field errors
        inv_T = [s['rel_l2_T'] for s in inv_data.get('seeds', []) if s.get('rel_l2_T') is not None]
        inv_S = [s['rel_l2_S'] for s in inv_data.get('seeds', []) if s.get('rel_l2_S') is not None]

        # Compute stats
        r_m = f"{np.mean(r_means):.2f} +/- {np.std(r_means):.2f}" if r_means else "N/A"
        r_x = f"{max(r_maxes):.1f}" if r_maxes else "N/A"
        d_m = f"{np.mean(d_means):.2f} +/- {np.std(d_means):.2f}" if d_means else "N/A"
        d_x = f"{max(d_maxes):.1f}" if d_maxes else "N/A"
        ft = f"{np.mean(fwd_T):.2e}" if fwd_T else "N/A"
        fs = f"{np.mean(fwd_S):.2e}" if fwd_S else "N/A"
        it = f"{np.mean(inv_T):.2e}" if inv_T else "N/A"
        is_ = f"{np.mean(inv_S):.2e}" if inv_S else "N/A"

        print(f"{name:<14} {unk:>3}  {r_m:>16} {r_x:>6}  {d_m:>16} {d_x:>6}"
              f"  {ft:>10} {fs:>10}  {it:>10} {is_:>10}")

        # Build LaTeX row
        if r_means:
            rm_val, rs_val = np.mean(r_means), np.std(r_means)
            dm_val, ds_val = np.mean(d_means), np.std(d_means)
            rx_val = max(r_maxes)
            dx_val = max(d_maxes)
        else:
            rm_val = rs_val = dm_val = ds_val = rx_val = dx_val = 0

        def sci(v):
            """Format as LaTeX scientific notation."""
            if v == 0:
                return "N/A"
            exp = int(np.floor(np.log10(abs(v))))
            coeff = v / 10**exp
            return f"${coeff:.2f}\\times10^{{{exp}}}$"

        ft_val = np.mean(fwd_T) if fwd_T else 0
        fs_val = np.mean(fwd_S) if fwd_S else 0
        it_val = np.mean(inv_T) if inv_T else 0
        is_val = np.mean(inv_S) if inv_S else 0

        latex_name = {'Baseline': 'Baseline', 'MV': 'Multi-Variable (MV)',
                      'MD': 'Multi-Diffusion (MD)', 'MVD': 'Combined (MVD)'}[name]
        latex_rows.append(
            f"{latex_name:<25} & {unk:>2} "
            f"& ${rm_val:.2f} \\pm {rs_val:.2f}$ & {rx_val:.1f} "
            f"& ${dm_val:.2f} \\pm {ds_val:.2f}$ & {dx_val:.1f} "
            f"& {sci(ft_val)} & {sci(fs_val)} "
            f"& {sci(it_val)} & {sci(is_val)} \\\\"
        )

    print("-" * 110)

    # Also print per-seed breakdown
    print("\n\nPER-SEED BREAKDOWN (inverse)")
    print("=" * 90)
    print(f"{'Seed':>6}", end='')
    for form in FORMULATIONS:
        n = form['name']
        print(f"  {n+' r%':>8} {n+' d%':>8}", end='')
    print()
    print("-" * 90)

    for i, label in enumerate(SEED_LABELS):
        print(f"{label:>6}", end='')
        for form in FORMULATIONS:
            inv = all_results.get(form['name'], {}).get('inverse', {}).get('seeds', [])
            if i < len(inv):
                s = inv[i]
                print(f"  {s.get('r_mean', 0):>8.2f} {s.get('d_mean', 0):>8.2f}", end='')
            else:
                print(f"  {'N/A':>8} {'N/A':>8}", end='')
        print()

    print("-" * 90)
    for stat_name, fn in [('Mean', np.mean), ('Std', np.std)]:
        print(f"{stat_name:>6}", end='')
        for form in FORMULATIONS:
            inv = all_results.get(form['name'], {}).get('inverse', {}).get('seeds', [])
            if inv:
                rm = [s['r_mean'] for s in inv if 'r_mean' in s]
                dm = [s['d_mean'] for s in inv if 'd_mean' in s]
                print(f"  {fn(rm):>8.2f} {fn(dm):>8.2f}", end='')
            else:
                print(f"  {'N/A':>8} {'N/A':>8}", end='')
        print()

    # LaTeX output
    print("\n\nLATEX SNIPPET FOR TABLE 7")
    print("=" * 90)
    print("\\begin{tabular}{lrrrrrrrrrr}")
    print("\\toprule")
    print("& & \\multicolumn{2}{c}{Reaction ($\\lambda,\\beta$)} & \\multicolumn{2}{c}{Diffusion ($w$)}")
    print("& \\multicolumn{2}{c}{Forward} & \\multicolumn{2}{c}{Inverse} \\\\")
    print("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}")
    print("Formulation & Unknowns & mean & max & mean & max & $\\tilde{T}$ & $\\tilde{S}$ & $\\tilde{T}$ & $\\tilde{S}$ \\\\")
    print("\\midrule")
    for row in latex_rows:
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description='Run PINN seed study for summary table.')
    parser.add_argument('--inverse', action='store_true', help='Run inverse only')
    parser.add_argument('--forward', action='store_true', help='Run forward only')
    parser.add_argument('--table', action='store_true', help='Just print table from saved results')
    args = parser.parse_args()

    run_fwd = True
    run_inv = True
    if args.inverse:
        run_fwd = False
    if args.forward:
        run_inv = False
    if args.table:
        run_fwd = False
        run_inv = False

    all_results = load_results()

    total_runs = sum([
        (5 if run_fwd else 0) + (5 if run_inv else 0)
        for _ in FORMULATIONS
    ])
    run_count = 0
    t_start_all = time.time()

    for form in FORMULATIONS:
        name = form['name']
        unk = form['unknowns']
        if name not in all_results:
            all_results[name] = {'unknowns': unk, 'forward': {'seeds': []}, 'inverse': {'seeds': []}}

        print(f"\n{'=' * 60}")
        print(f"  {name}  ({unk} unknowns)")
        print(f"{'=' * 60}")

        # ── Forward ──
        if run_fwd:
            all_results[name]['forward'] = {'seeds': []}
            for seed, label in zip(SEEDS, SEED_LABELS):
                run_count += 1
                print(f"\n  [Forward] Seed {label}  ({run_count}/{total_runs})")
                t0 = time.time()

                wrapper = build_forward_wrapper(
                    os.path.join(PROJ_ROOT, form['forward_script']), seed
                )
                stdout, stderr, rc = run_script(wrapper, f"_tmp_fwd_{name}_{seed}.py")
                elapsed = time.time() - t0

                rel_T, rel_S = parse_rel_l2(stdout)
                seed_result = {
                    'seed': label,
                    'rel_l2_T': rel_T,
                    'rel_l2_S': rel_S,
                    'returncode': rc,
                    'elapsed_s': round(elapsed, 1),
                }
                all_results[name]['forward']['seeds'].append(seed_result)

                if rel_T is not None:
                    print(f"    T~ = {rel_T:.6e}   S~ = {rel_S:.6e}   ({fmt_time(elapsed)})")
                else:
                    print(f"    ERROR (rc={rc})  ({fmt_time(elapsed)})")
                    if stderr:
                        print(f"    stderr: {stderr[-300:]}")

                save_results(all_results)

        # ── Inverse ──
        if run_inv:
            all_results[name]['inverse'] = {'seeds': []}
            for seed, label in zip(SEEDS, SEED_LABELS):
                run_count += 1
                print(f"\n  [Inverse] Seed {label}  ({run_count}/{total_runs})")
                t0 = time.time()

                wrapper = build_inverse_wrapper(
                    os.path.join(PROJ_ROOT, form['inverse_script']),
                    seed, form['params']
                )
                stdout, stderr, rc = run_script(wrapper, f"_tmp_inv_{name}_{seed}.py")
                elapsed = time.time() - t0

                rel_T, rel_S = parse_rel_l2(stdout)
                params, summary = parse_inverse_results(stdout)

                seed_result = {
                    'seed': label,
                    'rel_l2_T': rel_T,
                    'rel_l2_S': rel_S,
                    'r_mean': summary.get('r_mean'),
                    'r_max': summary.get('r_max'),
                    'd_mean': summary.get('d_mean'),
                    'd_max': summary.get('d_max'),
                    'params': params,
                    'returncode': rc,
                    'elapsed_s': round(elapsed, 1),
                }
                all_results[name]['inverse']['seeds'].append(seed_result)

                if summary:
                    print(f"    react={summary.get('r_mean', 0):.2f}%  diff={summary.get('d_mean', 0):.2f}%"
                          f"  T~={rel_T:.6e}  S~={rel_S:.6e}" if rel_T else
                          f"    react={summary.get('r_mean', 0):.2f}%  diff={summary.get('d_mean', 0):.2f}%"
                          f"  (field L2 not captured)")
                    print(f"    ({fmt_time(elapsed)})")
                    for p in params:
                        print(f"      {p['name']:>10}: true={p['true']:.6f}  est={p['estimated']:.6f}  err={p['error_pct']:.2f}%")
                else:
                    print(f"    ERROR (rc={rc})  ({fmt_time(elapsed)})")
                    if stderr:
                        print(f"    stderr: {stderr[-300:]}")

                save_results(all_results)

    total_time = time.time() - t_start_all
    if run_fwd or run_inv:
        print(f"\n\nTotal time: {fmt_time(total_time)}")

    print_summary_table(all_results)


if __name__ == '__main__':
    main()
