"""
Run all four inverse PINN formulations over 5 seeds.
Does NOT modify the original scripts — patches seed + matplotlib at runtime.

Run from the project root:  python run_inverse_seeds.py
"""
import subprocess, sys, re, os, textwrap
import numpy as np

SEEDS = [1234, 5678, 0, 6767, 9999]
SEED_LABELS = ['1234', '5678', '0000', '6767', '9999']

FORMULATIONS = [
    {
        'name': 'Baseline',
        'script': 'Fire_PINN_2D/pinn_inverse_2D.py',
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

# Injected after _pairs is built: prints per-param details + summary line, then exits
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


def build_wrapper(script_path, seed, param_snippet):
    with open(script_path, 'r', encoding='utf-8') as f:
        src = f.read()
    src = f"import matplotlib; matplotlib.use('Agg')\n" + src
    src = re.sub(r"np\.random\.seed\(\d+\)", f"np.random.seed({seed})", src)
    src = re.sub(r"torch\.manual_seed\(\d+\)", f"torch.manual_seed({seed})", src)
    src = src.replace("plt.show()", "pass")
    # Replace Greek Unicode chars with ASCII to avoid cp1252 crash on Windows
    src = src.replace('\u03bb', 'lambda').replace('\u03b2', 'beta')
    marker = "#------------------------ Visualization -----------------------------"
    src = src.replace(marker, param_snippet + RESULT_TAIL + marker, 1)
    return src


# ── Run ───────────────────────────────────────────────────────────
results = {f['name']: [] for f in FORMULATIONS}

for form in FORMULATIONS:
    name = form['name']
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)

    for seed, label in zip(SEEDS, SEED_LABELS):
        print(f"\n  --- Seed {label} ---")
        wrapper_src = build_wrapper(form['script'], seed, form['params'])
        tmp = f"_tmp_{name}_{seed}.py"
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
            # Print each parameter: true vs estimated
            for line in proc.stdout.splitlines():
                if '##PARAM##' in line:
                    print(f"    {line.split('##PARAM##')[1].strip()}")

            match = re.search(r'##RESULT##\s*(.+)', proc.stdout)
            if match:
                d = dict(tok.split('=', 1) for tok in match.group(1).strip().split())
                r = {
                    'label':  label,
                    'r_mean': float(d['r_mean']),
                    'd_mean': float(d['d_mean']),
                }
                results[name].append(r)
                print(f"  => mean reaction err={r['r_mean']:.2f}%  mean diffusion err={r['d_mean']:.2f}%")
            else:
                print("  ERROR")
                print(proc.stdout[-500:])
                print(proc.stderr[-300:])
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

# ── Summary ───────────────────────────────────────────────────────
names = [f['name'] for f in FORMULATIONS]
print("\n\n" + "="*60)
print("SUMMARY — mean relative error (%)")
print("="*60)
print(f"{'Seed':>6}", end='')
for n in names:
    print(f"  {n+' r':>10}  {n+' d':>10}", end='')
print()
print("-"*60)

for i, label in enumerate(SEED_LABELS):
    print(f"{label:>6}", end='')
    for n in names:
        if i < len(results[n]):
            r = results[n][i]
            print(f"  {r['r_mean']:>10.2f}  {r['d_mean']:>10.2f}", end='')
    print()

print("-"*60)
for stat, fn in [('Mean', np.mean), ('Std', np.std)]:
    print(f"{stat:>6}", end='')
    for n in names:
        if results[n]:
            print(f"  {fn([r['r_mean'] for r in results[n]]):>10.2f}"
                  f"  {fn([r['d_mean'] for r in results[n]]):>10.2f}", end='')
    print()
