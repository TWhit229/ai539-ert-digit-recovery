"""Driver for the full 12-combination sweep.

Axes (3 × 2 × 2 = 12):
  PRIOR    : base   (existing 1M-param rot15)
             bigger (6M-param UNet, EMA, rot15)
             cond   (6M-param class-conditional UNet with CFG, EMA, rot15)
  SCHEDULE : geomspace  (current default)
             neg_rho    (DAPS++ negative-rho, more iters at low noise)
  POLISH   : lm  (current GN-CG polish to float64 floor)
             tv  (TV-regularized polish)

Output layout:
  results/
    pre_polish/<prior>_<schedule>.mat       (best-of-N chains BEFORE polish)
    polished/<prior>_<schedule>_<polish>.mat (final answer for that combo)
    sweep_summary.csv                        (all 12 misfits, runtimes, paths)
    sweep_summary.json                       (same data, machine-readable)

Use --combos to restrict, e.g.:
    --combos base.geomspace.lm,bigger.neg_rho.tv
to run only those two combos. Default is all 12.

The script assumes the bigger and cond DDPMs have been trained already
(via train_bigger_ddpm.py and train_cond_ddpm.py). It will skip combos
whose required prior dir is missing, with a warning.
"""
import argparse, os, sys, time, json, csv, traceback
sys.path.insert(0, "../code")

import numpy as np
import torch
from scipy.io import loadmat, savemat
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler

from stage_a_daps_warmstart import ERTSetup, x_diff_to_pix
from gn_cg_polish import gn_cg_polish
from pnp_dm_v2 import pnp_dm_v2
from pnp_dm_cfg import pnp_dm_cfg, load_cond_ddpm
from tv_polish import tv_polish


PRIORS    = ["base", "bigger", "cond"]
SCHEDULES = ["geomspace", "neg_rho"]
POLISHES  = ["lm", "tv"]


def all_combos():
    return [(p, s, pol) for p in PRIORS for s in SCHEDULES for pol in POLISHES]


def install_prior(setup, prior_name, base_dir, bigger_dir, cond_dir):
    """Swap setup.unet/scheduler/alphas_cumprod to the requested prior.
    For 'cond' we return the unet separately because the call signature
    differs (needs class_labels).
    Returns: ('std', None) for base/bigger,
             ('cond', cond_unet) for cond.
    """
    if prior_name == "base":
        path = base_dir
    elif prior_name == "bigger":
        path = bigger_dir
    elif prior_name == "cond":
        if not os.path.isdir(cond_dir):
            raise FileNotFoundError(f"cond DDPM dir missing: {cond_dir}")
        cond_unet, sch = load_cond_ddpm(cond_dir)
        setup.scheduler = sch
        setup.alphas_cumprod = sch.alphas_cumprod.float()
        return ("cond", cond_unet)
    else:
        raise ValueError(prior_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"prior dir missing: {path}")
    pipe = DDPMPipeline.from_pretrained(path)
    setup.unet = pipe.unet.eval()
    setup.scheduler = pipe.scheduler
    setup.alphas_cumprod = pipe.scheduler.alphas_cumprod.float()
    for p_ in setup.unet.parameters():
        p_.requires_grad_(False)
    return ("std", None)


def run_pre_polish(setup, prior_name, schedule_name, cond_unet,
                   n_chains, n_iter, sigma_n, cfg_w, seed):
    """Returns dict with 'x_pix', 'final_misfit', 'extra' for the best chain."""
    if prior_name == "cond":
        chains = pnp_dm_cfg(setup, cond_unet, setup.alphas_cumprod,
                             schedule_name=schedule_name,
                             n_chains=n_chains, n_iter=n_iter,
                             sigma_n=sigma_n, cfg_w=cfg_w, seed=seed,
                             verbose=False)
        best = chains[0]
        return {'x_pix': best['x_pix'],
                'final_misfit': best['final_misfit'],
                'extra': {'class_label': int(best['class_label']),
                          'cfg_w': float(cfg_w)}}
    chains = pnp_dm_v2(setup, schedule_name=schedule_name,
                        n_chains=n_chains, n_iter=n_iter,
                        sigma_n=sigma_n, seed=seed, verbose=False)
    best = chains[0]
    return {'x_pix': best['x_pix'],
            'final_misfit': best['final_misfit'],
            'extra': {}}


def run_polish(setup, polish_name, x_init, K_lm=100, K_tv=120, tau=1e-5):
    """Returns dict with 'x_pix', 'final_misfit'."""
    # GN-CG and TV polishes both want x in pixel coords [0, ~1+]
    if polish_name == "lm":
        x_pol, log = gn_cg_polish(setup, x_init, K=K_lm, n_cg=60,
                                   lam_init=1e-4, lam_min=1e-10,
                                   target_misfit=1e-15, verbose=False)
        misfit = log[-1][1]
    elif polish_name == "tv":
        x_pol, log = tv_polish(setup, x_init, K=K_tv, n_cg=60,
                                tau=tau, target_misfit=1e-15, verbose=False)
        misfit = log[-1][1]
    else:
        raise ValueError(polish_name)
    return {'x_pix': x_pol, 'final_misfit': float(misfit)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir",   default="../code/ddpm_mnist_rot15")
    p.add_argument("--bigger_dir", default="./ddpm_bigger_rot15_ema")
    p.add_argument("--cond_dir",   default="./ddpm_cond_rot15_ema")
    p.add_argument("--n_chains",   type=int,   default=32)
    p.add_argument("--n_iter",     type=int,   default=80)
    p.add_argument("--sigma_n",    type=float, default=1e-4)
    p.add_argument("--cfg_w",      type=float, default=3.0)
    p.add_argument("--K_lm",       type=int,   default=100)
    p.add_argument("--K_tv",       type=int,   default=120)
    p.add_argument("--tau",        type=float, default=1e-5,
                   help="TV polish weight")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--out_dir",    default="./results")
    p.add_argument("--combos",     default="",
                   help="comma list of prior.schedule.polish triples to run "
                        "(default = all 12). Example: base.geomspace.lm,cond.neg_rho.tv")
    args = p.parse_args()

    os.makedirs(f"{args.out_dir}/pre_polish", exist_ok=True)
    os.makedirs(f"{args.out_dir}/polished", exist_ok=True)

    if args.combos.strip():
        wanted = set(tuple(c.split(".")) for c in args.combos.split(","))
    else:
        wanted = set(all_combos())

    print(f"Sweep: {len(wanted)} combo(s)")
    for c in sorted(wanted):
        print(f"  - {'.'.join(c)}")

    # Cache pre-polish runs by (prior, schedule) so we don't rerun PnP-DM
    # for the 2 polish variants of the same (prior, schedule).
    pre_polish_cache = {}
    summary = []

    setup = ERTSetup("../code")
    try:
        for prior_name in PRIORS:
            relevant = [c for c in wanted if c[0] == prior_name]
            if not relevant:
                continue

            # Install prior
            try:
                kind, cond_unet = install_prior(setup, prior_name,
                                                 args.base_dir, args.bigger_dir,
                                                 args.cond_dir)
            except FileNotFoundError as e:
                print(f"\n[skip prior '{prior_name}']: {e}")
                continue
            print(f"\n=== prior = {prior_name} ({'CFG' if kind=='cond' else 'std'}) ===")

            for schedule_name in SCHEDULES:
                relevant_s = [c for c in relevant if c[1] == schedule_name]
                if not relevant_s:
                    continue
                key = (prior_name, schedule_name)
                print(f"\n  -- PnP-DM: schedule={schedule_name} --")
                t0 = time.time()
                try:
                    pre = run_pre_polish(setup, prior_name, schedule_name,
                                          cond_unet,
                                          args.n_chains, args.n_iter,
                                          args.sigma_n, args.cfg_w, args.seed)
                except Exception as e:
                    print(f"    PnP-DM failed: {e}")
                    traceback.print_exc()
                    continue
                dt = time.time() - t0
                print(f"    pre-polish misfit {pre['final_misfit']:.3e}  "
                      f"({dt:.0f}s)")
                pp_path = f"{args.out_dir}/pre_polish/{prior_name}_{schedule_name}.mat"
                savemat(pp_path, {
                    'x_answer':     pre['x_pix'],
                    'sigma_answer': 1.0 + pre['x_pix'],
                    'final_misfit': float(pre['final_misfit']),
                    'prior':        prior_name,
                    'schedule':     schedule_name,
                    'runtime_s':    float(dt),
                    **pre['extra'],
                })
                pre_polish_cache[key] = pre

                for polish_name in POLISHES:
                    if (prior_name, schedule_name, polish_name) not in wanted:
                        continue
                    print(f"\n    -- polish: {polish_name} --")
                    t1 = time.time()
                    try:
                        pol = run_polish(setup, polish_name, pre['x_pix'],
                                          K_lm=args.K_lm, K_tv=args.K_tv,
                                          tau=args.tau)
                    except Exception as e:
                        print(f"      polish failed: {e}")
                        traceback.print_exc()
                        continue
                    dt2 = time.time() - t1
                    print(f"      polished misfit {pol['final_misfit']:.3e}  "
                          f"({dt2:.0f}s)")
                    out_path = (f"{args.out_dir}/polished/"
                                f"{prior_name}_{schedule_name}_{polish_name}.mat")
                    savemat(out_path, {
                        'x_answer':     pol['x_pix'],
                        'sigma_answer': 1.0 + pol['x_pix'],
                        'final_misfit': float(pol['final_misfit']),
                        'pre_polish_misfit': float(pre['final_misfit']),
                        'prior':    prior_name,
                        'schedule': schedule_name,
                        'polish':   polish_name,
                        'runtime_pre_s':    float(dt),
                        'runtime_polish_s': float(dt2),
                        **pre['extra'],
                    })
                    summary.append({
                        'prior':    prior_name,
                        'schedule': schedule_name,
                        'polish':   polish_name,
                        'pre_polish_misfit': float(pre['final_misfit']),
                        'final_misfit':      float(pol['final_misfit']),
                        'runtime_pre_s':     float(dt),
                        'runtime_polish_s':  float(dt2),
                        'out_path':          out_path,
                        **pre['extra'],
                    })

    finally:
        setup.quit()

    # Save summary
    csv_path  = f"{args.out_dir}/sweep_summary.csv"
    json_path = f"{args.out_dir}/sweep_summary.json"
    if summary:
        keys = sorted({k for r in summary for k in r.keys()})
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary:
                w.writerow({k: r.get(k, '') for k in keys})
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary → {csv_path}")
        print(f"Summary → {json_path}")
        print("\nFinal misfits (sorted):")
        for r in sorted(summary, key=lambda r: r['final_misfit']):
            tag = f"{r['prior']}.{r['schedule']}.{r['polish']}"
            print(f"  {tag:28s}  pre {r['pre_polish_misfit']:.2e}  "
                  f"polished {r['final_misfit']:.2e}")
    else:
        print("\nNo combos completed.")


if __name__ == "__main__":
    main()
