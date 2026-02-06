import argparse
import itertools
import json
import os
import sys
import torch

from transformers.utils import logging as hf_logging

from aries import ARIES
from plm_wrapper import PLMWrapper
from msa_dataset import get_dataset, prepare_ref_eval
from msa_tools import evaluate_msa
from clustal import MSAClustalO, MSAClustalW
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Run a small parameter sweep over test FASTA inputs.")
    p.add_argument("--input-dir", default="./tmp/test_inputs", help="Folder of test FASTA files")
    p.add_argument("--ref-dir", default="./tmp/test_refs", help="Folder of reference alignments")
    p.add_argument("--plms", nargs="+", default=["Rostlab/prot_t5_xl_uniref50"], help="List of PLM names to test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-combos", type=int, default=0, help="Limit number of param combos (0 = no limit)")
    p.add_argument("--out-json", default="./tmp/test_battery_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    hf_logging.set_verbosity_error()
    set_seed(args.seed)

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"input_dir not found: {args.input_dir}")
    if not os.path.isdir(args.ref_dir):
        raise SystemExit(f"ref_dir not found: {args.ref_dir}")

    # Small grid: a few values per parameter
    grid = {
        "plm": args.plms,
        "num_hidden_states": [4],
        "window": [3],
        "reciprocal": [100.0],
        "blur": [2.0],
        "batch": [8],
        "pad_char": ["X"],
        "medoid_mode": ["dnd"],
        "medoid_topk": ["log", 2],
        "sim_metric": ["l2", "cosine"],
        "compare": ["both"],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    if args.max_combos and args.max_combos > 0:
        combos = combos[: args.max_combos]

    # Load dataset with refs
    loader = get_dataset(args.input_dir, include_refs=True, ref_dir=args.ref_dir, max_len=10**9)

    results = {
        "input_dir": args.input_dir,
        "ref_dir": args.ref_dir,
        "device": args.device,
        "grid": grid,
        "failures": [],
        "passes": 0,
    }

    for combo_idx, combo in enumerate(combos, start=1):
        cfg = dict(zip(keys, combo))
        plm_name = cfg.pop("plm")
        compare_mode = cfg.pop("compare")
        print("=" * 100)
        print(f"Combo {combo_idx}/{len(combos)}: plm={plm_name} compare={compare_mode} cfg={cfg}")
        try:
            plm = PLMWrapper(plm_name).to(args.device)
        except Exception as e:
            results["failures"].append({"combo": {"plm": plm_name, "compare": compare_mode, **cfg}, "error": f"PLM load failed: {e}"})
            continue

        config = {
            "plm": plm,
            "aligner": "aries",
            **cfg,
        }
        aries = ARIES(**config)

        clustalo_aligner = MSAClustalO() if compare_mode in ("clustalo", "both") else None
        clustalw_aligner = MSAClustalW() if compare_mode == "both" else None

        with torch.no_grad():
            for i, batch in enumerate(loader):
                item = batch[0]
                msa_name = item[0]
                ungapped = item[2]
                ref_msa = item[3] if len(item) > 3 else None
                ref_indices = item[4] if len(item) > 4 else None
                ref_mask = item[5] if len(item) > 5 else None

                try:
                    alns, _ = aries.align(seqs=ungapped, msa_name=msa_name)
                    if clustalo_aligner is not None:
                        clustalo_aln = clustalo_aligner.align(seqs=ungapped, aln_name=msa_name)
                    if clustalw_aligner is not None:
                        clustalw_aln = clustalw_aligner.align(seqs=ungapped, aln_name=msa_name)

                    if ref_msa:
                        eval_alns = prepare_ref_eval(alns, ref_indices, ref_mask)
                        _ = evaluate_msa(eval_alns, ref_msa)
                        if clustalo_aligner is not None:
                            _ = evaluate_msa(prepare_ref_eval(clustalo_aln, ref_indices, ref_mask), ref_msa)
                        if clustalw_aligner is not None:
                            _ = evaluate_msa(prepare_ref_eval(clustalw_aln, ref_indices, ref_mask), ref_msa)
                except Exception as e:
                    results["failures"].append({
                        "combo": {"plm": plm_name, "compare": compare_mode, **cfg},
                        "msa": msa_name,
                        "error": str(e),
                    })
                    print(f"FAILED {msa_name}: {e}")
                    break
                else:
                    results["passes"] += 1

        del aries, plm
        torch.cuda.empty_cache()

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    if results["failures"]:
        print(f"Failures: {len(results['failures'])}")
        sys.exit(1)

    print(f"All tests passed. Total passes: {results['passes']}")


if __name__ == "__main__":
    main()
