"""Check that paired label evaluation leaves an active hybrid trajectory unchanged.

This is a bounded first-inner-loop test, not a convergence-rate benchmark.
Both runs start from the same atomic initialization and stop after three steps.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from check_label_representation import load, read_input, relative


def electronic_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"[A-Z]+\d+", fields[0]) and len(fields) == 5:
            rows.append([float(value) for value in fields[1:4]])
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    base = read_input(args.fixture / "INPUT")
    base.update(dft_functional="hse", deepks_scf="0", deepks_out_labels="1",
                deepks_grad="1", deepks_v_delta="0", cal_force="0", cal_stress="0",
                scf_nmax="3", init_chg="atomic", exx_separate_loop="1")
    for key in ("read_file_dir", "deepks_pair_base", "deepks_pair_target",
                "deepks_out_pair_freq", "deepks_out_base", "deepks_out_freq_elec"):
        base.pop(key, None)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", I_MPI_PIN="0")
    for name, paired in (("ordinary", False), ("paired", True)):
        directory = args.workdir / name
        directory.mkdir(exist_ok=True)
        inp = dict(base, deepks_out_pair=str(int(paired)))
        if paired:
            inp.update(deepks_pair_base="pbe", deepks_pair_target="hse", deepks_out_pair_freq="1")
        if args.analyze_only:
            assert read_input(directory / "INPUT") == inp
            continue
        if (directory / "OUT.ABACUS").exists():
            raise FileExistsError(f"Refusing stale output: {directory}")
        assets = directory / "assets"
        assets.mkdir(exist_ok=True)
        lines = []
        for line in (args.fixture / "STRU").read_text().splitlines():
            fields = line.split()
            for i, word in enumerate(fields):
                path = Path(word)
                if path.suffix.lower() in {".orb", ".upf"}:
                    path = path if path.is_absolute() else args.fixture / path
                    shutil.copy2(path, assets / path.name)
                    fields[i] = str((assets / path.name).resolve())
            lines.append(" ".join(fields))
        (directory / "STRU").write_text("\n".join(lines) + "\n")
        shutil.copy2(args.fixture / "KPT", directory / "KPT")
        (directory / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
            f"{key} {value}\n" for key, value in inp.items()))
        with (directory / "log.scf").open("w") as log:
            subprocess.run(["mpiexec", "-n", "2", str(args.abacus.resolve())],
                           cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
    ordinary = args.workdir / "ordinary"
    paired = args.workdir / "paired"
    a = electronic_rows(ordinary / "log.scf")
    b = electronic_rows(paired / "log.scf")
    assert a.shape == b.shape == (3, 3), (a.shape, b.shape)
    np.testing.assert_array_equal(a, b)
    report = {"steps": len(a), "printed_energy_residual_max_abs": float(np.max(np.abs(a - b)))}
    for stem in ("deepks_etot", "deepks_dm_eig", "deepks_vdrpre_square",
                 "deepks_dot_phialpha_hamilt"):
        a = load(ordinary / "OUT.ABACUS", stem)
        b = load(paired / "OUT.ABACUS", stem)
        report[stem + "_relative"] = relative(b, a)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    kinds = load(paired / "OUT.ABACUS", "deepks_sample_kind")
    report["sample_count"] = len(kinds)
    report["sample_kinds"] = kinds.tolist()
    assert len(kinds) == 3
    (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
