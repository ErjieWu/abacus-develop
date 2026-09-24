"""Check default base/model paired samples against the loaded TorchScript model.

--fixture is a completed ordinary gradient SCF directory with INPUT/STRU/KPT.
The test copies its assets/model, enables paired output without functional names,
and verifies passive SCF invariance, E_sample=E_model and b_sample=M*g_model.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from check_label_representation import load, read_input, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--explicit-none", action="store_true",
                        help="Exercise case-insensitive None/NONE for the default model pair")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    inp = read_input(args.fixture / "INPUT")
    assert inp["deepks_scf"] == "1"
    inp.update(deepks_out_pair="1", deepks_out_pair_freq="5", deepks_grad="1", deepks_v_delta="0")
    inp.pop("deepks_pair_base", None)
    inp.pop("deepks_pair_target", None)
    if args.explicit_none:
        inp.update(deepks_pair_base="None", deepks_pair_target="NONE")
    shutil.copytree(args.assets, args.workdir / "assets", dirs_exist_ok=True, symlinks=False)
    shutil.copy2(inp["deepks_model"], args.workdir / "model.ptg")
    inp["deepks_model"] = str((args.workdir / "model.ptg").resolve())
    output = args.workdir / "OUT.ABACUS"
    if args.analyze_only:
        assert read_input(args.workdir / "INPUT") == inp
    else:
        if output.exists():
            raise FileExistsError(f"Refusing stale output: {output}")
        stru = (args.fixture / "STRU").read_text().replace(
            str(args.assets.resolve()), str((args.workdir / "assets").resolve()))
        (args.workdir / "STRU").write_text(stru)
        shutil.copy2(args.fixture / "KPT", args.workdir / "KPT")
        (args.workdir / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
            f"{key} {value}\n" for key, value in inp.items()))
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", I_MPI_PIN="0")
        with (args.workdir / "log.scf").open("w") as log:
            subprocess.run(["mpiexec", "-n", "2", str(args.abacus.resolve())],
                           cwd=args.workdir, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1200)
    assert "#SCF IS CONVERGED#" in (output / "running_scf.log").read_text()
    report = {}
    for name in ("deepks_etot", "deepks_dm_eig", "deepks_ftot", "deepks_stot"):
        error = float(np.max(np.abs(load(output, name) - load(args.fixture / "OUT.ABACUS", name))))
        report[name + "_passive_max_abs"] = error
        assert error < 1e-10, (name, error)

    descriptors = load(output, "deepks_sample_dm_eig")
    metrics = np.load(output / "deepks_sample_hr_metric.npy", mmap_mode="r", allow_pickle=False)
    projections = load(output, "deepks_sample_hr_projection")
    energies = load(output, "deepks_sample_energy").reshape(-1)
    model = torch.jit.load(inp["deepks_model"], map_location="cpu").eval()
    errors = []
    energy_errors = []
    for i, d in enumerate(descriptors):
        descriptor = torch.tensor(d[None], dtype=torch.float64, requires_grad=True)
        energy = model(descriptor).sum()
        gradient, = torch.autograd.grad(energy, descriptor)
        expected = metrics[i] @ gradient.detach().numpy().reshape(-1)
        errors.append(relative(projections[i].reshape(-1), expected))
        energy_errors.append(abs(float(energy.detach()) - energies[i]))
    report["sample_count"] = len(descriptors)
    report["projection_relative_max"] = max(errors)
    report["energy_max_abs"] = max(energy_errors)
    assert report["sample_count"] > 1
    assert report["projection_relative_max"] < 1e-12
    assert report["energy_max_abs"] < 1e-12
    np.testing.assert_allclose(descriptors[-1], load(output, "deepks_dm_eig"), rtol=0, atol=1e-12)
    correction = float((load(output, "deepks_etot") - load(output, "deepks_ebase")).item())
    report["final_energy_correction_max_abs"] = abs(energies[-1] - correction)
    assert report["final_energy_correction_max_abs"] < 1e-12
    (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
