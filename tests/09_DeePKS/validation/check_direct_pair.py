"""Validate paired direct HR/Hk and precalculations against gradient labels.

Two completed gradient SCFs are required: an explicit named pair and a default
base/model pair. Optional --periodic-fixture exercises genuinely complex H(k)
on a one-step periodic density with a small analytic TorchScript model.
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
from deepks.physics.backends.abacus.utils import read_csr
from deepks.physics.properties.v_delta import v_delta_from_context
from deepks.physics.properties.vdr import vdr_from_context


def align(a, b):
    shape = tuple(max(a.shape[i], b.shape[i]) for i in range(3))
    def pad(x):
        return np.pad(x, [(0, shape[i] - x.shape[i]) for i in range(3)] + [(0, 0)] * (x.ndim - 3))
    return pad(a), pad(b)


class AnalyticCorrection(torch.nn.Module):
    def forward(self, descriptor):
        return 0.001 * descriptor.square().sum(dim=(-1, -2)).unsqueeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-reference", type=Path, required=True)
    parser.add_argument("--model-reference", type=Path, required=True)
    parser.add_argument("--periodic-fixture", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="validate existing cases and run missing cases")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    report = {}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", I_MPI_PIN="0")

    def run(name, reference, selection, periodic=False):
        directory = args.workdir / name
        directory.mkdir(exist_ok=True)
        inp = read_input(reference / "INPUT")
        inp.update(deepks_out_pair="1", deepks_grad="0", deepks_v_delta=str(selection))
        if periodic:
            inp.update(scf_nmax="1", deepks_scf="1", cal_force="0", cal_stress="0")
            inp.pop("deepks_pair_base", None)
            inp.pop("deepks_pair_target", None)
        reuse = args.analyze_only or (args.resume and (directory / "OUT.ABACUS/running_scf.log").exists())
        if not reuse:
            if (directory / "OUT.ABACUS").exists():
                raise FileExistsError(f"Refusing stale output: {directory}")
            assets = directory / "assets"
            assets.mkdir(exist_ok=True)
            lines = []
            for line in (reference / "STRU").read_text().splitlines():
                fields = line.split()
                for i, word in enumerate(fields):
                    path = Path(word)
                    if path.suffix.lower() in {".orb", ".upf"}:
                        path = path if path.is_absolute() else reference / path
                        shutil.copy2(path, assets / path.name)
                        fields[i] = str((assets / path.name).resolve())
                lines.append(" ".join(fields))
            (directory / "STRU").write_text("\n".join(lines) + "\n")
            if periodic:
                model = torch.jit.trace(AnalyticCorrection(), torch.ones((1, 2, 3), dtype=torch.float64))
                model.save(str(directory / "model.ptg"))
                (directory / "KPT").write_text("K_POINTS\n0\nMP\n2 2 2 0 0 0\n")
                seed = (reference / inp["read_file_dir"]).resolve()
                shutil.copytree(seed, directory / "seed", symlinks=False)
                inp["read_file_dir"] = str((directory / "seed").resolve())
            else:
                shutil.copy2(inp["deepks_model"], directory / "model.ptg")
                shutil.copy2(reference / "KPT", directory / "KPT")
            inp["deepks_model"] = str((directory / "model.ptg").resolve())
            (directory / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
                f"{key} {value}\n" for key, value in inp.items()))
            with (directory / "log.scf").open("w") as log:
                subprocess.run(["mpiexec", "-n", "2", str(args.abacus.resolve())],
                               cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1200)
        else:
            inp["deepks_model"] = str((directory / "model.ptg").resolve())
            if periodic:
                inp["read_file_dir"] = str((directory / "seed").resolve())
            assert read_input(directory / "INPUT") == inp, (name, "INPUT differs")
        output = directory / "OUT.ABACUS"
        if not periodic:
            assert "#SCF IS CONVERGED#" in (output / "running_scf.log").read_text()
        assert not (output / "deepks_sample_hr_metric.npy").exists()
        return directory, output

    cases = [("explicit_hr", args.explicit_reference, -1, False)]
    cases += [(f"model_{selection}", args.model_reference, selection, False) for selection in (-1, -2, 1, 2)]
    if args.periodic_fixture:
        cases += [(f"periodic_{selection}", args.periodic_fixture, selection, True) for selection in (1, 2)]

    for name, reference, selection, periodic in cases:
        directory, out = run(name, reference, selection, periodic)
        model = torch.jit.load(str(directory / "model.ptg"), map_location="cpu").eval()
        descriptors = load(out, "deepks_sample_dm_eig")
        errors = {"descriptor_max_abs": None if periodic else 0.0,
                  "energy_max_abs": None if periodic else 0.0,
                  "projection_relative_max": 0.0 if selection == -1 else None,
                  "metric_relative_max": 0.0 if selection == -1 else None,
                  "model_recovery_relative_max": None if name == "explicit_hr" else 0.0,
                  "imaginary_hk_max_abs": 0.0 if selection > 0 else None,
                  "final_scf_correction_relative": None}
        if not periodic:
            reference_out = reference / "OUT.ABACUS"
            for key, stem in (("descriptor_max_abs", "deepks_sample_dm_eig"),
                              ("energy_max_abs", "deepks_sample_energy")):
                errors[key] = float(np.max(np.abs(load(out, stem) - load(reference_out, stem))))
                assert errors[key] < 1e-10, (name, key, errors[key])
        for i, descriptor in enumerate(descriptors):
            suffix = f"_s{i}"
            d = torch.tensor(descriptor[None], dtype=torch.float64, requires_grad=True)
            g, = torch.autograd.grad(model(d).sum(), d)
            if selection < 0:
                observed = read_csr(out / f"deepks_sample_hr{suffix}.csr").to_dense().numpy()
            else:
                observed = load(out, "deepks_sample_hk" + suffix)
                errors["imaginary_hk_max_abs"] = max(errors["imaginary_hk_max_abs"], float(np.max(np.abs(observed.imag))))
                np.testing.assert_allclose(observed, observed.swapaxes(-1, -2).conj(), atol=1e-12, rtol=1e-12)
            if selection == -1:
                operator = load(out, "deepks_sample_vdrpre" + suffix)
                a, x = align(operator, observed)
                a = a.reshape(-1, descriptor.size)
                metric = load(reference_out, "deepks_sample_hr_metric")[i]
                projection = load(reference_out, "deepks_sample_hr_projection")[i].reshape(-1)
                errors["metric_relative_max"] = max(errors["metric_relative_max"], relative(a.T @ a, metric))
                errors["projection_relative_max"] = max(errors["projection_relative_max"], relative(a.T @ x.reshape(-1), projection))
                predicted = (a @ g.detach().numpy().reshape(-1)).reshape(x.shape)
            elif selection == 1:
                operator = torch.tensor(load(out, "deepks_sample_vdpre" + suffix)[None])
                predicted = v_delta_from_context(g.to(operator.dtype), {"vdp": operator}).detach().numpy()[0]
            else:
                gev = torch.tensor(load(out, "deepks_sample_gevdm" + suffix)[None])
                if selection == -2:
                    lmax = (gev.shape[-1] - 1) // 2
                    nzeta = gev.shape[2] // (lmax + 1)
                    context = dict(gevdm=gev, data_shape=(nzeta, lmax),
                                   overlap=torch.tensor(load(out, "deepks_sample_phialpha_r" + suffix)[None]),
                                   iR_mat=torch.tensor(load(out, "deepks_sample_iRmat" + suffix)[None]).long())
                    predicted = vdr_from_context(g, context).detach().numpy()[0]
                else:
                    context = dict(gevdm=gev, phialpha=torch.tensor(load(out, "deepks_sample_phialpha" + suffix)[None]))
                    predicted = v_delta_from_context(g, context).detach().numpy()[0]
            if name != "explicit_hr":
                if selection < 0:
                    predicted, observed = align(predicted, observed)
                errors["model_recovery_relative_max"] = max(errors["model_recovery_relative_max"], relative(predicted, observed))
        for key, tolerance in (("projection_relative_max", 1e-10),
                               ("metric_relative_max", 1e-12),
                               ("model_recovery_relative_max", 1e-10)):
            assert errors[key] is None or errors[key] < tolerance, (name, errors)
        if name != "explicit_hr" and not periodic:
            if selection < 0:
                actual_scf = read_csr(out / "deepks_hrdelta.csr").to_dense().numpy()
                observed, actual_scf = align(observed, actual_scf)
            else:
                actual_scf = load(out, "deepks_vdelta")
            errors["final_scf_correction_relative"] = relative(observed, actual_scf)
            # Ordinary CSR uses eight decimals and the SCF stopping tolerance.
            assert errors["final_scf_correction_relative"] < 1e-7, (name, errors)
        if periodic:
            assert errors["imaginary_hk_max_abs"] > 1e-6, (name, "complex branch was not exercised")
        report[name] = dict(samples=len(descriptors), **errors)
        (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, json.dumps(report[name]), flush=True)


if __name__ == "__main__":
    main()
