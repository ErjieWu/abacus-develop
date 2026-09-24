"""Check direct/gradient dispatch using a copied small SCF fixture.

The fixture must contain INPUT, STRU and KPT, with valid asset paths. Supply
--reference for an earlier gradient OUT directory to check numerical parity.
Runs are serial in time, with two MPI ranks and one thread per rank.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def load(output, name):
    return np.load(output / (name + ".npy"), allow_pickle=False)


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-25))


def read_input(path):
    result = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line == "INPUT_PARAMETERS":
            continue
        key, value = line.split(maxsplit=1)
        result[key] = value
    return result


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True,
                        help="asset directory referenced by the fixture STRU; copied into the test")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--paired-reference", type=Path,
                        help="earlier LDA/PBE paired gradient OUT directory for the same fixture")
    parser.add_argument("--analyze-only", action="store_true",
                        help="check existing SCF outputs and rerun only INPUT diagnostics")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    base = read_input(args.fixture / "INPUT")
    shutil.copytree(args.assets, args.workdir / "assets", dirs_exist_ok=True, symlinks=False)
    shutil.copy2(base["deepks_model"], args.workdir / "model.ptg")
    base["deepks_model"] = str((args.workdir / "model.ptg").resolve())
    stru = (args.fixture / "STRU").read_text().replace(
        str(args.assets.resolve()), str((args.workdir / "assets").resolve()))
    base.update(deepks_out_labels="1", cal_force="1", cal_stress="1",
                deepks_out_pair="0", deepks_v_delta="0", deepks_bandgap="0")
    for key in ("deepks_out_base", "deepks_out_freq_elec", "deepks_pair_base", "deepks_pair_target"):
        base.pop(key, None)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", I_MPI_PIN="0")

    def run(name, **settings):
        directory = args.workdir / name
        directory.mkdir(exist_ok=True)
        inp = dict(base, **settings)
        output = directory / "OUT.ABACUS"
        if args.analyze_only:
            assert read_input(directory / "INPUT") == inp, (name, "INPUT differs")
            assert "#SCF IS CONVERGED#" in (output / "running_scf.log").read_text(), name
            return output
        if output.exists():
            raise FileExistsError(f"Refusing stale output: {directory / 'OUT.ABACUS'}")
        # No inherited output or symlinked case data.
        (directory / "STRU").write_text(stru)
        shutil.copy2(args.fixture / "KPT", directory / "KPT")
        (directory / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
            f"{key} {value}\n" for key, value in inp.items()))
        with (directory / "log.scf").open("w") as log:
            subprocess.run(["mpiexec", "-n", "2", str(args.abacus.resolve())],
                           cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1200)
        assert "#SCF IS CONVERGED#" in (output / "running_scf.log").read_text(), name
        return output

    report = {}
    for active in (False, True):
        suffix = "model" if active else "no_model"
        direct = run("direct_" + suffix, deepks_grad="0", deepks_v_delta="-1", deepks_scf=str(int(active)))
        grad = run("gradient_" + suffix, deepks_grad="1", deepks_scf=str(int(active)))
        for name in ("deepks_etot", "deepks_dm_eig", "deepks_ftot", "deepks_stot",
                     "deepks_gradvx", "deepks_gvepsl"):
            error = float(np.max(np.abs(load(direct, name) - load(grad, name))))
            report[suffix + "_direct_gradient_" + name] = error
            assert error < 1e-10, (suffix, name, error)
        assert not (grad / "deepks_vdrpre.npy").exists()
        assert not (grad / "deepks_hrtot.csr").exists()
        assert not (direct / "deepks_vdrpre_square.npy").exists()
        d = load(grad, "deepks_dm_eig")
        for operator, metric, rows in (
            ("deepks_vdrpre", "deepks_vdrpre_square", -1),
            ("deepks_gradvx", "deepks_gradvx_square", 3 * d.shape[0]),
            ("deepks_gvepsl", "deepks_gvepsl_square", 6),
        ):
            a = load(direct, operator).reshape(rows, d.size)
            error = relative(load(grad, metric), a.T @ a)
            report[suffix + "_" + metric + "_relative"] = error
            assert error < 1e-12, (suffix, metric, error)
        if active:
            model = torch.jit.load(base["deepks_model"], map_location="cpu").eval()
            descriptor = torch.tensor(d[None], dtype=torch.float64, requires_grad=True)
            g, = torch.autograd.grad(model(descriptor).sum(), descriptor)
            g = g.detach().numpy().reshape(-1)
            for operator, total, baseline, sign in (
                ("deepks_gradvx", "deepks_ftot", "deepks_fbase", -1),
                ("deepks_gvepsl", "deepks_stot", "deepks_sbase", 1),
            ):
                predicted = sign * load(grad, operator).reshape(-1, d.size) @ g
                observed = (load(grad, total) - load(grad, baseline)).reshape(-1)
                error = relative(predicted, observed)
                report[operator + "_model_recovery_relative"] = error
                assert error < 1e-6, (operator, error)
            if args.reference:
                for name in ("deepks_etot", "deepks_dm_eig", "deepks_ftot", "deepks_stot",
                             "deepks_vdrpre_square", "deepks_dot_phialpha_hamilt"):
                    error = relative(load(grad, name), load(args.reference, name))
                    report["previous_" + name + "_relative"] = error
                    assert error < 1e-10, (name, error)

    if args.paired_reference:
        paired = run("gradient_pair", deepks_grad="1", deepks_scf="1",
                     deepks_out_pair="1", deepks_out_pair_freq="5",
                     deepks_pair_base="lda", deepks_pair_target="pbe")
        for name in ("deepks_etot", "deepks_dm_eig", "deepks_ftot", "deepks_stot",
                     "deepks_sample_dm_eig", "deepks_sample_hr_metric", "deepks_sample_hr_projection",
                     "deepks_sample_energy", "deepks_sample_step", "deepks_sample_kind"):
            error = relative(load(paired, name), load(args.paired_reference, name))
            report["previous_pair_" + name + "_relative"] = error
            assert error < 1e-10, (name, error)

    # Invalid combinations must fail explicitly, not silently emit both forms.
    for name, extra, diagnostic in (
        ("invalid_v_delta", {"deepks_v_delta": "-1"}, "direct labels only"),
        ("invalid_bandgap", {"deepks_bandgap": "1"}, "not supported with deepks_grad"),
        ("invalid_grad_spin", {"nspin": "2"}, "deepks_grad currently supports nspin 1 only"),
        ("retired_base", {"deepks_out_base": "pbe"}, "THE PARAMETER NAME 'deepks_out_base' IS INCORRECT"),
        ("retired_target", {"deepks_out_target": "hse"}, "THE PARAMETER NAME 'deepks_out_target' IS INCORRECT"),
        ("retired_frequency", {"deepks_out_freq_elec": "0"}, "THE PARAMETER NAME 'deepks_out_freq_elec' IS INCORRECT"),
        ("invalid_labels", {"deepks_out_labels": "2"}, "deepks_grad requires deepks_out_labels 1"),
        ("invalid_equiv", {"deepks_equiv": "1"}, "requires eigenvalue descriptors"),
        ("invalid_pair_no_model", {"deepks_out_pair": "1", "deepks_scf": "0"},
         "requires deepks_pair_base and deepks_pair_target"),
        ("invalid_pair_one_name", {"deepks_out_pair": "1", "deepks_pair_base": "lda"},
         "requires deepks_pair_base and deepks_pair_target"),
    ):
        directory = args.workdir / name
        directory.mkdir(exist_ok=True)
        (directory / "STRU").write_text(stru)
        shutil.copy2(args.fixture / "KPT", directory / "KPT")
        inp = dict(base, deepks_grad="1", **extra)
        (directory / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
            f"{key} {value}\n" for key, value in inp.items()))
        result = subprocess.run([str(args.abacus.resolve()), "--check-input"], cwd=directory,
                                env=env, capture_output=True, text=True, timeout=60)
        text = result.stdout + result.stderr
        (directory / "check.log").write_text(text)
        assert result.returncode != 0, (name, "invalid INPUT unexpectedly succeeded", text)
        assert diagnostic in text, (name, result.returncode, text)
        report[name] = diagnostic

    (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
