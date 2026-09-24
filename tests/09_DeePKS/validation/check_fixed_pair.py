"""Compare direct and gradient pairs on identical one-step density fixtures.

Each fixture contains INPUT/STRU/KPT, read_file_dir seed matrices, and a previous
OUT.ABACUS gradient result. Intended for the existing PBE/HSE H2O and Si checks.
SCF convergence is not claimed: these deliberately use one electronic step.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from check_label_representation import load, read_input, relative
from deepks.physics.backends.abacus.utils import read_csr


def projection(operator, sparse_hr):
    sparse_hr = sparse_hr.coalesce()
    indices = sparse_hr.indices().numpy()
    values = sparse_hr.values().numpy()
    nfeature = int(np.prod(operator.shape[-2:]))
    result = np.zeros(nfeature)
    # Do not pad the dense derivative tensor to the longer hybrid HR range.
    # Its rows outside the projector support are exactly zero.
    for start in range(0, len(values), 2048):
        ind = indices[:, start:start + 2048]
        val = values[start:start + 2048]
        mask = np.all(ind[:3] < np.asarray(operator.shape[:3])[:, None], axis=0)
        ind = ind[:, mask]
        rows = operator[tuple(ind)].reshape(-1, nfeature)
        result += rows.T @ val[mask]
    return result


def degenerate_groups(descriptor, gevdm):
    lmax = (gevdm.shape[-1] - 1) // 2
    nzeta = gevdm.shape[1] // (lmax + 1)
    flat = descriptor.reshape(-1)
    groups = []
    offset = 0
    for _ in descriptor:
        for angular in range(lmax + 1):
            for _ in range(nzeta):
                nm = 2 * angular + 1
                group = [offset]
                for i in range(1, nm):
                    if abs(flat[offset + i] - flat[offset + i - 1]) < 1e-12:
                        group.append(offset + i)
                    else:
                        groups.append(group)
                        group = [offset + i]
                groups.append(group)
                offset += nm
    assert offset == flat.size
    z = np.zeros((flat.size, len(groups)))
    for i, group in enumerate(groups):
        z[group, i] = 1
    return z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, nargs="+", required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    report = {}
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", I_MPI_PIN="0")
    for fixture in args.fixtures:
        name = fixture.parent.name
        outputs = []
        for mode in ("gradient", "direct"):
            directory = args.workdir / name / mode
            directory.mkdir(parents=True, exist_ok=True)
            inp = read_input(fixture / "INPUT")
            inp.update(deepks_out_pair="1", deepks_out_pair_freq="1", deepks_scf="0",
                       cal_force="0", cal_stress="0", scf_nmax="1",
                       deepks_grad="1" if mode == "gradient" else "0",
                       deepks_v_delta="0" if mode == "gradient" else "-1")
            source_seed = (fixture / inp["read_file_dir"]).resolve()
            inp["read_file_dir"] = str((directory / "seed").resolve())
            output = directory / "OUT.ABACUS"
            if args.analyze_only:
                assert read_input(directory / "INPUT") == inp
            else:
                if output.exists():
                    raise FileExistsError(f"Refusing stale output: {output}")
                shutil.copytree(source_seed, directory / "seed", symlinks=False)
                assets = directory / "assets"
                assets.mkdir(exist_ok=True)
                lines = []
                for line in (fixture / "STRU").read_text().splitlines():
                    fields = line.split()
                    for i, word in enumerate(fields):
                        path = Path(word)
                        if path.suffix.lower() in {".orb", ".upf"}:
                            path = path if path.is_absolute() else fixture / path
                            shutil.copy2(path, assets / path.name)
                            fields[i] = str((assets / path.name).resolve())
                    lines.append(" ".join(fields))
                (directory / "STRU").write_text("\n".join(lines) + "\n")
                shutil.copy2(fixture / "KPT", directory / "KPT")
                (directory / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
                    f"{key} {value}\n" for key, value in inp.items()))
                with (directory / "log.scf").open("w") as log:
                    subprocess.run(["mpiexec", "-n", "2", str(args.abacus.resolve())],
                                   cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=1200)
            outputs.append(output)
        grad, direct = outputs
        old = fixture / "OUT.ABACUS"
        errors = {}
        for stem in ("dm_eig", "energy"):
            a = load(grad, "deepks_sample_" + stem)
            b = load(direct, "deepks_sample_" + stem)
            errors[stem + "_direct_gradient_relative"] = relative(a, b)
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
            previous = load(old, "deepks_sample_" + stem)
            errors[stem + "_previous_relative"] = relative(a, previous)
            np.testing.assert_allclose(a, previous, rtol=2e-7, atol=1e-10)
        operator = np.load(direct / "deepks_sample_vdrpre_s0.npy", mmap_mode="r")
        d = load(grad, "deepks_sample_dm_eig")[0]
        a = operator.reshape(-1, d.size)
        metric = load(grad, "deepks_sample_hr_metric")[0]
        b = load(grad, "deepks_sample_hr_projection")[0].reshape(-1)
        errors["metric_direct_gradient_relative"] = relative(a.T @ a, metric)
        recovered = projection(operator, read_csr(direct / "deepks_sample_hr_s0.csr"))
        errors["projection_direct_gradient_relative"] = relative(recovered, b)
        assert errors["metric_direct_gradient_relative"] < 1e-12, errors
        assert errors["projection_direct_gradient_relative"] < 1e-9, errors
        z = degenerate_groups(load(old, "deepks_sample_dm_eig")[0], load(old, "deepks_gevdm"))
        previous_metric = load(old, "deepks_sample_hr_metric")[0]
        previous_b = load(old, "deepks_sample_hr_projection")[0].reshape(-1)
        for key, actual, previous in (
            ("metric_previous_invariant", z.T @ metric @ z, z.T @ previous_metric @ z),
            ("projection_previous_invariant", z.T @ b, z.T @ previous_b),
        ):
            errors[key] = relative(actual, previous)
            np.testing.assert_allclose(actual, previous, rtol=2e-7, atol=1e-10)
        report[name] = errors
        (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()
