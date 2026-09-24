"""Check independent force/stress selection against both-enabled gradient runs.

--reference contains gradient_model, gradient_no_model, assets, and model.ptg
from check_label_representation.py. Six new SCFs cover force/stress/HR only,
with and without a model, using independent copies of the inputs and assets.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from check_label_representation import load, read_input, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--paired", action="store_true",
                        help="Use model_gradient/named_gradient paired references")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", I_MPI_PIN="0")
    report = {}
    for model in (("model", "named") if args.paired else ("no_model", "model")):
        fixture = args.reference / (model + "_gradient" if args.paired else "gradient_" + model)
        assets_source = (fixture if args.paired else args.reference) / "assets"
        model_source = (fixture if args.paired else args.reference) / "model.ptg"
        for selection, force, stress in (("force", 1, 0), ("stress", 0, 1), ("hr", 0, 0)):
            name = model + "_" + selection
            directory = args.workdir / name
            directory.mkdir(exist_ok=True)
            inp = read_input(fixture / "INPUT")
            inp.update(cal_force=str(force), cal_stress=str(stress),
                       deepks_model=str((directory / "model.ptg").resolve()))
            output = directory / "OUT.ABACUS"
            if args.analyze_only:
                assert read_input(directory / "INPUT") == inp
            else:
                if output.exists():
                    raise FileExistsError(f"Refusing stale output: {output}")
                shutil.copytree(assets_source, directory / "assets", symlinks=False)
                shutil.copy2(model_source, directory / "model.ptg")
                stru = (fixture / "STRU").read_text().replace(
                    str(assets_source.resolve()), str((directory / "assets").resolve()))
                (directory / "STRU").write_text(stru)
                shutil.copy2(fixture / "KPT", directory / "KPT")
                (directory / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
                    f"{key} {value}\n" for key, value in inp.items()))
                with (directory / "log.scf").open("w") as log:
                    subprocess.run(["mpiexec", "-n", "2", str(args.abacus.resolve())],
                                   cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=1200)
            assert "#SCF IS CONVERGED#" in (output / "running_scf.log").read_text()
            stems = ["deepks_etot", "deepks_dm_eig", "deepks_vdrpre_square",
                     "deepks_dot_phialpha_hamilt"]
            if args.paired:
                stems += ["deepks_sample_" + field for field in
                          ("dm_eig", "energy", "hr_metric", "hr_projection", "step", "kind")]
                for enabled, field in ((force, "force"), (stress, "virial")):
                    for suffix in ("metric", "projection"):
                        stem = "deepks_pair_" + field + "_" + suffix
                        if enabled:
                            stems.append(stem)
                        else:
                            assert not (output / (stem + ".npy")).exists()
            for enabled, total, operator in ((force, "deepks_ftot", "deepks_gradvx"),
                                              (stress, "deepks_stot", "deepks_gvepsl")):
                if enabled:
                    stems += [total, operator, operator + "_square"]
                else:
                    assert not (output / (operator + ".npy")).exists()
                    assert not (output / (operator + "_square.npy")).exists()
            errors = {}
            for stem in stems:
                a = load(output, stem)
                b = load(fixture / "OUT.ABACUS", stem)
                errors[stem] = relative(a, b)
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
            report[name] = errors
            (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(name, json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()
