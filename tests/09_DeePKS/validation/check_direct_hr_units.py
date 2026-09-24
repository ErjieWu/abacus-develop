"""Verify packaged direct HR labels against references with explicit units."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from deepks.physics.backends.abacus.utils import read_csr

from check_label_representation import read_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--reference-unit", choices=("hartree", "rydberg"), required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    report = {"reference_unit": args.reference_unit, "output_unit": "hartree", "cases": {}}
    factor = 0.5 if args.reference_unit == "rydberg" else 1.0
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", I_MPI_PIN="0")
    args.workdir.mkdir(parents=True, exist_ok=True)
    for name in ("21_NO_GO_deepks_vdelta_r_1", "22_NO_GO_deepks_vdelta_r_2",
                 "23_NO_KP_deepks_vdelta_r_1", "24_NO_KP_deepks_vdelta_r_2"):
        fixture = repository / "tests/09_DeePKS" / name
        directory = args.workdir / name
        assets = directory / "assets"
        output = directory / "OUT.autotest"
        inp = read_input(fixture / "INPUT")
        inp.update(pseudo_dir=str(assets.resolve()), orbital_dir=str(assets.resolve()),
                   deepks_model=str((assets / "model_lda_pbe_18.ptg").resolve()))
        if not args.analyze_only and not (args.reuse_existing and output.exists()):
            assets.mkdir(parents=True, exist_ok=False)
            for filename in ("H_ONCV_PBE2PZ-1.0.upf", "O_ONCV_PBE2PZ-1.0.upf",
                             "H_gga_6au_60Ry_2s1p.orb", "O_gga_6au_60Ry_2s2p1d.orb"):
                shutil.copy2(repository / "tests/PP_ORB" / filename, assets / filename)
            for filename in ("2au_20Ry_jle.orb", "model_lda_pbe_18.ptg"):
                shutil.copy2(fixture.parent / "Model_ProjOrb" / filename, assets / filename)
            structure = (fixture / "STRU").read_text().replace("../Model_ProjOrb/", str(assets.resolve()) + "/")
            (directory / "STRU").write_text(structure)
            shutil.copy2(fixture / "KPT", directory / "KPT")
            (directory / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
                f"{key} {value}\n" for key, value in inp.items()))
            with (directory / "log.scf").open("w") as stream:
                subprocess.run(["mpiexec", "-n", "2", str(args.abacus.resolve())],
                               cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=900, check=True)
        else:
            assert read_input(directory / "INPUT") == inp
        assert "#SCF IS CONVERGED#" in (output / "running_scf.log").read_text()
        errors = {}
        for stem in ("deepks_hrtot", "deepks_hrdelta"):
            actual = read_csr(output / (stem + ".csr")).to_dense().numpy()
            expected = factor * read_csr(fixture / (stem + ".csr.ref")).to_dense().numpy()
            # The historical CSR references contain eight-digit serialized values.
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
            errors[stem + "_max_abs_hartree"] = float(np.max(np.abs(actual - expected)))
        report["cases"][name] = errors
        (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()
