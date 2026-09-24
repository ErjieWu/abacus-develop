"""Check passive paired output across the existing bounded HSE EXX fixture.

Retain tests/08_EXX/01_GO_S1_HSE SCF settings, including its two-outer-step
limit. This tests control-flow passivity, not fully converged hybrid accuracy.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from check_label_representation import load, read_input, relative
from check_pair_passivity import electronic_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--abacus", type=Path, required=True)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--outer-limit", type=int,
                        help="Increase only the outer-loop cap and require convergence before it")
    parser.add_argument("--mixed-case", action="store_true")
    parser.add_argument("--reference-results", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    fixture = repository / "tests/08_EXX/01_GO_S1_HSE"
    base = read_input(fixture / "INPUT")
    base.update(suffix="ABACUS", deepks_scf="0", deepks_out_labels="1",
                deepks_grad="1", deepks_v_delta="0")
    if args.outer_limit is not None:
        assert args.outer_limit > int(base["exx_hybrid_step"])
        base["exx_hybrid_step"] = str(args.outer_limit)
    if args.mixed_case:
        base["dft_functional"] = "HSE"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", I_MPI_PIN="0")
    args.workdir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for mode, paired in (("ordinary", False), ("paired", True)):
        directory = args.workdir / mode
        assets = directory / "assets"
        inp = dict(base, deepks_out_pair=str(int(paired)),
                   pseudo_dir=str(assets.resolve()), orbital_dir=str(assets.resolve()))
        if paired:
            inp.update(deepks_pair_base="pbe", deepks_pair_target="hse", deepks_out_pair_freq="5")
            if args.mixed_case:
                inp.update(deepks_pair_base="PbE", deepks_pair_target="hSe")
        if not args.analyze_only:
            assets.mkdir(parents=True, exist_ok=False)
            for filename in ("H_ONCV_PBE-1.0.upf", "O_ONCV_PBE-1.0.upf",
                             "H_gga_6au_60Ry_1s.orb", "O_gga_6au_60Ry_1s1p.orb"):
                shutil.copy2(repository / "tests/PP_ORB" / filename, assets / filename)
            projector = assets / "jle.orb"
            shutil.copy2(repository / "tests/09_DeePKS/Model_ProjOrb/2au_20Ry_jle.orb", projector)
            structure = (fixture / "STRU").read_text().replace(
                "LATTICE_CONSTANT", "NUMERICAL_DESCRIPTOR\n" + str(projector.resolve()) +
                "\n\nLATTICE_CONSTANT", 1)
            (directory / "STRU").write_text(structure)
            shutil.copy2(fixture / "KPT", directory / "KPT")
            (directory / "INPUT").write_text("INPUT_PARAMETERS\n" + "".join(
                f"{key} {value}\n" for key, value in inp.items()))
            with (directory / "log.scf").open("w") as stream:
                subprocess.run(["mpiexec", "-n", "2", str(args.abacus.resolve())],
                               cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=1800, check=True)
        else:
            assert read_input(directory / "INPUT") == inp
        outputs[mode] = directory / "OUT.ABACUS"

    a = electronic_rows(args.workdir / "ordinary/log.scf")
    b = electronic_rows(args.workdir / "paired/log.scf")
    assert a.shape == b.shape and len(a) > 3, (a.shape, b.shape)
    np.testing.assert_array_equal(a, b)
    report = {"fixture": str(fixture.relative_to(repository)),
              "exx_hybrid_step": int(base["exx_hybrid_step"]), "electronic_steps": len(a),
              "printed_energy_residual_max_abs": float(np.max(np.abs(a - b))),
              "scope": "bounded outer-loop passivity, not hybrid convergence accuracy"}
    for stem in ("deepks_etot", "deepks_dm_eig", "deepks_ftot", "deepks_stot",
                 "deepks_gradvx", "deepks_gvepsl", "deepks_vdrpre_square",
                 "deepks_dot_phialpha_hamilt"):
        ordinary = load(outputs["ordinary"], stem)
        paired = load(outputs["paired"], stem)
        np.testing.assert_allclose(paired, ordinary, rtol=1e-11, atol=1e-12)
        report[stem + "_relative"] = relative(paired, ordinary)
        if args.reference_results:
            previous = load(args.reference_results / "paired/OUT.ABACUS", stem)
            np.testing.assert_allclose(paired, previous, rtol=1e-11, atol=1e-12)
            report[stem + "_previous_relative"] = relative(paired, previous)
    kinds = load(outputs["paired"], "deepks_sample_kind")
    assert 1 in kinds and 2 in kinds, kinds
    report["sample_kinds"] = kinds.tolist()
    report["sample_steps"] = load(outputs["paired"], "deepks_sample_step").tolist()
    if args.outer_limit is not None:
        outer_count = int(np.count_nonzero(kinds == 1))
        assert outer_count < args.outer_limit, (outer_count, args.outer_limit)
        for output in outputs.values():
            assert "#SCF IS CONVERGED#" in (output / "running_scf.log").read_text()
        report.update(scope="converged hybrid passivity with unchanged SCF tolerances",
                      converged_outer_checkpoints=outer_count)
    for field in ("force", "virial"):
        for suffix in ("metric", "projection"):
            array = load(outputs["paired"], "deepks_pair_" + field + "_" + suffix)
            assert np.isfinite(array).all()
    if args.reference_results:
        previous = args.reference_results / "paired/OUT.ABACUS"
        paths = sorted(previous.glob("deepks_sample_*.npy")) + sorted(previous.glob("deepks_pair_*.npy"))
        errors = {}
        for path in paths:
            expected = np.load(path)
            observed = np.load(outputs["paired"] / path.name)
            np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-12)
            errors[path.stem] = float(np.max(np.abs(observed - expected)))
        report["paired_labels_previous_max_abs"] = errors
    (args.workdir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
