"""Run migrated paired-output fixtures and independent ordinary controls."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from check_label_representation import read_input


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--abacus', type=Path, required=True)
    parser.add_argument('--mixed-case', action='store_true')
    parser.add_argument('--reference-results', type=Path,
                        help='Also compare arrays against an earlier completed run')
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[3]
    tests = args.workdir.resolve() / 'tests'
    shutil.copytree(repository / 'tests/integrate/tools', tests / 'integrate/tools')
    shutil.copytree(repository / 'tests/09_DeePKS/tools', tests / '09_DeePKS/tools')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', I_MPI_PIN='0',
               PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    report = {}
    for name in ('25_NO_GO_deepks_out_freq_elec', '26_NO_KP_deepks_out_freq_elec'):
        fixture = repository / 'tests/09_DeePKS' / name
        outputs = {}
        for paired in (True, False):
            directory = tests / '09_DeePKS' / (name if paired else name + '_ordinary')
            assets = directory / 'assets'
            assets.mkdir(parents=True)
            for filename in ('H_ONCV_PBE2PZ-1.0.upf', 'O_ONCV_PBE2PZ-1.0.upf',
                             'H_gga_6au_60Ry_2s1p.orb', 'O_gga_6au_60Ry_2s2p1d.orb'):
                shutil.copy2(repository / 'tests/PP_ORB' / filename, assets / filename)
            shutil.copy2(fixture.parent / 'Model_ProjOrb/2au_20Ry_jle.orb', assets / 'jle.orb')
            structure = (fixture / 'STRU').read_text().replace('../Model_ProjOrb/2au_20Ry_jle.orb', str(assets / 'jle.orb'))
            (directory / 'STRU').write_text(structure)
            shutil.copy2(fixture / 'KPT', directory / 'KPT')
            inp = read_input(fixture / 'INPUT')
            inp.update(pseudo_dir=str(assets), orbital_dir=str(assets))
            if args.mixed_case:
                inp.update(dft_functional='PBE', deepks_pair_base='LDA', deepks_pair_target='pBe')
            if not paired:
                inp['deepks_out_pair'] = '0'
                inp.pop('deepks_pair_base')
                inp.pop('deepks_pair_target')
            (directory / 'INPUT').write_text('INPUT_PARAMETERS\n' + ''.join(f'{key} {value}\n' for key, value in inp.items()))
            with (directory / 'log.txt').open('w') as stream:
                subprocess.run(['mpiexec', '-n', '2', str(args.abacus.resolve())], cwd=directory, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, timeout=900, check=True)
            output = directory / 'OUT.autotest'
            assert '#SCF IS CONVERGED#' in (output / 'running_scf.log').read_text()
            outputs[paired] = output
            if paired:
                subprocess.run(['bash', str(tests / 'integrate/tools/catch_properties.sh'), 'result.out'],
                               cwd=directory, env=env, check=True, timeout=120)
                def checksums(path):
                    return {key: float(value) for key, value in
                            (line.split() for line in path.read_text().splitlines() if line.strip())
                            if key != 'totaltimeref'}
                expected = checksums(fixture / 'result.ref')
                observed = checksums(directory / 'result.out')
                assert observed.keys() == expected.keys(), (observed.keys(), expected.keys())
                for key, value in expected.items():
                    np.testing.assert_allclose(observed[key], value, rtol=1e-10, atol=1e-9, err_msg=key)
        errors = {}
        for stem in ('deepks_etot', 'deepks_dm_eig', 'deepks_ftot', 'deepks_stot', 'deepks_htot',
                     'deepks_gradvx', 'deepks_gvepsl', 'deepks_vdpre'):
            ordinary = np.load(outputs[False] / (stem + '.npy'))
            paired = np.load(outputs[True] / (stem + '.npy'))
            np.testing.assert_allclose(paired, ordinary, rtol=0, atol=1e-12)
            errors[stem] = float(np.max(np.abs(paired - ordinary)))
        if args.reference_results:
            previous = args.reference_results / 'tests/09_DeePKS' / name / 'OUT.autotest'
            for path in sorted(previous.glob('*.npy')):
                expected = np.load(path)
                observed = np.load(outputs[True] / path.name)
                np.testing.assert_allclose(observed, expected, rtol=1e-11, atol=1e-12)
                errors['previous_' + path.stem] = float(np.max(np.abs(observed - expected)))
        report[name] = errors
        (args.workdir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(name, json.dumps(errors), flush=True)


if __name__ == '__main__':
    main()
