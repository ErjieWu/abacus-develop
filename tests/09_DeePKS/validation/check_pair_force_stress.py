"""Check final paired derivatives and their direct/gradient representations."""

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
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--abacus', type=Path, required=True)
    parser.add_argument('--reuse-existing', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.workdir.mkdir(parents=True, exist_ok=True)
    report = {}
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', I_MPI_PIN='0')
    for pair in ('model', 'named'):
        outputs = {}
        for representation in ('direct', 'gradient'):
            directory = args.workdir / (pair + '_' + representation)
            directory.mkdir(exist_ok=args.reuse_existing)
            fixture = args.reference / 'gradient_model'
            inp = read_input(fixture / 'INPUT')
            inp.update(deepks_out_pair='1', deepks_out_pair_freq='5', deepks_v_delta='0',
                       deepks_grad=str(int(representation == 'gradient')), cal_force='1', cal_stress='1')
            if pair == 'named':
                inp.update(deepks_pair_base='lda', deepks_pair_target='pbe')
            else:
                inp.pop('deepks_pair_base', None)
                inp.pop('deepks_pair_target', None)
            inp['deepks_model'] = str((directory / 'model.ptg').resolve())
            output = directory / 'OUT.ABACUS'
            if output.exists():
                assert args.reuse_existing
                assert read_input(directory / 'INPUT') == inp
            else:
                shutil.copytree(args.reference / 'assets', directory / 'assets')
                shutil.copy2(args.reference / 'model.ptg', directory / 'model.ptg')
                structure = (fixture / 'STRU').read_text().replace(str((args.reference / 'assets').resolve()),
                                                                  str((directory / 'assets').resolve()))
                (directory / 'STRU').write_text(structure)
                shutil.copy2(fixture / 'KPT', directory / 'KPT')
                (directory / 'INPUT').write_text('INPUT_PARAMETERS\n' + ''.join(f'{key} {value}\n' for key, value in inp.items()))
                with (directory / 'log.scf').open('w') as stream:
                    subprocess.run(['mpiexec', '-n', '2', str(args.abacus.resolve())], cwd=directory, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1200)
            assert '#SCF IS CONVERGED#' in (output / 'running_scf.log').read_text()
            errors = {}
            for name in ('deepks_etot', 'deepks_dm_eig', 'deepks_ftot', 'deepks_stot'):
                measured = load(output, name)
                expected = load(fixture / 'OUT.ABACUS', name)
                np.testing.assert_allclose(measured, expected, rtol=1e-11, atol=1e-11)
                errors[name + '_passive_max_abs'] = float(np.max(np.abs(measured - expected)))
            report[directory.name] = errors
            outputs[representation] = output
        descriptor = torch.tensor(load(outputs['gradient'], 'deepks_dm_eig')[None], dtype=torch.float64, requires_grad=True)
        model = torch.jit.load(str(directory / 'model.ptg')).eval()
        gradient, = torch.autograd.grad(model(descriptor).sum(), descriptor)
        gradient = gradient.detach().numpy().reshape(-1)
        for property_name, operator_name, sign, total, base in (
                ('force', 'gradvx', -1, 'ftot', 'fbase'), ('virial', 'gvepsl', 1, 'stot', 'sbase')):
            operator = sign * load(outputs['gradient'], 'deepks_' + operator_name).reshape(-1, gradient.size)
            correction = load(outputs['direct'], 'deepks_pair_' + property_name).reshape(-1)
            metric = load(outputs['gradient'], 'deepks_pair_' + property_name + '_metric')
            projection = load(outputs['gradient'], 'deepks_pair_' + property_name + '_projection').reshape(-1)
            np.testing.assert_allclose(metric, operator.T @ operator, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(projection, operator.T @ correction, rtol=1e-10, atol=1e-12)
            errors = dict(metric_relative=relative(metric, operator.T @ operator),
                          projection_relative=relative(projection, operator.T @ correction))
            if pair == 'model':
                physical = (load(outputs['gradient'], 'deepks_' + total)
                            - load(outputs['gradient'], 'deepks_' + base)).reshape(-1)
                np.testing.assert_allclose(correction, physical, rtol=1e-8, atol=1e-11)
                np.testing.assert_allclose(projection, metric @ gradient, rtol=1e-10, atol=1e-12)
                errors.update(physical_recovery_relative=relative(correction, physical),
                              model_projection_relative=relative(projection, metric @ gradient))
            report[pair + '_' + property_name] = errors
        (args.workdir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
