"""Check model force/virial recovery from labels of a completed DeePKS SCF.

Requires deepks_grad=1, deepks_out_labels=1, cal_force=1, cal_stress=1,
and the exact TorchScript model used by the SCF. Units are those of ABACUS
labels (Hartree energy/virial and Hartree/Bohr force); no conversion is applied.
With --hr, pass --direct-output from a separate deepks_grad=0,
deepks_v_delta=-1 run with the same structure, SCF settings and model.
Installed DeePKS-L supplies the existing sparse-HR reader.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--hr", action="store_true")
    parser.add_argument("--direct-output", type=Path)
    args = parser.parse_args()
    if args.hr and args.direct_output is None:
        parser.error("--hr requires --direct-output from the matching direct-label run")
    load = lambda name: np.load(args.output / ("deepks_" + name + ".npy"))
    d = torch.tensor(load("dm_eig")[None], dtype=torch.float64, requires_grad=True)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    g, = torch.autograd.grad(model(d).sum(), d)
    g = g.detach().numpy().reshape(-1)
    for operator, total, base, sign in [("gradvx", "ftot", "fbase", -1),
                                        ("gvepsl", "stot", "sbase", 1)]:
        a = sign * load(operator).reshape(-1, len(g))
        expected = (load(total) - load(base)).reshape(-1)
        predicted = a @ g
        np.testing.assert_allclose(predicted, expected, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(a.T @ a, load(operator + "_square"), rtol=1e-12, atol=1e-14)
        print(operator, "relative error", np.linalg.norm(predicted - expected) / max(np.linalg.norm(expected), 1e-20))
    if args.hr:
        from deepks.physics.backends.abacus.utils import read_csr

        np.testing.assert_allclose(np.load(args.direct_output / "deepks_dm_eig.npy"),
                                   load("dm_eig"), rtol=1e-12, atol=1e-12)
        operator = np.load(args.direct_output / "deepks_vdrpre.npy")
        a = operator.reshape(-1, len(g))
        np.testing.assert_allclose(a.T @ a, load("vdrpre_square"), rtol=1e-12, atol=1e-14)
        predicted = (operator.reshape(-1, len(g)) @ g).reshape(operator.shape[:-2])
        expected = read_csr(args.direct_output / "deepks_hrdelta.csr").to_dense().numpy()
        shape = tuple(max(predicted.shape[k], expected.shape[k]) for k in range(3))
        pad = lambda a: np.pad(a, [(0, shape[k] - a.shape[k]) for k in range(3)] + [(0, 0), (0, 0)])
        predicted, expected = pad(predicted), pad(expected)
        np.testing.assert_allclose(predicted, expected, rtol=1e-7, atol=1e-10)
        print("HR", "relative error", np.linalg.norm(predicted - expected) / max(np.linalg.norm(expected), 1e-20))


if __name__ == "__main__":
    main()
