"""Compact checksums of named paired-label arrays for integration tests."""

import argparse
from collections import defaultdict
from pathlib import Path
import re

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    energy = np.load(args.output / 'deepks_sample_energy.npy', mmap_mode='r', allow_pickle=False)
    assert energy.shape[0] > 0
    totals = defaultdict(float)
    paths = sorted(args.output.glob('deepks_sample_*.npy'))
    paths += sorted(args.output.glob('deepks_pair_force*.npy'))
    paths += sorted(args.output.glob('deepks_pair_virial*.npy'))
    for path in paths:
        values = np.load(path, mmap_mode='r', allow_pickle=False).ravel(order='K')
        key = re.sub(r'_s[0-9]+$', '', path.stem)
        for start in range(0, values.size, 262144):
            block = values[start:start + 262144]
            assert np.isfinite(block).all(), path
            totals[key] += float(np.abs(block).sum(dtype=np.float64))
    print('deepks_pair_samples', energy.shape[0])
    for key, value in sorted(totals.items()):
        print(key, format(value, '.16g'))


if __name__ == '__main__':
    main()
