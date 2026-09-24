# Stress derivative reference convention

The Gamma and multi-k `gdmepsl_ref.dat` and `gvepsl_ref.dat` files use the
upper-triangular order `(xx, xy, xz, yy, yz, zz)` and the contraction
`gradient_overlap[i] * displacement[j]` for component `(i,j)`.

Commit `330da8a8095743e3bc3afd0587c06083c99a4111` changed the implementation
from `gradient_overlap[j] * displacement[i]` but did not update these files.
The references were regenerated during native-test reorganization after:

1. Checking that all three diagonal components match the old references to
   rounding precision (maximum absolute difference below 2e-15).
2. Checking model-gradient force and all six virial components against the
   independent `cal_f_delta` correction routine, in both Gamma and multi-k
   fixtures. The direct force/stress reference files were left unchanged.
3. Checking the Gram matrix against explicit `A.transpose(0,1) @ A`.

No numerical tolerance or production stress implementation was changed by
this reference update. `deepks_test_force_stress_delta.cpp` retains the
direct-recovery comparison so that a self-consistent but incorrectly scaled
or transposed gradient label cannot pass only by matching regenerated files.
