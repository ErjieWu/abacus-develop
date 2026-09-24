#ifndef DEEPKS_PROJECTOR_H
#define DEEPKS_PROJECTOR_H

#include "source_basis/module_nao/radial_set.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <vector>

namespace DeePKS_domain
{

// Keep a common projector cutoff in the legacy interface. The ordinary
// atomic-orbital conversion still uses RadialSet::to_numerical_orbital.
// Header-only so the radial library does not acquire a libtorch dependency.
inline void projectors_to_numerical_orbital(const RadialSet& projectors,
                                          Numerical_Orbital& orbital,
                                          const int nk,
                                          const double dk,
                                          const bool out_element_info,
                                          const bool cal_force)
{
    const auto& first = *projectors.cbegin();
    const double* grid = first.rgrid();
    const int nr = std::min(first.nr(),
                           static_cast<int>(std::lower_bound(grid, grid + first.nr(),
                                                             projectors.rcut_max()) - grid) + 1);
    std::vector<double> rab(nr, grid[1] - grid[0]);
    std::vector<int> nchi(projectors.lmax() + 1);
    for (int l = 0; l <= projectors.lmax(); ++l)
    {
        nchi[l] = projectors.nzeta(l);
    }

    orbital.chi().resize(projectors.nchi());
    for (int i = 0; i < projectors.nchi(); ++i)
    {
        const auto& radial = projectors.cbegin()[i];
        if (radial.nr() != first.nr() || radial.pr() != 0
            || !std::equal(grid, grid + first.nr(), radial.rgrid()))
        {
            ModuleBase::WARNING_QUIT("DeePKS projectors", "Projector channels must share one radial grid.");
        }
        orbital.chi()[i].set_orbital_info(projectors.symbol(), projectors.itype(),
                                        radial.l(), radial.izeta(), nr, rab.data(), grid,
                                        Numerical_Orbital_Lm::Psi_Type::Psi, radial.rvalue(),
                                        nk, dk, 0.001, out_element_info, true, cal_force);
    }
    orbital.set_orbital_info(projectors.itype(), projectors.symbol(), projectors.lmax(),
                            nchi.data(), projectors.nchi());
}

} // namespace DeePKS_domain

#endif
