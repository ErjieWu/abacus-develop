#ifndef DEEPKS_PAIR_EXX_H
#define DEEPKS_PAIR_EXX_H

#ifdef __EXX
#include "source_lcao/module_ri/exx_lri_interface.h"

namespace DeePKS_domain
{
template <typename TK, typename TR>
class PairExx : public Exx_LRI_Interface<TK, TR>
{
  public:
    PairExx(const Exx_Info_RI& info, const Exx_Info_Global& global)
        : Exx_LRI_Interface<TK, TR>(info, global)
    {
    }

    void exx_before_all_runners(const K_Vectors& kv, const UnitCell& cell, const Parallel_2D& orbitals);
    void cal_exx_elec(const elecstate::DensityMatrix<TK, double>& density,
                      const UnitCell& cell, const K_Vectors& kv,
                      const Parallel_Orbitals& orbitals, const int nspin);

  private:
    ModuleSymmetry::Symmetry_rotation rotation;
    bool use_symmetry = false;
};
}
#endif
#endif
