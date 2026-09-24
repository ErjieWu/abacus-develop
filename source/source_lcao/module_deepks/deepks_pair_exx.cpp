#include "deepks_pair_exx.h"

#ifdef __EXX
template <typename TK, typename TR>
void DeePKS_domain::PairExx<TK, TR>::exx_before_all_runners(
    const K_Vectors& kv, const UnitCell& cell, const Parallel_2D& orbitals)
{
    use_symmetry = ModuleSymmetry::Symmetry::symm_flag == 1;
    if (use_symmetry)
    {
        const auto period = RI_Util::get_Born_vonKarmen_period(kv);
        rotation.find_irreducible_sector(cell.symm, cell.atoms, cell.st,
            RI_Util::get_Born_von_Karmen_cells(period), period, cell.lat);
        rotation.set_abfs_Lmax(this->exx_ptr->abfs_Lmax());
        rotation.cal_Ms(kv, cell, orbitals);
    }
}

template <typename TK, typename TR>
void DeePKS_domain::PairExx<TK, TR>::cal_exx_elec(
    const elecstate::DensityMatrix<TK, double>& density, const UnitCell& cell,
    const K_Vectors& kv, const Parallel_Orbitals& orbitals, const int nspin)
{
    std::vector<std::vector<TK>> restored;
    const auto* matrices = &density.get_DMK_vector();
    if (use_symmetry)
    {
        restored = rotation.restore_dm(kv, density.get_DMK_vector(), orbitals);
        matrices = &restored;
    }
    std::vector<const std::vector<TK>*> pointers;
    pointers.reserve(matrices->size());
    for (const auto& matrix : *matrices)
    {
        pointers.push_back(&matrix);
    }
    const auto density_real = RI_2D_Comm::split_m2D_ktoR<TR>(
        cell, kv, pointers, orbitals, nspin, use_symmetry);
    const auto* symmetry = use_symmetry && this->exx_ptr->get_info_ri().exx_symmetry_realspace ? &rotation : nullptr;
    Exx_LRI_Interface<TK, TR>::cal_exx_elec(density_real, cell, orbitals, symmetry);
}

template class DeePKS_domain::PairExx<double, double>;
template class DeePKS_domain::PairExx<double, std::complex<double>>;
template class DeePKS_domain::PairExx<std::complex<double>, double>;
template class DeePKS_domain::PairExx<std::complex<double>, std::complex<double>>;
#endif
