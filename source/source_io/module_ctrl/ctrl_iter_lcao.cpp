#include "ctrl_iter_lcao.h" // use ctrl_iter_lcao() 

#include "source_base/global_variable.h" // use GlobalC
#include "source_hamilt/module_xc/exx_info.h"
#include "source_io/module_restart/restart.h"
#ifdef __EXX
#include "source_lcao/module_ri/exx_lri_interface.h"
#endif

namespace ModuleIO
{

template <typename TK, typename TR>
void ctrl_iter_lcao(UnitCell& ucell, // unit cell *
        const Input_para& inp, // input parameters *
		K_Vectors& kv, // k points *
		elecstate::ElecState* pelec, // electronic info * 
        elecstate::DensityMatrix<TK, double>& dm, // density matrix, mohan add 2025-11-03
		Parallel_Orbitals& pv, // parallel orbital info *
		Grid_Driver& gd, // adjacent atom info *
		psi::Psi<TK>* psi, // wave functions *
        Charge &chr, // charge density *
        Charge_Mixing* p_chgmix, // charge mixing *
		hamilt::HamiltLCAO<TK, TR>* p_hamilt, // hamiltonian *
		LCAO_Orbitals &orb, // orbital info *
        Setup_DeePKS<TK> &deepks,
        Exx_NAO<TK> &exx_nao,
        const Exx_Info& exx_info,
        int &iter,
        const int istep,
        bool &conv_esolver,
		const double &scf_ene_thr)
{
    ModuleBase::TITLE("ModuleIO", "ctrl_iter_lcao");
    ModuleBase::timer::start("ModuleIO", "ctrl_iter_lcao");

    // save charge density
    // Peize Lin add 2020.04.04
    if (GlobalC::restart.info_save.save_charge)
    {
        for (int is = 0; is < inp.nspin; ++is)
        {
            GlobalC::restart.save_disk("charge", is, chr.nrxx, chr.rho[is]);
        }
    }

#ifdef __EXX
    bool cal_exx = exx_info.info_global.cal_exx;
    bool real_number = exx_info.info_ri.real_number;

    if (inp.calculation != "nscf")
    {
        if (cal_exx)
        {
            real_number ?
              exx_nao.exd->exx_iter_finish(kv, ucell, *p_hamilt, *pelec, &dm, 
                *p_chgmix, scf_ene_thr, iter, istep, conv_esolver) :
              exx_nao.exc->exx_iter_finish(kv, ucell, *p_hamilt, *pelec, &dm,
                *p_chgmix, scf_ene_thr, iter, istep, conv_esolver);
        }
    }
#endif


    ModuleBase::timer::end("ModuleIO", "ctrl_iter_lcao");
}

// TK: double  TR: double 
template void ctrl_iter_lcao<double, double>(UnitCell& ucell, // unit cell *
        const Input_para& inp, // input parameters *
		K_Vectors& kv, // k points *
		elecstate::ElecState* pelec, // electronic info * 
        elecstate::DensityMatrix<double, double>& dm, // density matrix, mohan add 2025-11-03
		Parallel_Orbitals& pv, // parallel orbital info *
		Grid_Driver& gd, // adjacent atom info *
		psi::Psi<double>* psi, // wave functions *
        Charge &chr, // charge density *
        Charge_Mixing* p_chgmix, // charge mixing *
		hamilt::HamiltLCAO<double, double>* p_hamilt, // hamiltonian *
		LCAO_Orbitals &orb, // orbital info *
        Setup_DeePKS<double> &deepks,
        Exx_NAO<double> &exx_nao,
        const Exx_Info& exx_info,
        int &iter,
        const int istep,
        bool &conv_esolver,
		const double &scf_ene_thr);

// TK: complex<double>  TR: double 
template void ctrl_iter_lcao<std::complex<double>, double>(UnitCell& ucell, // unit cell *
        const Input_para& inp, // input parameters *
		K_Vectors& kv, // k points *
		elecstate::ElecState* pelec, // electronic info * 
        elecstate::DensityMatrix<std::complex<double>, double>& dm, // density matrix, mohan add 2025-11-03
		Parallel_Orbitals& pv, // parallel orbital info *
		Grid_Driver& gd, // adjacent atom info *
		psi::Psi<std::complex<double>>* psi, // wave functions *
        Charge &chr, // charge density *
        Charge_Mixing* p_chgmix, // charge mixing *
		hamilt::HamiltLCAO<std::complex<double>, double>* p_hamilt, // hamiltonian *
		LCAO_Orbitals &orb, // orbital info *
        Setup_DeePKS<std::complex<double>> &deepks,
        Exx_NAO<std::complex<double>> &exx_nao,
        const Exx_Info& exx_info,
        int &iter,
        const int istep,
        bool &conv_esolver,
		const double &scf_ene_thr);

// TK: complex<double>  TR: complex<double> 
template void ctrl_iter_lcao<std::complex<double>, std::complex<double>>(UnitCell& ucell, // unit cell *
        const Input_para& inp, // input parameters *
		K_Vectors& kv, // k points *
		elecstate::ElecState* pelec, // electronic info * 
        elecstate::DensityMatrix<std::complex<double>, double>& dm, // density matrix, mohan add 2025-11-03
		Parallel_Orbitals& pv, // parallel orbital info *
		Grid_Driver& gd, // adjacent atom info *
		psi::Psi<std::complex<double>>* psi, // wave functions *
        Charge &chr, // charge density *
        Charge_Mixing* p_chgmix, // charge mixing *
		hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>* p_hamilt, // hamiltonian *
		LCAO_Orbitals &orb, // orbital info *
        Setup_DeePKS<std::complex<double>> &deepks,
        Exx_NAO<std::complex<double>> &exx_nao,
        const Exx_Info& exx_info,
        int &iter,
        const int istep,
        bool &conv_esolver,
		const double &scf_ene_thr);

} // end ModuleIO
