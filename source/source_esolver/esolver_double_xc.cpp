#include "esolver_double_xc.h"

#ifdef __MLALGO

#include "source_hamilt/module_ewald/h_ewald_pw.h"
#include "source_hamilt/module_xc/xc_functional.h"
#ifdef __MLALGO
#include "source_lcao/module_deepks/lcao_deepks.h"
#include "source_lcao/module_deepks/lcao_deepks_iface.h"
#include "source_lcao/module_deepks/lcao_deepks_io.h"
#include "source_lcao/module_deepks/deepks_grad.h"
#include "source_lcao/module_deepks/deepks_vdrpre.h"
#include "source_lcao/module_deepks/deepks_vdpre.h"
#include "source_hamilt/module_hcontainer/hcontainer_funcs.h"
#include "source_hamilt/module_hcontainer/output_hcontainer.h"
#endif
#include "source_lcao/force_stress_lcao.h"
#include "source_lcao/rho_tau_lcao.h"

//-----HSolver ElecState Hamilt--------
#include "source_estate/elecstate_lcao.h"
#include "source_estate/elecstate_tools.h"
#include "source_hsolver/hsolver_lcao.h"
#include "source_io/module_parameter/parameter.h"
#include "source_io/module_restart/restart.h" // GlobalC::restart for load_exx_flag
#include "source_lcao/hamilt_lcao.h"
#include "source_lcao/setup_deepks.h" // use deepks, mohan add 2025-10-10
#include "source_base/global_function.h"
#ifdef __EXX
#include "source_lcao/module_operator_lcao/op_exx_lcao.h"
#include "source_lcao/module_ri/ri_2d_comm.h"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace
{
std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value;
}

#ifdef __EXX
struct HybridFunctionalSpec
{
    double alpha = 0.0;
    double omega = 0.0;
    double ccp_rmesh_times = 1.0;
    std::map<Conv_Coulomb_Pot_K::Coulomb_Type, std::vector<std::map<std::string, std::string>>> coulomb;
};

HybridFunctionalSpec hybrid_functional_spec(const std::string& functional)
{
    using CoulombType = Conv_Coulomb_Pot_K::Coulomb_Type;
    const std::string name = lowercase(functional);
    HybridFunctionalSpec spec;
    double fock = 0.0;
    double erfc = 0.0;
    double omega = 0.0;
    std::string singularity = "spencer";

    if (name == "hf" || name == "muller" || name == "power")
    {
        fock = 1.0;
        spec.ccp_rmesh_times = 5.0;
    }
    else if (name == "pbe0" || name == "scan0")
    {
        fock = 0.25;
        spec.ccp_rmesh_times = 5.0;
    }
    else if (name == "b3lyp")
    {
        fock = 0.2;
    }
    else if (name == "hse")
    {
        erfc = 0.25;
        omega = 0.11;
        singularity = "limits";
        spec.ccp_rmesh_times = 1.5;
    }
    else if (name == "cwp22")
    {
        erfc = 1.0;
        omega = 0.11;
        singularity = "limits";
        spec.ccp_rmesh_times = 1.5;
    }
    else if (name == "wp22")
    {
        fock = 1.0;
        erfc = -1.0;
        omega = 0.11;
        spec.ccp_rmesh_times = 5.0;
    }
    else if (name == "lc_pbe")
    {
        fock = 1.0;
        erfc = -1.0;
        omega = 0.33;
    }
    else if (name == "lc_wpbe")
    {
        fock = 1.0;
        erfc = -1.0;
        omega = 0.4;
    }
    else if (name == "lrc_wpbe")
    {
        fock = 1.0;
        erfc = -1.0;
        omega = 0.3;
    }
    else if (name == "lrc_wpbeh")
    {
        fock = 1.0;
        erfc = -0.8;
        omega = 0.2;
    }
    else if (name == "cam_pbeh")
    {
        fock = 0.2;
        erfc = 0.8;
        omega = 0.7;
    }
    else
    {
        throw std::invalid_argument("No ABACUS fixed-density EXX specification is defined for functional "
                                    + functional);
    }

    spec.alpha = std::max(std::abs(fock), std::abs(erfc));
    spec.omega = omega;
    if (spec.alpha <= 0.0)
    {
        throw std::invalid_argument("Hybrid functional has zero exact-exchange fraction: " + functional);
    }
    if (fock != 0.0)
    {
        spec.coulomb[CoulombType::Fock] = {{{"alpha", std::to_string(fock / spec.alpha)},
                                             {"singularity_correction", singularity}}};
    }
    if (erfc != 0.0)
    {
        spec.coulomb[CoulombType::Erfc] = {{{"alpha", std::to_string(erfc / spec.alpha)},
                                            {"omega", std::to_string(omega)},
                                            {"singularity_correction", singularity}}};
    }
    return spec;
}
#endif
} // namespace

namespace ModuleESolver
{

template <typename TK, typename TR>
ESolver_DoubleXC<TK, TR>::ESolver_DoubleXC()
{
    this->classname = "ESolver_DoubleXC";
    this->basisname = "LCAO";
}

template <typename TK, typename TR>
ESolver_DoubleXC<TK, TR>::~ESolver_DoubleXC()
{
    delete this->p_hamilt_base;
    delete this->pelec_base;
}

template <typename TK, typename TR>
bool ESolver_DoubleXC<TK, TR>::pair_uses_model() const
{
    return this->inp_->deepks_out_pair && this->inp_->deepks_scf && this->pair_base_functional() == "none"
           && this->pair_target_functional() == "none";
}

template <typename TK, typename TR>
bool ESolver_DoubleXC<TK, TR>::active_is_pair_base() const
{
    return this->inp_->deepks_out_pair && lowercase(this->inp_->dft_functional) == this->pair_base_functional();
}

template <typename TK, typename TR>
std::string ESolver_DoubleXC<TK, TR>::pair_target_functional() const
{
    return lowercase(this->inp_->deepks_pair_target);
}

template <typename TK, typename TR>
std::string ESolver_DoubleXC<TK, TR>::pair_base_functional() const
{
    return lowercase(this->inp_->deepks_pair_base);
}

template <typename TK, typename TR>
std::string ESolver_DoubleXC<TK, TR>::auxiliary_functional() const
{
    return this->active_is_pair_base() ? this->pair_target_functional() : this->pair_base_functional();
}

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::before_all_runners(BaseCell& basecell, const Input_para& inp)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell& ucell = static_cast<UnitCell&>(basecell);

    ModuleBase::TITLE("ESolver_DoubleXC", "before_all_runners");
    ModuleBase::timer::start("ESolver_DoubleXC", "before_all_runners");

    ESolver_KS_LCAO<TK, TR>::before_all_runners(ucell, inp);

    if (this->pair_uses_model())
    {
        ModuleBase::timer::end("ESolver_DoubleXC", "before_all_runners");
        return;
    }

    // init some items for base functional

    // 2) init ElecState
    if (this->pelec_base == nullptr)
    {
        this->pelec_base = new elecstate::ElecStateLCAO<TK>(&(this->chr_base), // use which parameter?
                                                            &(this->kv),
                                                            this->kv.get_nks(),
                                                            this->pw_big);
    }



    // 6) initialize the density matrix
    this->dmat_base.allocate_dm(&this->kv, &this->pv, this->inp_->nspin);

    // 10) inititlize the charge density
	this->chr_base.set_rhopw(this->pw_rhod); // mohan add 20251130
    const std::string target_functional = this->pair_target_functional();
    const std::string base_functional = this->pair_base_functional();
    bool kin_den = false;
    {
        const DeePKS_domain::ScopedXCState active_xc;
        XC_Functional::set_xc_type(target_functional);
        kin_den = XC_Functional::get_ked_flag();
        XC_Functional::set_xc_type(base_functional);
        kin_den = kin_den || XC_Functional::get_ked_flag();
    }
	this->chr_base.allocate(this->inp_->nspin, kin_den);
	this->chr_base.init_rho(ucell, this->Pgrid, this->sf.strucFac, ucell.symm, &this->kv);
	this->chr_base.check_rho();

    // 11) initialize the potential
    if (this->pelec_base->pot == nullptr)
    {
        this->pelec_base->pot = new elecstate::Potential(this->pw_rhod,
                                                         this->pw_rho,
                                                         &ucell,
                                                         &(this->locpp.vloc),
                                                         &(this->sf),
                                                         &(this->solvent),
                                                         &(this->pelec_base->f_en.etxc),
                                                         &(this->pelec_base->f_en.vtxc));
    }

#ifdef __EXX
    this->initialize_pair_target_exx(ucell);
    this->initialize_base_exx(ucell);
#endif

    ModuleBase::timer::end("ESolver_DoubleXC", "before_all_runners");
}

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::before_scf(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_DoubleXC", "before_scf");
    ModuleBase::timer::start("ESolver_DoubleXC", "before_scf");

    ESolver_KS_LCAO<TK, TR>::before_scf(ucell, istep);

    if (this->inp_->deepks_out_pair && istep == 0)
    {
#ifdef __MLALGO
        this->sample_descriptors.clear();
        this->sample_metrics.clear();
        this->sample_projections.clear();
        this->sample_energies.clear();
        this->sample_steps.clear();
        this->sample_kinds.clear();
        this->sample_state_counter = 0;
#endif
        if (GlobalV::MY_RANK == 0)
        {
            for (const auto& entry : std::filesystem::directory_iterator(PARAM.globalv.global_out_dir))
            {
                const std::string name = entry.path().filename().string();
                if (entry.is_regular_file() && name.find("deepks_sample_") == 0)
                {
                    std::filesystem::remove(entry.path());
                }
            }
        }
#ifdef __MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

    if (this->pair_uses_model())
    {
        ModuleBase::timer::end("ESolver_DoubleXC", "before_scf");
        return;
    }

    //----------------------------------------------------------
    //! Reuse the vdW correction prepared by ESolver_FP::before_scf.
    //----------------------------------------------------------
    this->pelec_base->f_en.evdw = this->pelec->f_en.evdw;

    //----------------------------------------------------------
    //! calculate ewald energy
    //----------------------------------------------------------
    if (!this->inp_->test_skip_ewald)
    {
        // this->pelec_base->f_en.ewald_energy = H_Ewald_pw::compute_ewald(ucell, this->pw_rhod, this->sf.strucFac);
        this->pelec_base->f_en.ewald_energy = this->pelec->f_en.ewald_energy;
    }

    if (this->p_hamilt_base != nullptr)
    {
        delete this->p_hamilt_base;
        this->p_hamilt_base = nullptr;
    }
    if (this->p_hamilt_base == nullptr)
    {
        const bool load_exx_flag = !GlobalC::restart.info_load.restart_exx
                                   && GlobalC::restart.info_load.load_H;
        this->p_hamilt_base = new hamilt::HamiltLCAO<TK, TR>(ucell,
                                                             this->gd,
                                                             &this->pv,
                                                             this->pelec_base->pot,
                                                             this->kv,
                                                             this->two_center_bundle_,
                                                             this->orb_,
                                                             this->dmat_base.dm,
                                                             this->dftu_.get(),
                                                             this->deepks,
                                                             istep,
                                                             this->exx_nao,
                                                             this->exx_info_,
                                                             *this->inp_,
                                                             load_exx_flag);
    }

    const double auxiliary_alpha
        = this->active_is_pair_base() ? this->pair_target_exx_alpha : this->base_exx_alpha;
    const double auxiliary_omega
        = this->active_is_pair_base() ? this->pair_target_exx_omega : this->base_exx_omega;
    {
        const DeePKS_domain::ScopedXCState active_xc;
        XC_Functional::set_xc_type(this->auxiliary_functional());
        XC_Functional::set_hybrid_alpha(auxiliary_alpha);
        XC_Functional::set_hse_omega(auxiliary_omega);
        elecstate::init_scf(ucell, this->Pgrid, this->sf.strucFac, this->locpp.numeric, istep,
                           PARAM.globalv.global_out_dir, *this->inp_, this->pelec_base);
    }

    // DMR should be same size with Hamiltonian(R)
    this->dmat_base.dm->init_DMR(*(dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt_base)->getHR()));

    if (istep > 0)
    {
        this->dmat_base.dm->cal_DMR();
    }

    ModuleBase::timer::end("ESolver_DoubleXC", "before_scf");
    return;
}

template <typename TK, typename TR>
int ESolver_DoubleXC<TK, TR>::get_exx_outer_step() const
{
#ifdef __EXX
    if (!this->exx_info_.info_global.cal_exx)
    {
        return -1;
    }
    return this->exx_info_.info_ri.real_number ? this->exx_nao.exd->get_two_level_step()
                                                  : this->exx_nao.exc->get_two_level_step();
#else
    return -1;
#endif
}

#ifdef __EXX
template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::initialize_fixed_density_exx(
    const UnitCell& ucell,
    const std::string& functional,
    const bool is_active_functional,
    Exx_Info_RI& info,
    Exx_Info_Global& global,
    std::shared_ptr<DeePKS_domain::PairExx<TK, double>>& evaluator_real,
    std::shared_ptr<DeePKS_domain::PairExx<TK, std::complex<double>>>& evaluator_complex,
    double& alpha,
    double& omega)
{
    bool has_exx = false;
    {
        const DeePKS_domain::ScopedXCState active_xc;
        XC_Functional::set_xc_type(functional);
        has_exx = XC_Functional::get_func_type() == 4 || XC_Functional::get_func_type() == 5;
    }
    if (!has_exx)
    {
        return;
    }

    info = this->exx_info_.info_ri;
    global = this->exx_info_.info_global;
    if (is_active_functional)
    {
        if (!this->exx_info_.info_global.cal_exx)
        {
            ModuleBase::WARNING_QUIT("ESolver_DoubleXC",
                                     "active hybrid functional has no initialized EXX evaluator: " + functional);
        }
        alpha = global.hybrid_alpha;
        omega = global.hse_omega;
    }
    else
    {
        HybridFunctionalSpec spec;
        try
        {
            spec = hybrid_functional_spec(functional);
        }
        catch (const std::invalid_argument& error)
        {
            ModuleBase::WARNING_QUIT("ESolver_DoubleXC", error.what());
        }
        alpha = spec.alpha;
        omega = spec.omega;
        info.coulomb_param = std::move(spec.coulomb);
        info.ccp_rmesh_times = spec.ccp_rmesh_times;
        global.cal_exx = true;
        global.hybrid_alpha = alpha;
        global.hse_omega = omega;
        global.coulomb_param = info.coulomb_param;
    }

    if (info.real_number)
    {
        evaluator_real = std::make_shared<DeePKS_domain::PairExx<TK, double>>(info, global);
        evaluator_real->init(MPI_COMM_WORLD, ucell, this->kv, this->orb_);
        evaluator_real->exx_before_all_runners(this->kv, ucell, this->pv);
        evaluator_real->cal_exx_ions(ucell);
    }
    else
    {
        evaluator_complex = std::make_shared<DeePKS_domain::PairExx<TK, std::complex<double>>>(info, global);
        evaluator_complex->init(MPI_COMM_WORLD, ucell, this->kv, this->orb_);
        evaluator_complex->exx_before_all_runners(this->kv, ucell, this->pv);
        evaluator_complex->cal_exx_ions(ucell);
    }
}

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::initialize_pair_target_exx(const UnitCell& ucell)
{
    if (!this->inp_->deepks_out_pair)
    {
        return;
    }
    this->initialize_fixed_density_exx(ucell,
                                       this->pair_target_functional(),
                                       !this->active_is_pair_base(),
                                       this->pair_target_exx_info,
                                       this->pair_target_exx_global,
                                       this->pair_target_exx_real,
                                       this->pair_target_exx_complex,
                                       this->pair_target_exx_alpha,
                                       this->pair_target_exx_omega);
}

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::initialize_base_exx(const UnitCell& ucell)
{
    if (!this->inp_->deepks_out_pair)
    {
        return;
    }
    this->initialize_fixed_density_exx(ucell,
                                       this->pair_base_functional(),
                                       this->active_is_pair_base(),
                                       this->base_exx_info,
                                       this->base_exx_global,
                                       this->base_exx_real,
                                       this->base_exx_complex,
                                       this->base_exx_alpha,
                                       this->base_exx_omega);
}
#endif

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::output_deepks_sample(UnitCell& ucell,
                                                   const int state_step,
                                                   const std::string& state_kind)
{
#ifdef __MLALGO
    if (this->pair_uses_model())
    {
        this->output_model_sample(ucell, state_step, state_kind);
        return;
    }
    // The descriptor, metric, and both Hamiltonians belong to this same DM.
    // Independent fixed-density EXX evaluators cannot perturb the active SCF.
    const DeePKS_domain::ScopedXCState active_xc;
    const int nks = this->kv.get_nks();
    for (int ik = 0; ik < nks; ++ik)
    {
        this->dmat_base.dm->set_DMK_pointer(ik, this->dmat.dm->get_DMK_pointer(ik));
    }
    this->dmat_base.dm->cal_DMR();

    const std::string target_functional = this->pair_target_functional();
    const std::string base_functional = this->pair_base_functional();
    XC_Functional::set_xc_type(target_functional);
    const bool target_needs_tau = XC_Functional::get_ked_flag();
    XC_Functional::set_xc_type(base_functional);
    const bool base_needs_tau = XC_Functional::get_ked_flag();
    if (target_needs_tau || base_needs_tau)
    {
        LCAO_domain::dm2tau(
            this->dmat_base.dm->get_DMR_vector(), this->inp_->nspin, &this->chr_base);
    }
    XC_Functional::set_xc_type(target_functional);

    auto* p_ham_base = dynamic_cast<hamilt::HamiltLCAO<TK, TR>*>(this->p_hamilt_base);
    if (p_ham_base == nullptr)
    {
        ModuleBase::WARNING_QUIT("ESolver_DoubleXC", "paired base Hamiltonian is not LCAO");
    }

    torch::Tensor dot_target;
    torch::Tensor dot_base;
    std::unique_ptr<hamilt::HContainer<TR>> correction;
    if (!this->inp_->deepks_grad && this->inp_->deepks_v_delta != 0)
    {
        correction.reset(new hamilt::HContainer<TR>(&this->pv));
    }

    // Target operator and energy at the selected density.
    XC_Functional::set_xc_type(target_functional);
    XC_Functional::set_hybrid_alpha(this->pair_target_exx_alpha);
    XC_Functional::set_hse_omega(this->pair_target_exx_omega);
    LCAO_domain::dm2rho(this->dmat_base.dm->get_DMR_vector(), this->inp_->nspin, &this->chr_base);
    this->pelec_base->pot->update_from_charge(&this->chr_base, &ucell);
    const double target_etxc = this->pelec_base->f_en.etxc;
    const double target_etxcc = this->pelec_base->f_en.etxcc;
    double target_raw_exx = 0.0;
#ifdef __EXX
    const bool target_has_exx
        = this->pair_target_exx_real != nullptr || this->pair_target_exx_complex != nullptr;
#else
    const bool target_has_exx = false;
#endif
    // The independently evaluated target EXX is added below. Temporarily use
    // a non-hybrid operator chain while preserving the target semilocal
    // potential just built from this density.
    if (target_has_exx)
    {
        XC_Functional::set_xc_type("pbe");
    }
    p_ham_base->refresh(true);
    p_ham_base->updateHk(0);
    if (correction)
    {
        correction->add_value_union(*p_ham_base->getHR(), TR(1));
    }
    DeePKS_domain::cal_phialpha_hamilt_proj<TR>(PARAM.globalv.nlocal,
                                                ucell.nat,
                                                this->deepks.ld.deepks_param,
                                                this->deepks.ld.phialpha,
                                                *p_ham_base->getHR(),
                                                ucell,
                                                this->orb_,
                                                this->pv,
                                                this->gd,
                                                dot_target);
    if (this->inp_->deepks_scf)
    {
        const auto* h_deepks = p_ham_base->get_V_delta_R();
        if (h_deepks == nullptr)
        {
            ModuleBase::WARNING_QUIT("ESolver_DoubleXC", "missing DeePKS correction in paired target operator");
        }
        torch::Tensor dot_deepks;
        DeePKS_domain::cal_phialpha_hamilt_proj<TR>(PARAM.globalv.nlocal,
                                                    ucell.nat,
                                                    this->deepks.ld.deepks_param,
                                                    this->deepks.ld.phialpha,
                                                    *h_deepks,
                                                    ucell,
                                                    this->orb_,
                                                    this->pv,
                                                    this->gd,
                                                    dot_deepks);
        dot_target -= dot_deepks;
        if (correction)
        {
            correction->add_value_union(*h_deepks, TR(-1));
        }
    }

#ifdef __EXX
    if (target_has_exx)
    {
        torch::Tensor dot_target_exx;
        hamilt::HContainer<TR> h_target_exx(&this->pv);
        std::unique_ptr<RI::Cell_Nearest<int, int, 3, double, 3>> cell_nearest;
        const ModuleBase::Vector3<double> koffset(std::fmod(this->kv.get_koffset(0), 1.0),
                                                  std::fmod(this->kv.get_koffset(1), 1.0),
                                                  std::fmod(this->kv.get_koffset(2), 1.0));
        if (koffset.norm() < 1.0e-10)
        {
            const std::array<int, 3> periods = RI_Util::get_Born_vonKarmen_period(this->kv);
            cell_nearest = std::make_unique<RI::Cell_Nearest<int, int, 3, double, 3>>(
                hamilt::init_cell_nearest(ucell, periods));
        }
        const auto* cell_nearest_ptr = cell_nearest.get();
        if (this->pair_target_exx_real != nullptr)
        {
            this->pair_target_exx_real->cal_exx_elec(
                *this->dmat_base.dm, ucell, this->kv, this->pv, this->inp_->nspin);
            target_raw_exx = this->pair_target_exx_real->get_Eexx();
            hamilt::reallocate_hcontainer<double, TR>(
                this->pair_target_exx_real->get_Hexxs(), &h_target_exx, cell_nearest_ptr);
            RI_2D_Comm::add_HexxR(0,
                                  this->pair_target_exx_alpha,
                                  this->pair_target_exx_real->get_Hexxs(),
                                  this->pv,
                                  PARAM.globalv.npol,
                                  h_target_exx,
                                  cell_nearest_ptr);
        }
        else
        {
            this->pair_target_exx_complex->cal_exx_elec(
                *this->dmat_base.dm, ucell, this->kv, this->pv, this->inp_->nspin);
            target_raw_exx = this->pair_target_exx_complex->get_Eexx();
            hamilt::reallocate_hcontainer<std::complex<double>, TR>(
                this->pair_target_exx_complex->get_Hexxs(), &h_target_exx, cell_nearest_ptr);
            RI_2D_Comm::add_HexxR(0,
                                  this->pair_target_exx_alpha,
                                  this->pair_target_exx_complex->get_Hexxs(),
                                  this->pv,
                                  PARAM.globalv.npol,
                                  h_target_exx,
                                  cell_nearest_ptr);
        }
        DeePKS_domain::cal_phialpha_hamilt_proj<TR>(PARAM.globalv.nlocal,
                                                    ucell.nat,
                                                    this->deepks.ld.deepks_param,
                                                    this->deepks.ld.phialpha,
                                                    h_target_exx,
                                                    ucell,
                                                    this->orb_,
                                                    this->pv,
                                                    this->gd,
                                                    dot_target_exx);
        dot_target += dot_target_exx;
        if (correction)
        {
            correction->add_value_union(h_target_exx, TR(1));
        }
    }
#endif

    // Base local/semi-local operator and energy at exactly the same density.
    XC_Functional::set_xc_type(base_functional);
    XC_Functional::set_hybrid_alpha(this->base_exx_alpha);
    XC_Functional::set_hse_omega(this->base_exx_omega);
    LCAO_domain::dm2rho(this->dmat_base.dm->get_DMR_vector(), this->inp_->nspin, &this->chr_base);
    this->pelec_base->pot->update_from_charge(&this->chr_base, &ucell);
    const double base_etxc = this->pelec_base->f_en.etxc;
    const double base_etxcc = this->pelec_base->f_en.etxcc;
    double base_raw_exx = 0.0;
#ifdef __EXX
    const bool base_has_exx = this->base_exx_real != nullptr || this->base_exx_complex != nullptr;
#else
    const bool base_has_exx = false;
#endif
    // The base EXX operator is evaluated independently below. Temporarily use
    // a non-hybrid XC type while refreshing so a target EXX operator cannot
    // leak into the base Hamiltonian; the already-built base potential is kept.
    if (base_has_exx)
    {
        XC_Functional::set_xc_type("pbe");
    }
    p_ham_base->refresh(true);
    p_ham_base->updateHk(0);
    if (correction)
    {
        correction->add_value_union(*p_ham_base->getHR(), TR(-1));
    }
    DeePKS_domain::cal_phialpha_hamilt_proj<TR>(PARAM.globalv.nlocal,
                                                ucell.nat,
                                                this->deepks.ld.deepks_param,
                                                this->deepks.ld.phialpha,
                                                *p_ham_base->getHR(),
                                                ucell,
                                                this->orb_,
                                                this->pv,
                                                this->gd,
                                                dot_base);
    if (this->inp_->deepks_scf)
    {
        const auto* h_deepks = p_ham_base->get_V_delta_R();
        if (h_deepks == nullptr)
        {
            ModuleBase::WARNING_QUIT("ESolver_DoubleXC", "missing DeePKS correction in paired base operator");
        }
        torch::Tensor dot_deepks;
        DeePKS_domain::cal_phialpha_hamilt_proj<TR>(PARAM.globalv.nlocal,
                                                    ucell.nat,
                                                    this->deepks.ld.deepks_param,
                                                    this->deepks.ld.phialpha,
                                                    *h_deepks,
                                                    ucell,
                                                    this->orb_,
                                                    this->pv,
                                                    this->gd,
                                                    dot_deepks);
        dot_base -= dot_deepks;
        if (correction)
        {
            correction->add_value_union(*h_deepks, TR(1));
        }
    }

#ifdef __EXX
    if (base_has_exx)
    {
        torch::Tensor dot_base_exx;
        hamilt::HContainer<TR> h_base_exx(&this->pv);
        std::unique_ptr<RI::Cell_Nearest<int, int, 3, double, 3>> cell_nearest;
        const ModuleBase::Vector3<double> koffset(std::fmod(this->kv.get_koffset(0), 1.0),
                                                  std::fmod(this->kv.get_koffset(1), 1.0),
                                                  std::fmod(this->kv.get_koffset(2), 1.0));
        if (koffset.norm() < 1.0e-10)
        {
            const std::array<int, 3> periods = RI_Util::get_Born_vonKarmen_period(this->kv);
            cell_nearest = std::make_unique<RI::Cell_Nearest<int, int, 3, double, 3>>(
                hamilt::init_cell_nearest(ucell, periods));
        }
        const auto* cell_nearest_ptr = cell_nearest.get();
        if (this->base_exx_real != nullptr)
        {
            this->base_exx_real->cal_exx_elec(
                *this->dmat_base.dm, ucell, this->kv, this->pv, this->inp_->nspin);
            base_raw_exx = this->base_exx_real->get_Eexx();
            hamilt::reallocate_hcontainer<double, TR>(
                this->base_exx_real->get_Hexxs(), &h_base_exx, cell_nearest_ptr);
            RI_2D_Comm::add_HexxR(0,
                                  this->base_exx_alpha,
                                  this->base_exx_real->get_Hexxs(),
                                  this->pv,
                                  PARAM.globalv.npol,
                                  h_base_exx,
                                  cell_nearest_ptr);
        }
        else
        {
            this->base_exx_complex->cal_exx_elec(
                *this->dmat_base.dm, ucell, this->kv, this->pv, this->inp_->nspin);
            base_raw_exx = this->base_exx_complex->get_Eexx();
            hamilt::reallocate_hcontainer<std::complex<double>, TR>(
                this->base_exx_complex->get_Hexxs(), &h_base_exx, cell_nearest_ptr);
            RI_2D_Comm::add_HexxR(0,
                                  this->base_exx_alpha,
                                  this->base_exx_complex->get_Hexxs(),
                                  this->pv,
                                  PARAM.globalv.npol,
                                  h_base_exx,
                                  cell_nearest_ptr);
        }
        DeePKS_domain::cal_phialpha_hamilt_proj<TR>(PARAM.globalv.nlocal,
                                                    ucell.nat,
                                                    this->deepks.ld.deepks_param,
                                                    this->deepks.ld.phialpha,
                                                    h_base_exx,
                                                    ucell,
                                                    this->orb_,
                                                    this->pv,
                                                    this->gd,
                                                    dot_base_exx);
        dot_base += dot_base_exx;
        if (correction)
        {
            correction->add_value_union(h_base_exx, TR(-1));
        }
    }
#endif

    DeePKS_domain::HRLabelSample state
        = DeePKS_domain::build_hr_label_sample<TK, TR>(PARAM.globalv.nlocal,
                                                                    ucell.nat,
                                                                    this->kv.kvec_d,
                                                                    ucell,
                                                                    this->orb_,
                                                                    this->pv,
                                                                    this->gd,
                                                                    *this->dmat_base.dm,
                                                                    this->deepks.ld,
                                                                    dot_target,
                                                                    dot_base,
                                                                    this->inp_->deepks_grad);

    // At fixed density all non-XC pieces of the density functional are shared.
    // etxc-etxcc is ABACUS' semilocal XC energy, while exact exchange is
    // stored separately as alpha * raw_Eexx. Internal energies are Ry; DeePKS
    // labels and model outputs are Hartree.
    const double target_functional_energy
        = target_etxc - target_etxcc + this->pair_target_exx_alpha * target_raw_exx;
    const double base_functional_energy
        = base_etxc - base_etxcc + this->base_exx_alpha * base_raw_exx;
    const double energy_correction_hartree
        = 0.5 * (target_functional_energy - base_functional_energy);

    if (correction)
    {
        this->write_direct_sample(ucell, state, *correction);
    }
    this->write_deepks_sample(state, energy_correction_hartree, state_step, state_kind);
#else
    ModuleBase::WARNING_QUIT("ESolver_DoubleXC", "deepks_out_pair requires DeePKS support");
#endif
}

#ifdef __MLALGO
template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::output_model_sample(UnitCell& ucell,
                                                 const int state_step,
                                                 const std::string& state_kind)
{
    const auto& param = this->deepks.ld.deepks_param;
    const int mmax = 2 * param.lmaxd + 1;
    const torch::Tensor zero = torch::zeros({param.inlmax, mmax, mmax}, torch::kFloat64);
    // Reuse the same descriptor/metric builder as explicit functional pairs.
    // The model supplies DeltaH=A*g at this density, hence b=M*g exactly.
    auto state = DeePKS_domain::build_hr_label_sample<TK, TR>(this->pv.get_global_row_size(),
                                                               ucell.nat,
                                                               this->kv.kvec_d,
                                                               ucell,
                                                               this->orb_,
                                                               this->pv,
                                                               this->gd,
                                                               *this->dmat.dm,
                                                               this->deepks.ld,
                                                               zero,
                                                               zero,
                                                               this->inp_->deepks_grad);
    const double energy = DeePKS_domain::project_model_gradient(state, this->deepks.ld);
    if (!this->inp_->deepks_grad && this->inp_->deepks_v_delta != 0)
    {
        hamilt::HContainer<TR> correction(&this->pv);
        DeePKS_domain::build_model_hr(state, this->deepks.ld.phialpha, ucell, this->orb_,
                                      this->pv, this->gd, correction);
        this->write_direct_sample(ucell, state, correction);
    }
    this->write_deepks_sample(state, energy, state_step, state_kind);
}

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::write_direct_sample(const UnitCell& ucell,
                                                  const DeePKS_domain::HRLabelSample& state,
                                                  const hamilt::HContainer<TR>& correction)
{
    const int rank = GlobalV::MY_RANK;
    const int nlocal = this->pv.get_global_row_size();
    const int nks = this->kv.get_nks();
    const int selection = this->inp_->deepks_v_delta;
    const std::string prefix = PARAM.globalv.global_out_dir + "deepks_sample_";
    const std::string suffix = "_s" + std::to_string(this->sample_state_counter);
    const auto& ld = this->deepks.ld;
    // The full derivative kernel uses exp(-ikR), whereas folding_HR uses
    // exp(+ikR). Only that kernel needs -k; prepare_phialpha already gives
    // the correct phase when contracted as phi_x * conj(phi_y).
    auto kernel_kvec = this->kv.kvec_d;
    for (auto& k : kernel_kvec)
    {
        k.x = -k.x;
        k.y = -k.y;
        k.z = -k.z;
    }

    Parallel_Orbitals serial_pv;
    serial_pv.set_serial(nlocal, nlocal);
    serial_pv.set_atomic_trace(ucell.get_iat2iwt(), ucell.nat, nlocal);
    hamilt::HContainer<TR> serial(&serial_pv);
#ifdef __MPI
    hamilt::gatherParallels(correction, &serial, 0);
#else
    serial.add_value_union(correction, TR(1));
#endif
    if (rank == 0 && selection < 0)
    {
        std::ofstream out(prefix + "hr" + suffix + ".csr");
        out << "Matrix Dimension of H(R): " << nlocal << '\n';
        out << "Matrix number of H(R): " << serial.size_R_loop() << '\n';
        hamilt::Output_HContainer<TR> writer(&serial, out, 1e-14, 15, LCAO_deepks_io::Ry2Hartree);
        writer.write(true);
    }
    if (rank == 0 && selection > 0)
    {
        const auto dtype = std::is_same<TK, double>::value ? torch::kFloat64 : torch::kComplexDouble;
        std::vector<torch::Tensor> matrices;
        for (int ik = 0; ik < nks; ++ik)
        {
            std::vector<TK> hk(static_cast<std::size_t>(nlocal) * nlocal, TK(0));
            hamilt::folding_HR(serial, hk.data(), this->kv.kvec_d[ik], nlocal, 0);
            matrices.push_back(torch::from_blob(hk.data(), {nlocal, nlocal}, dtype).clone()
                               * LCAO_deepks_io::Ry2Hartree);
        }
        LCAO_deepks_io::save_tensor2npy<TK>(prefix + "hk" + suffix + ".npy", torch::stack(matrices), rank);
    }

    const int r_size = 2 * DeePKS_domain::get_R_size(*ld.phialpha[0]);
    torch::Tensor precalc;
    if (selection == -1)
    {
        DeePKS_domain::cal_vdr_precalc(nlocal, ucell.nat, nks, r_size, ld.deepks_param,
                                       this->kv.kvec_d, ld.phialpha, state.gevdm, ucell,
                                       this->orb_, this->pv, this->gd, precalc);
        LCAO_deepks_io::save_tensor2npy<double>(prefix + "vdrpre" + suffix + ".npy", precalc, rank);
    }
    else if (selection == 1)
    {
        DeePKS_domain::cal_v_delta_precalc<TK>(nlocal, ucell.nat, nks, ld.deepks_param,
                                              kernel_kvec, ld.phialpha, state.gevdm, ucell,
                                              this->orb_, this->pv, this->gd, precalc);
        LCAO_deepks_io::save_tensor2npy<TK>(prefix + "vdpre" + suffix + ".npy", precalc, rank);
    }
    else if (std::abs(selection) == 2)
    {
        DeePKS_domain::prepare_gevdm(ucell.nat, ld.deepks_param, this->orb_, state.gevdm, precalc);
        LCAO_deepks_io::save_tensor2npy<double>(prefix + "gevdm" + suffix + ".npy", precalc, rank);
        if (selection == -2)
        {
            torch::Tensor indices;
            DeePKS_domain::prepare_phialpha_iRmat(nlocal, r_size, ld.deepks_param, ld.phialpha,
                                                 ucell, this->orb_, this->pv, this->gd, precalc, indices);
            LCAO_deepks_io::save_tensor2npy<double>(prefix + "phialpha_r" + suffix + ".npy", precalc, rank);
            LCAO_deepks_io::save_tensor2npy<int>(prefix + "iRmat" + suffix + ".npy", indices, rank);
        }
        else
        {
            DeePKS_domain::prepare_phialpha<TK>(nlocal, ucell.nat, nks, ld.deepks_param,
                                               this->kv.kvec_d, ld.phialpha, ucell,
                                               this->orb_, this->pv, this->gd, precalc);
            LCAO_deepks_io::save_tensor2npy<TK>(prefix + "phialpha" + suffix + ".npy", precalc, rank);
        }
    }
}

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::write_deepks_sample(const DeePKS_domain::HRLabelSample& state,
                                                const double energy_correction_hartree,
                                                const int state_step,
                                                const std::string& state_kind)
{
    const std::string base_functional
        = this->pair_uses_model() ? this->inp_->dft_functional : this->pair_base_functional();
    const std::string target_functional
        = this->pair_uses_model() ? base_functional + "+deepks" : this->pair_target_functional();
    if (GlobalV::MY_RANK == 0)
    {
        this->sample_descriptors.push_back(state.descriptor);
        if (this->inp_->deepks_grad)
        {
            this->sample_metrics.push_back(state.metric);
            this->sample_projections.push_back(state.projection);
        }
        this->sample_energies.push_back(energy_correction_hartree);
        this->sample_steps.push_back(state_step);
        const int state_kind_code = state_kind == "exx_outer" ? 1 : (state_kind == "exx_inner" ? 2 : 0);
        this->sample_kinds.push_back(state_kind_code);

        const std::string& output_dir = PARAM.globalv.global_out_dir;
        LCAO_deepks_io::save_tensor2npy<double>(output_dir + "deepks_sample_dm_eig.npy",
                                                torch::stack(this->sample_descriptors),
                                                GlobalV::MY_RANK);
        if (this->inp_->deepks_grad)
        {
            LCAO_deepks_io::save_tensor2npy<double>(output_dir + "deepks_sample_hr_metric.npy",
                                                torch::stack(this->sample_metrics),
                                                GlobalV::MY_RANK);
            LCAO_deepks_io::save_tensor2npy<double>(output_dir + "deepks_sample_hr_projection.npy",
                                                torch::stack(this->sample_projections),
                                                GlobalV::MY_RANK);
        }
        LCAO_deepks_io::save_tensor2npy<double>(output_dir + "deepks_sample_energy.npy",
                                                torch::tensor(this->sample_energies, torch::kFloat64).reshape({-1, 1}),
                                                GlobalV::MY_RANK);
        LCAO_deepks_io::save_tensor2npy<int>(output_dir + "deepks_sample_step.npy",
                                             torch::tensor(this->sample_steps, torch::kInt32),
                                             GlobalV::MY_RANK);
        LCAO_deepks_io::save_tensor2npy<int>(output_dir + "deepks_sample_kind.npy",
                                             torch::tensor(this->sample_kinds, torch::kInt32),
                                             GlobalV::MY_RANK);

        std::ofstream metadata(output_dir + "deepks_sample_meta.dat");
        metadata << "nstate " << this->sample_descriptors.size() << '\n';
        metadata << "state_kind 0=scf_iteration,1=exx_outer,2=exx_inner\n";
        metadata << "target_functional " << target_functional << '\n';
        metadata << "base_functional " << base_functional << '\n';
        metadata << "sample_driver_functional " << this->inp_->dft_functional << '\n';
        metadata << "sample_driver_role "
                 << (this->inp_->deepks_scf ? "model_corrected_base"
                                          : (this->active_is_pair_base() ? "base" : "target"))
                 << '\n';
        metadata << "sample_driver_model_active " << (this->inp_->deepks_scf ? 1 : 0) << '\n';
        metadata << "sample_frequency " << this->inp_->deepks_out_pair_freq << '\n';
        metadata << "step_definition kind0_or_2=electronic_iteration,kind1=exx_outer_step\n";
        metadata << "descriptor_shape " << state.descriptor.size(0) << ' ' << state.descriptor.size(1) << '\n';
        metadata << "label_representation " << (this->inp_->deepks_grad ? "gradient" : "direct") << '\n';
        if (this->inp_->deepks_grad)
        {
            metadata << "metric_shape " << state.metric.size(0) << ' ' << state.metric.size(1) << '\n';
        }
        metadata << "direct_sample_index zero_based_save_order_sN\n";
        metadata << "hamiltonian_definition target_minus_base_at_same_density\n";
        metadata << "hk_convention sum_R_HR_exp_plus_i_2pi_kR\n";
        metadata << "energy_definition target_minus_base_at_same_density\n";
        metadata << "energy_unit Hartree\n";
        metadata << "hamiltonian_unit Hartree\n";
    }
    ++this->sample_state_counter;
}
#endif

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver)
{
    ModuleBase::TITLE("ESolver_DoubleXC", "iter_finish");
    ModuleBase::timer::start("ESolver_DoubleXC", "iter_finish");



    const int exx_outer_step_before = this->get_exx_outer_step();
    const bool sample_due
        = iter == 1 || iter % this->inp_->deepks_out_pair_freq == 0 || conv_esolver;

    // A separate-loop hybrid starts with a semilocal inner SCF. Preserve that
    // complete initial path before the EXX controller can advance its outer
    // step. The converged endpoint is kept here as an inner state and is not
    // duplicated as the first outer state.
    if (this->inp_->deepks_out_pair && this->exx_info_.info_global.cal_exx
        && exx_outer_step_before == 0 && sample_due)
    {
        this->output_deepks_sample(ucell, iter, "exx_inner");
    }

    ESolver_KS_LCAO<TK, TR>::iter_finish(ucell, istep, iter, conv_esolver);

    // ESolver_KS_LCAO::iter_finish determines inner-loop convergence and then
    // lets the active EXX controller either advance the outer step or accept
    // the final state. The density matrix itself is unchanged by that EXX
    // update, so the independent label-only evaluator still sees exactly the
    // just-converged density here.
    const int exx_outer_step_after = this->get_exx_outer_step();
    if (this->inp_->deepks_out_pair && this->exx_info_.info_global.cal_exx
        && exx_outer_step_before > 0
        && (exx_outer_step_after > exx_outer_step_before || conv_esolver))
    {
        this->output_deepks_sample(ucell, exx_outer_step_before, "exx_outer");
    }

    if (this->inp_->deepks_out_pair && !this->exx_info_.info_global.cal_exx
        && (sample_due || conv_esolver))
    {
        this->output_deepks_sample(ucell, iter, "scf_iteration");
    }


    ModuleBase::timer::end("ESolver_DoubleXC", "iter_finish");
}

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::evaluate_pair_force_stress(UnitCell& ucell,
                                                        const bool target,
                                                        ModuleBase::matrix& force,
                                                        ModuleBase::matrix& stress)
{
    const DeePKS_domain::ScopedXCState active_xc;
    const std::string functional = target ? this->pair_target_functional() : this->pair_base_functional();
    XC_Functional::set_xc_type(functional);
    XC_Functional::set_hybrid_alpha(target ? this->pair_target_exx_alpha : this->base_exx_alpha);
    XC_Functional::set_hse_omega(target ? this->pair_target_exx_omega : this->base_exx_omega);
    for (int ik = 0; ik < this->kv.get_nks(); ++ik)
    {
        this->dmat_base.dm->set_DMK_pointer(ik, this->dmat.dm->get_DMK_pointer(ik));
    }
    this->dmat_base.dm->cal_DMR();
    LCAO_domain::dm2rho(this->dmat_base.dm->get_DMR_vector(), this->inp_->nspin, &this->chr_base);
    if (XC_Functional::get_ked_flag())
    {
        LCAO_domain::dm2tau(this->dmat_base.dm->get_DMR_vector(), this->inp_->nspin, &this->chr_base);
    }
    this->pelec_base->pot->update_from_charge(&this->chr_base, &ucell);
    // Both derivatives use the same active energy-weighted DM. Their common
    // overlap term cancels; no auxiliary self-consistency response is claimed.
    this->pelec_base->ekb = this->pelec->ekb;
    this->pelec_base->wg = this->pelec->wg;
    this->pelec_base->vnew_exist = false;

    Exx_Info info = this->exx_info_;
    info.info_global.cal_exx = false;
    Exx_NAO<TK> evaluator;
#ifdef __EXX
    evaluator.exd = target ? this->pair_target_exx_real : this->base_exx_real;
    evaluator.exc = target ? this->pair_target_exx_complex : this->base_exx_complex;
    if (evaluator.exd || evaluator.exc)
    {
        info.info_global = target ? this->pair_target_exx_global : this->base_exx_global;
        info.info_ri = target ? this->pair_target_exx_info : this->base_exx_info;
        if (evaluator.exd)
        {
            (target ? this->pair_target_exx_real : this->base_exx_real)->cal_exx_elec(
                *this->dmat_base.dm, ucell, this->kv, this->pv, this->inp_->nspin);
        }
        else
        {
            (target ? this->pair_target_exx_complex : this->base_exx_complex)->cal_exx_elec(
                *this->dmat_base.dm, ucell, this->kv, this->pv, this->inp_->nspin);
        }
    }
#endif
    Force_Stress_LCAO<TK> fsl(this->RA, ucell.nat);
    const FSCalcConfig cfg{this->inp_->nspin, this->inp_->nbands, this->inp_->t_in_h,
                           this->inp_->sc_mag_switch, this->inp_->device};
    const std::string previous_output = this->deepks.dpks_out_type;
    this->deepks.dpks_out_type = "pair";
    fsl.getForceStress(ucell, this->get_vdw_result(), this->inp_->cal_force, this->inp_->cal_stress,
                       false, false, this->gd, this->pv, this->pelec_base, this->dmat_base, this->psi,
                       this->two_center_bundle_, this->orb_, force, stress, this->locpp, this->sf,
                       this->kv, this->pw_rho, this->solvent, *this->dftu_, this->deepks,
                       evaluator, &ucell.symm, info, cfg);
    this->deepks.dpks_out_type = previous_output;
}

template <typename TK, typename TR>
void ESolver_DoubleXC<TK, TR>::cal_force(BaseCell& basecell, ModuleBase::matrix& force)
{
    basecell.require_kind(BaseCell::Kind::unitcell, __FUNCTION__);
    UnitCell& ucell = static_cast<UnitCell&>(basecell);
    ModuleBase::TITLE("ESolver_DoubleXC", "cal_force");
    ModuleBase::timer::start("ESolver_DoubleXC", "cal_force");

#ifdef __MLALGO
    auto state = DeePKS_domain::build_hr_label_sample<TK, TR>(
        PARAM.globalv.nlocal, ucell.nat, this->kv.kvec_d, ucell, this->orb_, this->pv, this->gd,
        *this->dmat.dm, this->deepks.ld, torch::Tensor(), torch::Tensor(), false);
    const auto operators = DeePKS_domain::build_force_stress_operators<TK>(
        ucell, this->orb_, this->pv, this->gd, this->kv.kvec_d, this->deepks.ld, state.gevdm,
        this->inp_->cal_force, this->inp_->cal_stress, GlobalV::MY_RANK);
    torch::Tensor delta_force;
    torch::Tensor delta_virial;
    if (this->pair_uses_model())
    {
        DeePKS_domain::project_model_gradient(state, this->deepks.ld);
        if (GlobalV::MY_RANK == 0)
        {
            const torch::Tensor gradient = state.model_gradient.reshape({-1});
            if (this->inp_->cal_force)
            {
                delta_force = torch::matmul(operators.force, gradient).reshape({ucell.nat, 3});
            }
            if (this->inp_->cal_stress)
            {
                delta_virial = torch::matmul(operators.virial, gradient);
            }
        }
    }
    else
    {
        ModuleBase::matrix base_force;
        ModuleBase::matrix target_force;
        ModuleBase::matrix base_stress;
        ModuleBase::matrix target_stress;
        this->evaluate_pair_force_stress(ucell, false, base_force, base_stress);
        this->evaluate_pair_force_stress(ucell, true, target_force, target_stress);
        if (GlobalV::MY_RANK == 0)
        {
            const auto options = torch::TensorOptions().dtype(torch::kFloat64);
            if (this->inp_->cal_force)
            {
                delta_force = torch::empty({ucell.nat, 3}, options);
                for (int i = 0; i < ucell.nat; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        delta_force.data_ptr<double>()[3 * i + j] = 0.5 * (target_force(i, j) - base_force(i, j));
                    }
                }
            }
            if (this->inp_->cal_stress)
            {
                delta_virial = torch::empty({6}, options);
                int index = 0;
                for (int i = 0; i < 3; ++i)
                {
                    for (int j = i; j < 3; ++j)
                    {
                        delta_virial.data_ptr<double>()[index++] = 0.5 * ucell.omega
                            * (target_stress(i, j) - base_stress(i, j));
                    }
                }
            }
        }
    }
    if (this->inp_->cal_force)
    {
        DeePKS_domain::write_pair_property(PARAM.globalv.global_out_dir + "deepks_pair_force",
                                          operators.force, delta_force, this->inp_->deepks_grad, GlobalV::MY_RANK);
    }
    if (this->inp_->cal_stress)
    {
        DeePKS_domain::write_pair_property(PARAM.globalv.global_out_dir + "deepks_pair_virial",
                                          operators.virial, delta_virial, this->inp_->deepks_grad, GlobalV::MY_RANK);
    }
    if (GlobalV::MY_RANK == 0)
    {
        std::ofstream meta(PARAM.globalv.global_out_dir + "deepks_pair_final_meta.txt");
        meta << "density final_active_density_matrix\n"
             << "difference target_minus_base\n"
             << "derivative fixed_density_matrix_shared_active_energy_weighted_density\n"
             << "independent_auxiliary_scf false\n"
             << "model_pair " << this->pair_uses_model() << "\n"
             << "force_unit Hartree/Bohr\nvirial_unit Hartree\n"
             << "virial_order xx_xy_xz_yy_yz_zz\n"
             << "force_operator -deepks_gradvx\nvirial_operator deepks_gvepsl\n";
    }
#endif
    ESolver_KS_LCAO<TK, TR>::cal_force(ucell, force);
    ModuleBase::timer::end("ESolver_DoubleXC", "cal_force");
}

template class ESolver_DoubleXC<double, double>;
template class ESolver_DoubleXC<std::complex<double>, double>;
template class ESolver_DoubleXC<std::complex<double>, std::complex<double>>;

} // namespace ModuleESolver
#endif
