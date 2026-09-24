#ifndef ESOLVER_DOUBLE_XC_H
#define ESOLVER_DOUBLE_XC_H

#include "source_esolver/esolver_ks_lcao.h"
#include "source_lcao/module_deepks/deepks_pair_xc.h"

#ifdef __EXX
#include "source_hamilt/module_xc/exx_info_ri.h"
#include "source_lcao/module_deepks/deepks_pair_exx.h"
#endif

#include <memory>
#include <string>
#include <vector>

#ifdef __MLALGO
namespace DeePKS_domain
{
struct HRLabelSample;
}
#endif

namespace ModuleESolver
{
// used in deepks, run target and base xc functional simultaneously
template <typename TK, typename TR>
class ESolver_DoubleXC : public ESolver_KS_LCAO<TK, TR>
{
  public:
    ESolver_DoubleXC();
    ~ESolver_DoubleXC();

    void before_all_runners(BaseCell& basecell, const Input_para& inp) override;

    void cal_force(BaseCell& basecell, ModuleBase::matrix& force) override;

  protected:
    void before_scf(UnitCell& ucell, const int istep) override;

    void iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver) override;

    int get_exx_outer_step() const;

    std::string pair_target_functional() const;

    std::string pair_base_functional() const;

    std::string auxiliary_functional() const;

    bool active_is_pair_base() const;

    bool pair_uses_model() const;

    void output_deepks_sample(UnitCell& ucell, const int state_step, const std::string& state_kind);

    void evaluate_pair_force_stress(UnitCell& ucell,
                                    const bool target,
                                    ModuleBase::matrix& force,
                                    ModuleBase::matrix& stress);

#ifdef __MLALGO
    void output_model_sample(UnitCell& ucell, const int state_step, const std::string& state_kind);

    void write_deepks_sample(const DeePKS_domain::HRLabelSample& state,
                            const double energy_correction_hartree,
                            const int state_step,
                            const std::string& state_kind);

    void write_direct_sample(const UnitCell& ucell,
                             const DeePKS_domain::HRLabelSample& state,
                             const hamilt::HContainer<TR>& correction);

    std::vector<torch::Tensor> sample_descriptors;
    std::vector<torch::Tensor> sample_metrics;
    std::vector<torch::Tensor> sample_projections;
    std::vector<double> sample_energies;
    std::vector<int> sample_steps;
    std::vector<int> sample_kinds;
#endif

#ifdef __EXX
    void initialize_fixed_density_exx(
        const UnitCell& ucell,
        const std::string& functional,
        const bool is_active_functional,
        Exx_Info_RI& info,
        Exx_Info_Global& global,
        std::shared_ptr<DeePKS_domain::PairExx<TK, double>>& evaluator_real,
        std::shared_ptr<DeePKS_domain::PairExx<TK, std::complex<double>>>& evaluator_complex,
        double& alpha,
        double& omega);

    void initialize_pair_target_exx(const UnitCell& ucell);

    void initialize_base_exx(const UnitCell& ucell);

    Exx_Info_RI pair_target_exx_info;
    Exx_Info_Global pair_target_exx_global;
    std::shared_ptr<DeePKS_domain::PairExx<TK, double>> pair_target_exx_real = nullptr;
    std::shared_ptr<DeePKS_domain::PairExx<TK, std::complex<double>>> pair_target_exx_complex = nullptr;

    Exx_Info_RI base_exx_info;
    Exx_Info_Global base_exx_global;
    std::shared_ptr<DeePKS_domain::PairExx<TK, double>> base_exx_real = nullptr;
    std::shared_ptr<DeePKS_domain::PairExx<TK, std::complex<double>>> base_exx_complex = nullptr;
#endif
    double pair_target_exx_alpha = 0.0;
    double pair_target_exx_omega = 0.0;
    double base_exx_alpha = 0.0;
    double base_exx_omega = 0.0;

    int sample_state_counter = 0;

    //! Hamiltonian
    hamilt::Hamilt<TK>* p_hamilt_base = nullptr;

    //! Electronic states
    elecstate::ElecState* pelec_base = nullptr;

    //! Density Matrix, mohan add 2025-11-03
    LCAO_domain::Setup_DM<TK> dmat_base;

    //! Electorn charge density
    Charge chr_base;
};
} // namespace ModuleESolver
#endif
