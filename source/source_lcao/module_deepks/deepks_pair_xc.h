#ifndef DEEPKS_PAIR_XC_H
#define DEEPKS_PAIR_XC_H

#include "source_hamilt/module_xc/xc_functional.h"
#include <utility>

namespace DeePKS_domain
{
// Preserve the actual runtime functional, including hybrid inner-loop state.
class ScopedXCState
{
  public:
    ScopedXCState()
        : saved_func_id(XC_Functional::func_id),
          saved_func_type(XC_Functional::func_type),
          saved_ked_flag(XC_Functional::ked_flag),
          saved_need_laplacian(XC_Functional::need_laplacian),
          saved_use_libxc(XC_Functional::use_libxc),
          saved_hybrid_alpha(XC_Functional::hybrid_alpha),
          saved_hse_omega(XC_Functional::hse_omega),
          saved_runtime_parameters(XC_Functional::runtime_parameters),
          saved_scaling_factor_xc(XC_Functional::scaling_factor_xc)
    {
    }

    ~ScopedXCState() noexcept
    {
        XC_Functional::func_id.swap(saved_func_id);
        XC_Functional::func_type = saved_func_type;
        XC_Functional::ked_flag = saved_ked_flag;
        XC_Functional::need_laplacian = saved_need_laplacian;
        XC_Functional::use_libxc = saved_use_libxc;
        XC_Functional::hybrid_alpha = saved_hybrid_alpha;
        XC_Functional::hse_omega = saved_hse_omega;
        std::swap(XC_Functional::runtime_parameters, saved_runtime_parameters);
        XC_Functional::scaling_factor_xc.swap(saved_scaling_factor_xc);
    }

    ScopedXCState(const ScopedXCState&) = delete;
    ScopedXCState& operator=(const ScopedXCState&) = delete;

  private:
    std::vector<int> saved_func_id;
    int saved_func_type;
    bool saved_ked_flag;
    bool saved_need_laplacian;
    bool saved_use_libxc;
    double saved_hybrid_alpha;
    double saved_hse_omega;
    XCFunctionalParameters saved_runtime_parameters;
    std::map<int, double> saved_scaling_factor_xc;
};
}
#endif
