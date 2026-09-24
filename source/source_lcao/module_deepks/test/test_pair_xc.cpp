#include "../deepks_pair_xc.h"
#include "gtest/gtest.h"

#include <stdexcept>

TEST(XCFunctionalState, RestoresActualSemilocalInnerLoop)
{
    const DeePKS_domain::ScopedXCState original;
    XC_Functional::set_xc_type("pbe");
    XC_Functional::set_hybrid_alpha(0.31);
    XC_Functional::set_hse_omega(0.17);
    const auto ids = XC_Functional::get_func_id();
    const std::string info = XC_Functional::output_info();
    {
        const DeePKS_domain::ScopedXCState inner_loop;
        XC_Functional::set_xc_type("hse");
        XC_Functional::set_hybrid_alpha(0.25);
        XC_Functional::set_hse_omega(0.11);
        EXPECT_EQ(XC_Functional::get_func_type(), 4);
    }
    EXPECT_EQ(XC_Functional::get_func_type(), 2);
    EXPECT_EQ(XC_Functional::get_func_id(), ids);
    EXPECT_EQ(XC_Functional::output_info(), info);
    EXPECT_DOUBLE_EQ(XC_Functional::get_hybrid_alpha(), 0.31);
    EXPECT_DOUBLE_EQ(XC_Functional::get_hse_omega(), 0.17);
}

TEST(XCFunctionalState, NestedScopesAndExceptionRestoreAllRuntimeParameters)
{
    const DeePKS_domain::ScopedXCState original;
    XCFunctionalParameters parameters;
    parameters.xc_temperature = 0.12;
    parameters.exx_fock_alpha = {"0.3"};
    parameters.exx_erfc_alpha = {"0.2"};
    parameters.xc_exch_ext = {101.0, 0.7};
    parameters.xc_corr_ext = {130.0, 0.6};
    XC_Functional::set_runtime_parameters(parameters);
    XC_Functional::set_xc_type("scan");
    const auto ids = XC_Functional::get_func_id();
    const bool needs_laplacian = XC_Functional::get_need_laplacian();
    try
    {
        const DeePKS_domain::ScopedXCState outer;
        XC_Functional::set_xc_type("pbe");
        XC_Functional::set_runtime_parameters(XCFunctionalParameters{});
        {
            const DeePKS_domain::ScopedXCState inner;
            XC_Functional::set_xc_type("lda");
        }
        EXPECT_EQ(XC_Functional::get_func_type(), 2);
        throw std::runtime_error("auxiliary evaluation failed");
    }
    catch (const std::runtime_error&)
    {
    }
    EXPECT_EQ(XC_Functional::get_func_id(), ids);
    EXPECT_EQ(XC_Functional::get_func_type(), 3);
    EXPECT_TRUE(XC_Functional::get_ked_flag());
    EXPECT_EQ(XC_Functional::get_need_laplacian(), needs_laplacian);
    const auto& restored = XC_Functional::get_runtime_parameters();
    EXPECT_DOUBLE_EQ(restored.xc_temperature, parameters.xc_temperature);
    EXPECT_EQ(restored.exx_fock_alpha, parameters.exx_fock_alpha);
    EXPECT_EQ(restored.exx_erfc_alpha, parameters.exx_erfc_alpha);
    EXPECT_EQ(restored.xc_exch_ext, parameters.xc_exch_ext);
    EXPECT_EQ(restored.xc_corr_ext, parameters.xc_corr_ext);
}
