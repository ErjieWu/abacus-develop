#include "deepks_test_runner.h"

#include "source_lcao/module_deepks/deepks_force.h"
#include "source_lcao/module_deepks/deepks_basic.h"
#include "source_lcao/module_deepks/deepks_descriptor.h"
#include "source_lcao/module_deepks/deepks_grad.h"

#include <complex>
#include <iomanip>

template <typename T>
void test_deepks<T>::check_f_delta_and_stress_delta()
{
    ModuleBase::matrix fvnl_dalpha;
    fvnl_dalpha.create(ucell.nat, 3);

    ModuleBase::matrix svnl_dalpha;
    svnl_dalpha.create(3, 3);
    const int cal_stress = 1;
    const int nks = kv.get_nkstot();
    DeePKS_domain::cal_f_delta<T>(ucell,
                                  ORB,
                                  Test_Deepks::GridD,
                                  ParaO,
                                  nks,
                                  this->ld.deepks_param,
                                  kv.kvec_d,
                                  this->ld.phialpha,
                                  fvnl_dalpha,
                                  cal_stress,
                                  svnl_dalpha,
                                  this->ld.dm_r,
                                  this->ld.gedm);
    std::ofstream ofs_f("F_delta.dat");
    std::ofstream ofs_s("stress_delta.dat");
    ofs_f << std::setprecision(10);
    ofs_s << std::setprecision(10);
    fvnl_dalpha.print(ofs_f);
    ofs_f.close();
    svnl_dalpha.print(ofs_s);
    ofs_s.close();

    this->assert_file_matches_reference("F_delta.dat", "F_delta_ref.dat");
    this->assert_file_matches_reference("stress_delta.dat", "stress_delta_ref.dat");

    std::vector<torch::Tensor> descriptor;
    DeePKS_domain::cal_descriptor(ucell.nat, ld.deepks_param, ld.pdm, descriptor);
    DeePKS_domain::HRLabelSample sample;
    sample.descriptor = torch::cat(descriptor, 0).reshape({ucell.nat, ld.deepks_param.des_per_atom});
    DeePKS_domain::cal_gevdm(ucell.nat, ld.deepks_param, ld.pdm, sample.gevdm);
    const double energy = DeePKS_domain::project_model_gradient(sample, ld);
    EXPECT_NEAR(energy, 0.5 * ld.E_delta, 1e-12);
    const auto operators = DeePKS_domain::build_force_stress_operators<T>(
        ucell, ORB, ParaO, Test_Deepks::GridD, kv.kvec_d, ld, sample.gevdm, true, true, 0);
    const auto force_only = DeePKS_domain::build_force_stress_operators<T>(
        ucell, ORB, ParaO, Test_Deepks::GridD, kv.kvec_d, ld, sample.gevdm, true, false, 0);
    const auto virial_only = DeePKS_domain::build_force_stress_operators<T>(
        ucell, ORB, ParaO, Test_Deepks::GridD, kv.kvec_d, ld, sample.gevdm, false, true, 0);
    EXPECT_FALSE(force_only.virial.defined());
    EXPECT_FALSE(virial_only.force.defined());
    EXPECT_TRUE(torch::equal(force_only.force, operators.force));
    EXPECT_TRUE(torch::equal(virial_only.virial, operators.virial));
    const auto force = operators.force.mv(sample.model_gradient).reshape({ucell.nat, 3});
    for (int atom = 0; atom < ucell.nat; ++atom)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            EXPECT_NEAR(force[atom][axis].template item<double>(), 0.5 * fvnl_dalpha(atom, axis), 1e-10);
        }
    }
    const auto virial = operators.virial.mv(sample.model_gradient);
    int component = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = i; j < 3; ++j)
        {
            // Internal stress is Ry/Bohr^3; exported virial is Hartree.
            EXPECT_NEAR(virial[component++].template item<double>(),
                        0.5 * ucell.omega * svnl_dalpha(i, j), 1e-10);
        }
    }
}

template void test_deepks<double>::check_f_delta_and_stress_delta();
template void test_deepks<std::complex<double>>::check_f_delta_and_stress_delta();

template <typename T>
void run_deepks_unit_force_stress_delta(test_deepks<T>& test)
{
    std::vector<torch::Tensor> descriptor;
    DeepksTestRunner::build_edelta(test, descriptor);
    if (testing::Test::HasFatalFailure())
    {
        return;
    }
    test.check_f_delta_and_stress_delta();
}

template void run_deepks_unit_force_stress_delta<double>(test_deepks<double>& test);
template void run_deepks_unit_force_stress_delta<std::complex<double>>(test_deepks<std::complex<double>>& test);
