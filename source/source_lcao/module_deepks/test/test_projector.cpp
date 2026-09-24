#include "source_lcao/module_deepks/deepks_projector.h"
#include "source_basis/module_nao/atomic_radials.h"
#include "source_basis/module_nao/two_center_bundle.h"
#include "source_base/global_variable.h"

#include "gtest/gtest.h"

#ifdef __MPI
#include <mpi.h>
#endif

TEST(DeepksProjector, ZeroPaddedChannelsUseCommonCutoff)
{
    AtomicRadials alpha;
    alpha.build("support/padded_projectors.orb", 0, nullptr, GlobalV::MY_RANK);
    ASSERT_DOUBLE_EQ(alpha.chi(0, 0).rcut(), 4.0);
    ASSERT_DOUBLE_EQ(alpha.chi(0, 1).rcut(), 5.0);
    ASSERT_DOUBLE_EQ(alpha.chi(1, 0).rcut(), 4.0);
    Numerical_Orbital legacy;
    DeePKS_domain::projectors_to_numerical_orbital(alpha, legacy, 101, 0.1, false, true);
    EXPECT_DOUBLE_EQ(legacy.getRcut(), 5.0);
    for (int i = 0; i < alpha.nchi(); ++i)
    {
        const auto& source = alpha.cbegin()[i];
        const auto& dest = legacy.PhiLN(source.l(), source.izeta());
        EXPECT_DOUBLE_EQ(dest.getRcut(), 5.0);
        ASSERT_EQ(dest.getNr(), 11);
        for (int ir = 0; ir < 11; ++ir)
        {
            EXPECT_DOUBLE_EQ(dest.getPsi(ir), source.rvalue(ir));
        }
    }
    EXPECT_DOUBLE_EQ(alpha.chi(0, 0).rcut(), 4.0);
    EXPECT_DOUBLE_EQ(legacy.PhiLN(0, 0).getPsi(9), 0.0);
    EXPECT_GT(legacy.PhiLN(0, 1).getPsi(9), 0.0);

    alpha.set_uniform_grid(true, 21, 10.0, 'i', true);
    DeePKS_domain::projectors_to_numerical_orbital(alpha, legacy, 101, 0.1, false, true);
    EXPECT_DOUBLE_EQ(legacy.getRcut(), 5.0);
    for (int i = 0; i < alpha.nchi(); ++i)
    {
        const auto& source = alpha.cbegin()[i];
        EXPECT_DOUBLE_EQ(legacy.PhiLN(source.l(), source.izeta()).getRcut(), 5.0);
        EXPECT_EQ(legacy.PhiLN(source.l(), source.izeta()).getNr(), 11);
    }
}

TEST(DeepksProjector, BundleKeepsOrdinaryOrbitalsUnchanged)
{
    std::string ordinary = std::string(DEEPKS_PROJECTOR_TEST_ORB);
    std::string projector = "support/padded_projectors.orb";
    TwoCenterBundle bundle;
    bundle.build_orb(1, &ordinary, "");
    bundle.build_alpha(1, &projector);
    LCAO_Orbitals legacy;
    bundle.to_LCAO_Orbitals(legacy, 100.0, 0.1, 0.01, 20.0, false, false);

    Numerical_Orbital reference;
    (*bundle.orb_)(0).to_numerical_orbital(reference, legacy.get_kmesh(), legacy.get_dk(), false, false);
    for (int i = 0; i < reference.getTotal_nchi(); ++i)
    {
        const auto& want = reference.chi()[i];
        const auto& got = legacy.Phi[0].chi()[i];
        ASSERT_EQ(got.getNr(), want.getNr());
        EXPECT_DOUBLE_EQ(got.getRcut(), want.getRcut());
        for (int ir = 0; ir < want.getNr(); ++ir)
        {
            EXPECT_DOUBLE_EQ(got.getPsi(ir), want.getPsi(ir));
        }
    }
    EXPECT_DOUBLE_EQ(legacy.Alpha[0].getRcut(), 5.0);
    for (const auto& channel : legacy.Alpha[0].chi())
    {
        EXPECT_DOUBLE_EQ(channel.getRcut(), 5.0);
    }

    // With ordinary equal-cutoff channels, the new projector adapter is identical
    // to the original conversion, including its k-space radial functions.
    Numerical_Orbital equal_cutoff;
    DeePKS_domain::projectors_to_numerical_orbital(
        (*bundle.orb_)(0), equal_cutoff, legacy.get_kmesh(), legacy.get_dk(), false, false);
    for (int i = 0; i < reference.getTotal_nchi(); ++i)
    {
        const auto& want = reference.chi()[i];
        const auto& got = equal_cutoff.chi()[i];
        ASSERT_EQ(got.getNr(), want.getNr());
        ASSERT_EQ(got.getNk(), want.getNk());
        for (int ir = 0; ir < want.getNr(); ++ir)
        {
            EXPECT_DOUBLE_EQ(got.getPsi(ir), want.getPsi(ir));
        }
        for (int ik = 0; ik < want.getNk(); ++ik)
        {
            EXPECT_DOUBLE_EQ(got.getPsif(ik), want.getPsif(ik));
        }
    }
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &GlobalV::MY_RANK);
#endif
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}
