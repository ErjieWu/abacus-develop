#include "source_lcao/module_deepks/deepks_grad.h"
#include "npy.hpp"
#include <gtest/gtest.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>

namespace
{
class GradientLabels : public testing::Test
{
  protected:
    std::string prefix;
    void SetUp() override
    {
        char directory[] = "/tmp/abacus-gradient-test-XXXXXX";
        const char* result = mkdtemp(directory);
        ASSERT_NE(result, nullptr);
        prefix = std::string(result) + "/property";
    }
    void TearDown() override
    {
        for (const auto* suffix : {".npy", "_metric.npy", "_projection.npy"})
        {
            std::remove((prefix + suffix).c_str());
        }
        rmdir(prefix.substr(0, prefix.find_last_of('/')).c_str());
    }
    torch::Tensor load(const std::string& suffix)
    {
        std::vector<unsigned long> shape;
        std::vector<double> values;
        npy::LoadArrayFromNumpy(prefix + suffix, shape, values);
        std::vector<int64_t> dimensions(shape.begin(), shape.end());
        return torch::from_blob(values.data(), dimensions, torch::kFloat64).clone();
    }
};

TEST_F(GradientLabels, WritesMetricAndProjectionWithoutUnitRescaling)
{
    // Includes a zero column and signed values: do not assume full rank.
    const auto a = torch::tensor({{1., -2., 0.}, {3., 4., 0.}}, torch::kFloat64);
    const auto x = torch::tensor({0.25, -0.5}, torch::kFloat64);
    DeePKS_domain::write_pair_property(prefix, a, x, true, 0);
    EXPECT_FALSE(std::ifstream(prefix + ".npy").good());
    const auto metric = load("_metric.npy");
    const auto projection = load("_projection.npy");
    EXPECT_TRUE(torch::equal(metric, torch::tensor({{10., 10., 0.}, {10., 20., 0.},
                                                    {0., 0., 0.}}, torch::kFloat64)));
    EXPECT_TRUE(torch::equal(projection, torch::tensor({-1.25, -2.5, 0.}, torch::kFloat64)));
    auto g = torch::tensor({0.2, -0.3, 1.7}, torch::TensorOptions().dtype(torch::kFloat64)
                                                              .requires_grad(true));
    const auto direct = (a.mv(g) - x).square().sum();
    const auto quadratic = g.dot(metric.mv(g)) - 2 * g.dot(projection);
    EXPECT_NEAR((direct - quadratic).item<double>(), x.dot(x).item<double>(), 1e-14);
    const auto dg = torch::autograd::grad({direct}, {g})[0];
    const auto qg = torch::autograd::grad({quadratic}, {g})[0];
    EXPECT_TRUE(torch::allclose(dg, qg, 1e-12, 1e-14));
}

TEST_F(GradientLabels, DirectOutputPreservesHartreeValuesAndShape)
{
    const auto x = torch::tensor({{0.25, -0.5}, {0.75, 1.}}, torch::kFloat64);
    DeePKS_domain::write_pair_property(prefix, {}, x, false, 0);
    EXPECT_TRUE(torch::equal(load(".npy"), x));
    EXPECT_FALSE(std::ifstream(prefix + "_metric.npy").good());
    EXPECT_FALSE(std::ifstream(prefix + "_projection.npy").good());
}

TEST_F(GradientLabels, NonRootDoesNotWrite)
{
    DeePKS_domain::write_pair_property(prefix, {}, {}, true, 1);
    DeePKS_domain::write_pair_property(prefix, {}, {}, false, 1);
    EXPECT_FALSE(std::ifstream(prefix + ".npy").good());
    EXPECT_FALSE(std::ifstream(prefix + "_metric.npy").good());
    EXPECT_FALSE(std::ifstream(prefix + "_projection.npy").good());
}

TEST(GradientMetric, ForceAndVirialFlattenAllDescriptorColumns)
{
    const auto a = torch::arange(24, torch::kFloat64).reshape({6, 4}) - 7.;
    torch::Tensor force_metric, virial_metric;
    DeePKS_domain::cal_gvx_square(a.reshape({2, 3, 2, 2}), force_metric);
    DeePKS_domain::cal_gvepsl_square(a.reshape({6, 2, 2}), virial_metric);
    EXPECT_TRUE(torch::equal(force_metric, a.t().mm(a)));
    EXPECT_TRUE(torch::equal(virial_metric, force_metric));
    torch::Tensor undefined_metric;
    DeePKS_domain::cal_gvx_square({}, undefined_metric);
    EXPECT_FALSE(undefined_metric.defined());
    DeePKS_domain::cal_gvepsl_square({}, undefined_metric);
    EXPECT_FALSE(undefined_metric.defined());
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    torch::set_num_threads(1);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
