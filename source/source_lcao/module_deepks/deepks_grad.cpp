// deepks_grad.cpp  --  Descriptor-gradient label building blocks
// See deepks_grad.h for the full derivation and Python usage.

#ifdef __MLALGO

#include "deepks_grad.h"

#include "lcao_deepks.h"
#include "lcao_deepks_io.h"
#include "deepks_descriptor.h"
#include "deepks_fpre.h"
#include "deepks_spre.h"
#include "deepks_iterate.h"
#include "deepks_pdm.h"
#include "deepks_vdpre.h"
#include "deepks_vdrpre.h"
#include "source_base/parallel_reduce.h"
#include "source_estate/module_dm/density_matrix.h"

#include <cstdint>
#include <unordered_map>

namespace
{
torch::Tensor flatten_operator(const torch::Tensor& op, const int nrow_dims)
{
    int64_t nrows = 1;
    for (int i = 0; i < nrow_dims; ++i)
    {
        nrows *= op.size(i);
    }
    return op.reshape({nrows, op.numel() / nrows});
}

} // namespace

template <typename TK>
DeePKS_domain::ForceStressLabelOperators DeePKS_domain::build_force_stress_operators(
    const UnitCell& ucell,
    const LCAO_Orbitals& orb,
    const Parallel_Orbitals& pv,
    const Grid_Driver& grid,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d,
    const LCAO_Deepks<TK>& ld,
    const std::vector<torch::Tensor>& gevdm,
    const bool force,
    const bool stress,
    const int rank)
{
    ForceStressLabelOperators result;
    const int nks = kvec_d.size();
    if (force)
    {
        torch::Tensor gdmx;
        torch::Tensor gvx;
        cal_gdmx<TK>(nks, ld.deepks_param, kvec_d, ld.phialpha, ld.dm_r, ucell, orb, pv, grid, gdmx);
        cal_gvx(ucell.nat, ld.deepks_param, gevdm, gdmx, gvx, rank);
        if (rank == 0)
        {
            result.force = -gvx.reshape({3 * ucell.nat, -1});
        }
    }
    if (stress)
    {
        torch::Tensor gdmepsl;
        torch::Tensor gvepsl;
        cal_gdmepsl<TK>(nks, ld.deepks_param, kvec_d, ld.phialpha, ld.dm_r, ucell, orb, pv, grid, gdmepsl);
        cal_gvepsl(ucell.nat, ld.deepks_param, gevdm, gdmepsl, gvepsl, rank);
        if (rank == 0)
        {
            result.virial = gvepsl.reshape({6, -1});
        }
    }
    return result;
}

void DeePKS_domain::write_pair_property(const std::string& prefix,
                                      const torch::Tensor& op,
                                      const torch::Tensor& correction,
                                      const bool gradient_labels,
                                      const int rank)
{
    if (rank != 0)
    {
        return;
    }
    if (gradient_labels)
    {
        const torch::Tensor metric = torch::matmul(op.transpose(0, 1), op).contiguous();
        const torch::Tensor projection = torch::matmul(op.transpose(0, 1), correction.reshape({-1})).contiguous();
        LCAO_deepks_io::save_tensor2npy<double>(prefix + "_metric.npy", metric, rank);
        LCAO_deepks_io::save_tensor2npy<double>(prefix + "_projection.npy", projection, rank);
    }
    else
    {
        LCAO_deepks_io::save_tensor2npy<double>(prefix + ".npy", correction.contiguous(), rank);
    }
}

template DeePKS_domain::ForceStressLabelOperators DeePKS_domain::build_force_stress_operators<double>(
    const UnitCell&, const LCAO_Orbitals&, const Parallel_Orbitals&, const Grid_Driver&,
    const std::vector<ModuleBase::Vector3<double>>&, const LCAO_Deepks<double>&,
    const std::vector<torch::Tensor>&, const bool, const bool, const int);
template DeePKS_domain::ForceStressLabelOperators DeePKS_domain::build_force_stress_operators<std::complex<double>>(
    const UnitCell&, const LCAO_Orbitals&, const Parallel_Orbitals&, const Grid_Driver&,
    const std::vector<ModuleBase::Vector3<double>>&, const LCAO_Deepks<std::complex<double>>&,
    const std::vector<torch::Tensor>&, const bool, const bool, const int);

// ---------------------------------------------------------------------------
// Step 1: dot_phialpha_hamilt[inl, m1, m2]
//         = <phialpha^I_{nl,m1} | H | phialpha^I_{nl,m2}>
// ---------------------------------------------------------------------------
template <typename TR>
void DeePKS_domain::cal_phialpha_hamilt_proj(const int nlocal,
                                             const int nat,
                                             const DeePKS_Param& deepks_param,
                                             const std::vector<hamilt::HContainer<double>*>& phialpha,
                                             const hamilt::HContainer<TR>& hR,
                                             const UnitCell& ucell,
                                             const LCAO_Orbitals& orb,
                                             const Parallel_Orbitals& pv,
                                             const Grid_Driver& GridD,
                                             torch::Tensor& dot_phialpha_hamilt)
{
    ModuleBase::TITLE("DeePKS_domain", "cal_phialpha_hamilt_proj");
    ModuleBase::timer::start("DeePKS_domain", "cal_phialpha_hamilt_proj");

    using TorchScalar = typename std::conditional<std::is_same<TR, double>::value, double, c10::complex<double>>::type;

    const int inlmax = deepks_param.inlmax;
    const int lmaxd = deepks_param.lmaxd;
    const int nm_max = 2 * lmaxd + 1;

    const torch::Dtype dtype = std::is_same<TR, double>::value ? torch::kFloat64 : torch::kComplexDouble;
    torch::Tensor dot_ph = torch::zeros({inlmax, nm_max, nm_max}, torch::TensorOptions().dtype(dtype));
    auto dot_ph_acc = dot_ph.accessor<TorchScalar, 3>();

    DeePKS_domain::iterate_ad2(
        ucell,
        GridD,
        orb,
        false,
        [&](const int iat,
            const ModuleBase::Vector3<double>& tau0,
            const int ibt1,
            const ModuleBase::Vector3<double>& tau1,
            const int start1,
            const int nw1_tot,
            ModuleBase::Vector3<int> dR1,
            const int ibt2,
            const ModuleBase::Vector3<double>& tau2,
            const int start2,
            const int nw2_tot,
            ModuleBase::Vector3<int> dR2) {
            if (phialpha[0]->find_matrix(iat, ibt1, dR1.x, dR1.y, dR1.z) == nullptr
                || phialpha[0]->find_matrix(iat, ibt2, dR2.x, dR2.y, dR2.z) == nullptr)
            {
                return;
            }

            const ModuleBase::Vector3<int> R_H = dR2 - dR1;

            const auto row_indexes = pv.get_indexes_row(ibt1);
            const auto col_indexes = pv.get_indexes_col(ibt2);
            const int row_size = static_cast<int>(row_indexes.size());
            const int col_size = static_cast<int>(col_indexes.size());
            if (row_size == 0 || col_size == 0)
            {
                return;
            }

            const hamilt::BaseMatrix<TR>* h_mat = hR.find_matrix(ibt1, ibt2, R_H.x, R_H.y, R_H.z);
            if (h_mat == nullptr)
            {
                return;
            }
            const TR* h_data = h_mat->get_pointer();

            hamilt::BaseMatrix<double>* pa1 = phialpha[0]->find_matrix(iat, ibt1, dR1);
            hamilt::BaseMatrix<double>* pa2 = phialpha[0]->find_matrix(iat, ibt2, dR2);

            const int T0 = ucell.iat2it[iat];
            const int I0 = ucell.iat2ia[iat];

            int ib = 0;
            for (int L0 = 0; L0 <= orb.Alpha[0].getLmax(); ++L0)
            {
                for (int N0 = 0; N0 < orb.Alpha[0].getNchi(L0); ++N0)
                {
                    const int inl = deepks_param.inl_index[T0](I0, L0, N0);
                    const int nm = 2 * L0 + 1;

                    for (int irow = 0; irow < row_size; ++irow)
                    {
                        for (int icol = 0; icol < col_size; ++icol)
                        {
                            const TR h_val = h_data[irow * col_size + icol];
                            if (h_val == TR{})
                            {
                                continue;
                            }
                            for (int m1 = 0; m1 < nm; ++m1)
                            {
                                const double pa1_val = pa1->get_value(row_indexes[irow], ib + m1);
                                for (int m2 = 0; m2 < nm; ++m2)
                                {
                                    dot_ph_acc[inl][m1][m2] += static_cast<TorchScalar>(
                                        pa1_val * h_val * pa2->get_value(col_indexes[icol], ib + m2));
                                }
                            }
                        }
                    }
                    ib += nm;
                }
            }
        });

#ifdef __MPI
    if (std::is_same<TR, double>::value)
    {
        Parallel_Reduce::reduce_all(dot_ph.data_ptr<double>(), dot_ph.numel());
    }
    else
    {
        Parallel_Reduce::reduce_all(reinterpret_cast<double*>(dot_ph.data_ptr<c10::complex<double>>()),
                                    2 * dot_ph.numel());
    }
#endif

    // H(R) is stored internally in Ry, while all DeePKS labels and model
    // energies use Hartree. Keep this projected Hamiltonian in Hartree so
    // B^T H and B^T B are directly compatible. The direct H(R) writer applies
    // the same conversion at its output boundary.
    dot_phialpha_hamilt = 0.5 * dot_ph;

    ModuleBase::timer::end("DeePKS_domain", "cal_phialpha_hamilt_proj");
}

// ---------------------------------------------------------------------------
void DeePKS_domain::cal_gvx_square(const torch::Tensor& gvx, torch::Tensor& gvx_square)
{
    ModuleBase::TITLE("DeePKS_domain", "cal_gvx_square");
    ModuleBase::timer::start("DeePKS_domain", "cal_gvx_square");

    if (gvx.defined())
    {
        const torch::Tensor flat = flatten_operator(gvx.to(torch::kFloat64), 2);
        gvx_square = flat.t().mm(flat);
    }

    ModuleBase::timer::end("DeePKS_domain", "cal_gvx_square");
}

// ---------------------------------------------------------------------------
void DeePKS_domain::cal_gvepsl_square(const torch::Tensor& gvepsl, torch::Tensor& gvepsl_square)
{
    ModuleBase::TITLE("DeePKS_domain", "cal_gvepsl_square");
    ModuleBase::timer::start("DeePKS_domain", "cal_gvepsl_square");

    if (gvepsl.defined())
    {
        const torch::Tensor flat = flatten_operator(gvepsl.to(torch::kFloat64), 1);
        gvepsl_square = flat.t().mm(flat);
    }

    ModuleBase::timer::end("DeePKS_domain", "cal_gvepsl_square");
}

// ---------------------------------------------------------------------------
template <typename TK, typename TR>
DeePKS_domain::HRLabelSample DeePKS_domain::build_hr_label_sample(
    const int nlocal,
    const int nat,
    const std::vector<ModuleBase::Vector3<double>>& kvec_d,
    const UnitCell& ucell,
    const LCAO_Orbitals& orb,
    const Parallel_Orbitals& pv,
    const Grid_Driver& GridD,
    const elecstate::DensityMatrix<TK, double>& dm,
    LCAO_Deepks<TK>& ld,
    const torch::Tensor& dot_phialpha_hamilt_target,
    const torch::Tensor& dot_phialpha_hamilt_base,
    const bool gradient_labels)
{
    ModuleBase::TITLE("DeePKS_domain", "build_hr_label_sample");
    ModuleBase::timer::start("DeePKS_domain", "build_hr_label_sample");

    const DeePKS_Param& deepks_param = ld.deepks_param;
    const std::vector<hamilt::HContainer<double>*>& phialpha = ld.phialpha;

    DeePKS_domain::update_dmr(kvec_d,
                              dm.get_DMK_vector(),
                              ucell,
                              orb,
                              pv,
                              GridD,
                              ld.dm_r);

    bool init_pdm = false;
    // cal_pdm writes into pre-sized tensors; use the initialized DeePKS
    // layout but recompute every entry for this sampled density.
    std::vector<torch::Tensor> pdm = ld.pdm;
    DeePKS_domain::cal_pdm<TK>(init_pdm,
                               deepks_param,
                               kvec_d,
                               ld.dm_r,
                               phialpha,
                               ucell,
                               orb,
                               GridD,
                               pv,
                               pdm);

    std::vector<torch::Tensor> descriptor;
    DeePKS_domain::cal_descriptor(nat, deepks_param, pdm, descriptor);
    torch::Tensor descriptor_out = torch::cat(descriptor).reshape({nat, deepks_param.des_per_atom});

    std::vector<torch::Tensor> gevdm;
    DeePKS_domain::cal_gevdm(nat, deepks_param, pdm, gevdm);

    HRLabelSample state;
    state.descriptor = descriptor_out.detach().contiguous();
    state.gevdm = gevdm;
    if (!gradient_labels)
    {
        ModuleBase::timer::end("DeePKS_domain", "build_hr_label_sample");
        return state;
    }

    const int btb_R_size = 2 * DeePKS_domain::get_R_size(*phialpha[0]);
    torch::Tensor vdrpre_square;
    DeePKS_domain::cal_vdrpre_square(nlocal,
                                     nat,
                                     btb_R_size,
                                     deepks_param,
                                     phialpha,
                                     gevdm,
                                     ucell,
                                     orb,
                                     pv,
                                     GridD,
                                     vdrpre_square);
    torch::Tensor dot_delta = dot_phialpha_hamilt_target - dot_phialpha_hamilt_base;
    if (dot_delta.is_complex())
    {
        dot_delta = torch::real(dot_delta);
    }
    dot_delta = dot_delta.to(torch::kFloat64);

    const int nlmax = deepks_param.inlmax / nat;
    const int mmax = 2 * deepks_param.lmaxd + 1;
    dot_delta = dot_delta.reshape({nat, nlmax, mmax, mmax});
    std::vector<torch::Tensor> projection_parts;
    projection_parts.reserve(nlmax);
    for (int nl = 0; nl < nlmax; ++nl)
    {
        const int nm = gevdm[nl].size(-1);
        const torch::Tensor dot_nl = dot_delta.select(1, nl).slice(1, 0, nm).slice(2, 0, nm);
        projection_parts.push_back(torch::einsum("akmn,amn->ak", {gevdm[nl], dot_nl}));
    }

    state.metric = vdrpre_square.detach().contiguous();
    state.projection = torch::cat(projection_parts, 1).detach().contiguous();

    ModuleBase::timer::end("DeePKS_domain", "build_hr_label_sample");
    return state;
}

template <typename TK>
double DeePKS_domain::project_model_gradient(HRLabelSample& sample, LCAO_Deepks<TK>& ld)
{
    torch::Tensor descriptor = sample.descriptor.detach().clone().unsqueeze(0);
    descriptor.requires_grad_(true);
    const torch::Tensor energy = ld.model_deepks.forward({descriptor}).toTensor().sum();
    const torch::Tensor gradient = torch::autograd::grad({energy}, {descriptor})[0].reshape({-1});
    sample.model_gradient = gradient.detach().contiguous();
    if (sample.metric.defined())
    {
        sample.projection = torch::matmul(sample.metric, gradient)
                                .reshape(sample.descriptor.sizes()).detach().contiguous();
    }
    return energy.item<double>();
}

template double DeePKS_domain::project_model_gradient<double>(DeePKS_domain::HRLabelSample&, LCAO_Deepks<double>&);
template double DeePKS_domain::project_model_gradient<std::complex<double>>(
    DeePKS_domain::HRLabelSample&, LCAO_Deepks<std::complex<double>>&);

template <typename TR>
void DeePKS_domain::build_model_hr(const HRLabelSample& sample,
                                  const std::vector<hamilt::HContainer<double>*>& phialpha,
                                  const UnitCell& ucell,
                                  const LCAO_Orbitals& orb,
                                  const Parallel_Orbitals& pv,
                                  const Grid_Driver& grid,
                                  hamilt::HContainer<TR>& hr)
{
    const auto g = sample.model_gradient.reshape(sample.descriptor.sizes());
    std::vector<torch::Tensor> gdm;
    int offset = 0;
    for (const auto& gev : sample.gevdm)
    {
        const int nm = gev.size(-1);
        gdm.push_back(torch::einsum("av,avmn->amn", {g.slice(1, offset, offset + nm), gev}).contiguous());
        offset += nm;
    }
    DeePKS_domain::iterate_ad2(
        ucell, grid, orb, false,
        [&](const int iat, const ModuleBase::Vector3<double>&,
            const int atom1, const ModuleBase::Vector3<double>&, const int, const int,
            ModuleBase::Vector3<int> r1,
            const int atom2, const ModuleBase::Vector3<double>&, const int, const int,
            ModuleBase::Vector3<int> r2) {
            const auto rows = pv.get_indexes_row(atom1);
            const auto cols = pv.get_indexes_col(atom2);
            const auto* left = phialpha[0]->find_matrix(iat, atom1, r1);
            const auto* right = phialpha[0]->find_matrix(iat, atom2, r2);
            if (rows.empty() || cols.empty() || left == nullptr || right == nullptr)
            {
                return;
            }
            const auto r = r2 - r1;
            auto* matrix = hr.find_matrix(atom1, atom2, r);
            if (matrix == nullptr)
            {
                hamilt::AtomPair<TR> pair(atom1, atom2, r, &pv);
                hr.insert_pair(pair);
                matrix = hr.find_matrix(atom1, atom2, r);
                matrix->allocate(nullptr, true);
            }
            int ib = 0;
            for (const auto& block : gdm)
            {
                const int nm = block.size(-1);
                const double* derivative = block.data_ptr<double>() + iat * nm * nm;
                for (std::size_t i = 0; i < rows.size(); ++i)
                {
                    for (std::size_t j = 0; j < cols.size(); ++j)
                    {
                        double value = 0.0;
                        for (int m = 0; m < nm; ++m)
                        {
                            for (int n = 0; n < nm; ++n)
                            {
                                value += left->get_value(rows[i], ib + m)
                                         * derivative[m * nm + n] * right->get_value(cols[j], ib + n);
                            }
                        }
                        // Model derivatives are Hartree; HContainer is internal Ry.
                        matrix->add_element(i, j, TR(value / LCAO_deepks_io::Ry2Hartree));
                    }
                }
                ib += nm;
            }
        });
}

template void DeePKS_domain::build_model_hr<double>(
    const HRLabelSample&, const std::vector<hamilt::HContainer<double>*>&, const UnitCell&,
    const LCAO_Orbitals&, const Parallel_Orbitals&, const Grid_Driver&, hamilt::HContainer<double>&);
template void DeePKS_domain::build_model_hr<std::complex<double>>(
    const HRLabelSample&, const std::vector<hamilt::HContainer<double>*>&, const UnitCell&,
    const LCAO_Orbitals&, const Parallel_Orbitals&, const Grid_Driver&, hamilt::HContainer<std::complex<double>>&);

// ---------------------------------------------------------------------------
void DeePKS_domain::cal_vdrpre_square(const int nlocal,
                                      const int nat,
                                      const int R_size,
                                      const DeePKS_Param& deepks_param,
                                      const std::vector<hamilt::HContainer<double>*>& phialpha,
                                      const std::vector<torch::Tensor>& gevdm,
                                      const UnitCell& ucell,
                                      const LCAO_Orbitals& orb,
                                      const Parallel_Orbitals& pv,
                                      const Grid_Driver& GridD,
                                      torch::Tensor& vdrpre_square)
{
    ModuleBase::TITLE("DeePKS_domain", "cal_vdrpre_square");
    ModuleBase::timer::start("DeePKS_domain", "cal_vdrpre_square");

    const int N_alpha = nat * deepks_param.des_per_atom;
    using SparseRows = std::unordered_map<std::int64_t, std::vector<double>>;
    SparseRows rows;

    // Build only nonzero B_(R,mu,nu) rows.  The old implementation first
    // allocated the full R^3*nlocal^2*N_alpha tensor, which can require
    // several GiB per MPI rank even though almost all rows are zero.
    DeePKS_domain::iterate_ad2(
        ucell,
        GridD,
        orb,
        false,
        [&](const int iat,
            const ModuleBase::Vector3<double>&,
            const int ibt1,
            const ModuleBase::Vector3<double>&,
            const int start1,
            const int nw1_tot,
            ModuleBase::Vector3<int> dR1,
            const int ibt2,
            const ModuleBase::Vector3<double>&,
            const int start2,
            const int nw2_tot,
            ModuleBase::Vector3<int> dR2) {
            auto* overlap_1 = phialpha[0]->find_matrix(iat, ibt1, dR1);
            auto* overlap_2 = phialpha[0]->find_matrix(iat, ibt2, dR2);
            if (overlap_1 == nullptr || overlap_2 == nullptr)
            {
                return;
            }

            const ModuleBase::Vector3<int> dR = dR2 - dR1;
            const int iRx = DeePKS_domain::mapping_R(dR.x);
            const int iRy = DeePKS_domain::mapping_R(dR.y);
            const int iRz = DeePKS_domain::mapping_R(dR.z);
            if (iRx >= R_size || iRy >= R_size || iRz >= R_size)
            {
                return;
            }

            std::vector<int> row_iw;
            std::vector<int> row_iw_all;
            std::vector<int> col_iw;
            std::vector<int> col_iw_all;
            for (int iw1 = 0; iw1 < nw1_tot; ++iw1)
            {
                if (pv.global2local_row(start1 + iw1) >= 0)
                {
                    row_iw.push_back(iw1);
                    row_iw_all.push_back(start1 + iw1);
                }
            }
            for (int iw2 = 0; iw2 < nw2_tot; ++iw2)
            {
                if (pv.global2local_col(start2 + iw2) >= 0)
                {
                    col_iw.push_back(iw2);
                    col_iw_all.push_back(start2 + iw2);
                }
            }

            int ib = 0;
            int nl = 0;
            for (int L0 = 0; L0 <= orb.Alpha[0].getLmax(); ++L0)
            {
                for (int N0 = 0; N0 < orb.Alpha[0].getNchi(L0); ++N0, ++nl)
                {
                    const int nm = 2 * L0 + 1;
                    const int nm2 = nm * nm;
                    const double* gdata = gevdm[nl].data_ptr<double>() + iat * nm * nm2;

                    for (std::size_t k1 = 0; k1 < row_iw.size(); ++k1)
                    {
                        for (std::size_t k2 = 0; k2 < col_iw.size(); ++k2)
                        {
                            const std::int64_t key
                                = (((static_cast<std::int64_t>(iRx) * R_size + iRy) * R_size + iRz) * nlocal
                                       + row_iw_all[k1])
                                      * nlocal
                                  + col_iw_all[k2];
                            auto inserted = rows.try_emplace(key, N_alpha, 0.0);
                            std::vector<double>& brow = inserted.first->second;

                            for (int v = 0; v < nm; ++v)
                            {
                                double value = 0.0;
                                const double* g_row = gdata + v * nm2;
                                for (int m1 = 0; m1 < nm; ++m1)
                                {
                                    const double left = overlap_1->get_value(row_iw[k1], ib + m1);
                                    for (int m2 = 0; m2 < nm; ++m2)
                                    {
                                        value += left * overlap_2->get_value(col_iw[k2], ib + m2)
                                                 * g_row[m1 * nm + m2];
                                    }
                                }
                                brow[iat * deepks_param.des_per_atom + ib + v] += value;
                            }
                        }
                    }
                    ib += nm;
                }
            }
        });

    torch::Tensor M = torch::empty({static_cast<long>(rows.size()), N_alpha},
                                   torch::TensorOptions().dtype(torch::kFloat64));
    double* M_data = M.data_ptr<double>();
    std::size_t row_index = 0;
    for (const auto& entry : rows)
    {
        std::copy(entry.second.begin(), entry.second.end(), M_data + row_index * N_alpha);
        ++row_index;
    }

    vdrpre_square = M.t().mm(M);
#ifdef __MPI
    Parallel_Reduce::reduce_all(vdrpre_square.data_ptr<double>(), vdrpre_square.numel());
#endif

    ModuleBase::timer::end("DeePKS_domain", "cal_vdrpre_square");
}

template void DeePKS_domain::cal_phialpha_hamilt_proj<double>(const int,
                                                              const int,
                                                              const DeePKS_Param&,
                                                              const std::vector<hamilt::HContainer<double>*>&,
                                                              const hamilt::HContainer<double>&,
                                                              const UnitCell&,
                                                              const LCAO_Orbitals&,
                                                              const Parallel_Orbitals&,
                                                              const Grid_Driver&,
                                                              torch::Tensor&);

template void DeePKS_domain::cal_phialpha_hamilt_proj<std::complex<double>>(
    const int,
    const int,
    const DeePKS_Param&,
    const std::vector<hamilt::HContainer<double>*>&,
    const hamilt::HContainer<std::complex<double>>&,
    const UnitCell&,
    const LCAO_Orbitals&,
    const Parallel_Orbitals&,
    const Grid_Driver&,
    torch::Tensor&);

template DeePKS_domain::HRLabelSample
DeePKS_domain::build_hr_label_sample<double, double>(
    const int,
    const int,
    const std::vector<ModuleBase::Vector3<double>>&,
    const UnitCell&,
    const LCAO_Orbitals&,
    const Parallel_Orbitals&,
    const Grid_Driver&,
    const elecstate::DensityMatrix<double, double>&,
    LCAO_Deepks<double>&,
    const torch::Tensor&,
    const torch::Tensor&,
    const bool);

template DeePKS_domain::HRLabelSample
DeePKS_domain::build_hr_label_sample<std::complex<double>, double>(
    const int,
    const int,
    const std::vector<ModuleBase::Vector3<double>>&,
    const UnitCell&,
    const LCAO_Orbitals&,
    const Parallel_Orbitals&,
    const Grid_Driver&,
    const elecstate::DensityMatrix<std::complex<double>, double>&,
    LCAO_Deepks<std::complex<double>>&,
    const torch::Tensor&,
    const torch::Tensor&,
    const bool);

template DeePKS_domain::HRLabelSample
DeePKS_domain::build_hr_label_sample<std::complex<double>, std::complex<double>>(
    const int,
    const int,
    const std::vector<ModuleBase::Vector3<double>>&,
    const UnitCell&,
    const LCAO_Orbitals&,
    const Parallel_Orbitals&,
    const Grid_Driver&,
    const elecstate::DensityMatrix<std::complex<double>, double>&,
    LCAO_Deepks<std::complex<double>>&,
    const torch::Tensor&,
    const torch::Tensor&,
    const bool);

#endif
