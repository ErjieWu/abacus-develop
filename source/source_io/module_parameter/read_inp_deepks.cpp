#include "source_base/constants.h"
#include "source_base/tool_quit.h"
#include "source_io/module_parameter/parameter.h"
#include "read_input.h"
#include "read_input_tool.h"

#include <algorithm>
#include <cctype>

namespace ModuleIO
{
namespace
{
std::string functional_name(std::string name)
{
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return name;
}
}

void ReadInput::item_deepks()
{
    // NOTE: The order of add_item() calls below determines the parameter order
    // in the generated documentation (docs/advanced/input_files/input-main.md).
    // Please preserve this ordering when adding new parameters.
    {
        Input_Item item("deepks_out_labels");
        item.annotation = ">0 compute descriptor for deepks. 1 used during training, 2 used for label production";
        item.category = "DeePKS";
        item.type = "Integer";
        item.description = R"(Print labels and descriptors for DeePKS in OUT.${suffix}. The names of these files start with "deepks".
* 0 : No output.
* 1 : Output intermediate files needed during DeePKS training.
* 2 : Output target labels for label preperation. The label files are named as deepks_<property>.npy or deepks_<property>.csr, where the units and formats are the same as label files <property>.npy or <property>.csr required for training, except that the first dimension (nframes) is excluded. System structrue files are also given in deepks_atom.npy and deepks_box.npy in the unit of Bohr, which means lattice_constant should be set to 1 when training.

[NOTE] When deepks_out_labels equals 1, the path of a numerical descriptor (an orb file) is needed to be specified under the NUMERICAL_DESCRIPTOR tag in the STRU file. This is not needed when deepks_out_labels equals 2.)";
        item.default_value = "0";
        item.unit = "";
        item.set_availability("basis_type==lcao");
        read_sync_int(input.deepks_out_labels);
        this->add_item(item);
    }
    {
        Input_Item item("deepks_out_pair");
        item.annotation = "output same-density paired-functional labels in direct or gradient form";
        item.category = "DeePKS";
        item.type = "Boolean";
        item.description = R"(Output paired fixed-density energy and real-space Hamiltonian-gradient samples. deepks_pair_base and deepks_pair_target define one ordered functional pair; dft_functional must equal one of them and determines which functional drives the active SCF. Ordinary active functionals preserve the first, interval-selected, and final electronic states according to deepks_out_pair_freq. For an active functional containing exact exchange, ABACUS applies the interval to the initial semilocal inner loop and preserves every later density-converged outer checkpoint. Independent label-only EXX evaluators construct any required hybrid operator at each selected density without changing the active SCF. The descriptor d[rho], energy E_target[rho]-E_base[rho], raw metric M=A_base[rho]^T A_base[rho], and raw projection b=A_base[rho]^T(H_target[rho]-H_base[rho]) are evaluated at the same density matrix. Outputs use the deepks_sample_*.npy prefix in OUT.${suffix}.

[NOTE] This mode requires deepks_out_labels=1, nspin=1 and eigenvalue descriptors. deepks_grad selects gradient samples (raw metric and projection); when false, deepks_v_delta selects direct Hamiltonian differences and precalculation operators. Direct files use deepks_sample_hr_sN.csr or deepks_sample_hk_sN.npy plus the selected *_pre or factorized intermediates, where N is the zero-based saved-sample index. Energy, descriptor, step and kind arrays are shared by both representations. The Hamiltonian and energy are target-minus-base quantities at the same density, in Hartree. Paired band-gap labels are not implemented.

With deepks_scf=1 and both pair names omitted, compare the underlying dft_functional with that functional plus the loaded model: the sampled energy is E_model[d] and the Hamiltonian projection is M*dE_model/dd. With explicit pair names, the functionals must differ; dft_functional must be deepks_pair_base when a model is active, and the model correction is excluded from both named reference operators.)";
        item.default_value = "False";
        item.unit = "";
        item.set_availability("basis_type==lcao");
        read_sync_bool(input.deepks_out_pair);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (!para.input.deepks_out_pair)
            {
                return;
            }
            if (para.input.deepks_out_labels != 1)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_out_pair requires deepks_out_labels 1");
            }
            if (para.input.deepks_bandgap != 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "paired band-gap labels are not implemented");
            }
            if (para.input.deepks_equiv)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_out_pair requires eigenvalue descriptors");
            }
            if (para.input.nspin != 1)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_out_pair currently supports nspin 1 only");
            }
            const std::string base = functional_name(para.input.deepks_pair_base);
            const std::string target = functional_name(para.input.deepks_pair_target);
            const std::string active = functional_name(para.input.dft_functional);
            const bool has_base = base != "none";
            const bool has_target = target != "none";
            if (!has_base && !has_target && para.input.deepks_scf)
            {
                return; // Default pair: underlying functional versus its model-corrected result.
            }
            if (!has_base || !has_target)
            {
                ModuleBase::WARNING_QUIT("ReadInput",
                                         "deepks_out_pair requires deepks_pair_base and deepks_pair_target");
            }
            if (base == target)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_pair_base and deepks_pair_target must differ");
            }
            const bool active_is_base = active == base;
            const bool active_is_target = active == target;
            if (!active_is_base && !active_is_target)
            {
                ModuleBase::WARNING_QUIT(
                    "ReadInput", "dft_functional must equal deepks_pair_base or deepks_pair_target");
            }
            if (para.input.deepks_scf && !active_is_base)
            {
                ModuleBase::WARNING_QUIT(
                    "ReadInput",
                    "deepks_scf paired output requires dft_functional=deepks_pair_base");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("deepks_out_pair_freq");
        item.annotation = "electronic-step interval for same-density paired labels";
        item.category = "DeePKS";
        item.type = "Integer";
        item.description = "Save the first ordinary electronic state, every deepks_out_pair_freq states, and the final/converged state. Exact-exchange outer checkpoints remain preserved because they are already sparse.";
        item.default_value = "1";
        item.unit = "";
        item.set_availability("basis_type==lcao and deepks_out_pair==true");
        read_sync_int(input.deepks_out_pair_freq);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.deepks_out_pair_freq < 1)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_out_pair_freq must be at least 1");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("deepks_pair_base");
        item.annotation = "base functional for paired fixed-density labels";
        item.category = "DeePKS";
        item.type = "String";
        item.description = "Base functional B in E_target[rho]-E_base[rho] and A[rho]^T(H_target[rho]-H_base[rho]). With deepks_scf=1, omit both pair names to compare the underlying dft_functional with its DeePKS-corrected result. Otherwise specify both names.";
        item.default_value = "None";
        item.unit = "";
        item.set_availability("basis_type==lcao and deepks_out_pair==true");
        read_sync_string(input.deepks_pair_base);
        this->add_item(item);
    }
    {
        Input_Item item("deepks_pair_target");
        item.annotation = "target functional for paired fixed-density labels";
        item.category = "DeePKS";
        item.type = "String";
        item.description = "Target functional T in E_target[rho]-E_base[rho] and A[rho]^T(H_target[rho]-H_base[rho]). With deepks_scf=1 and both pair names omitted, the target is the underlying dft_functional plus the loaded DeePKS model. An explicit named pair excludes the active model from both reference operators.";
        item.default_value = "None";
        item.unit = "";
        item.set_availability("basis_type==lcao and deepks_out_pair==true");
        read_sync_string(input.deepks_pair_target);
        this->add_item(item);
    }
    {
        Input_Item item("deepks_scf");
        item.annotation = ">0 add V_delta to Hamiltonian";
        item.category = "DeePKS";
        item.type = "Boolean";
        item.description = "perform self-consistent field iteration in DeePKS method"
                          "\n\n[NOTE] A trained, traced model file is needed.";
        item.default_value = "False";
        item.unit = "";
        item.set_availability("basis_type==lcao");
        read_sync_bool(input.deepks_scf);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
#ifndef __MLALGO
            if (para.input.deepks_scf || para.input.deepks_out_labels || para.input.deepks_bandgap
                || para.input.deepks_v_delta)
            {
                ModuleBase::WARNING_QUIT("Input_conv", "please compile with DeePKS");
            }
#endif
            // if (!para.input.deepks_scf && para.input.deepks_out_labels == 1)
            // {
            //     ModuleBase::WARNING_QUIT("Input_conv", "deepks_out_labels = 1 requires deepks_scf = 1");
            // }
        };
        this->add_item(item);
    }
    {
        Input_Item item("deepks_equiv");
        item.annotation = "whether to use equivariant version of DeePKS";
        item.category = "DeePKS";
        item.type = "Boolean";
        item.description = "whether to use equivariant version of DeePKS"
                          "\n\n[NOTE] The equivariant version of DeePKS-kit is still under development, "
                          "so this feature is currently only intended for internal usage.";
        item.default_value = "False";
        item.unit = "";
        item.set_availability("basis_type==lcao");
        read_sync_bool(input.deepks_equiv);
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (para.input.deepks_equiv && para.input.deepks_bandgap)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "equivariant version of DeePKS is not implemented yet");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("deepks_model");
        item.annotation = "file dir of traced pytorch model: 'model.ptg";
        item.category = "DeePKS";
        item.type = "String";
        item.description = "the path of the trained, traced neural network model file generated by deepks-kit";
        item.default_value = "None";
        item.unit = "";
        item.set_availability("basis_type==lcao and deepks_scf==true");
        read_sync_string(input.deepks_model);
        this->add_item(item);
    }
    {
        Input_Item item("bessel_descriptor_lmax");
        item.annotation = "lmax used in generating spherical bessel functions";
        item.category = "DeePKS";
        item.type = "Integer";
        item.description = "the maximum angular momentum of the Bessel functions generated as the projectors in DeePKS - NOte: To generate such projectors, set calculation type to gen_bessel in ABACUS. See also calculation.";
        item.default_value = "2";
        item.unit = "";
        item.set_availability("calculation==gen_bessel");
        read_sync_int(input.bessel_descriptor_lmax);
        this->add_item(item);
    }
    {
        Input_Item item("bessel_descriptor_ecut");
        item.annotation = "energy cutoff for spherical bessel functions(Ry)";
        item.category = "DeePKS";
        item.type = "String";
        item.description = "energy cutoff of Bessel functions";
        item.default_value = "same as ecutwfc";
        item.unit = "Ry";
        item.set_availability("calculation==gen_bessel");
        read_sync_string(input.bessel_descriptor_ecut);
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (para.input.bessel_descriptor_ecut == "default")
            {
                para.input.bessel_descriptor_ecut = std::to_string(para.input.ecutwfc);
            }
        };
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (std::stod(para.input.bessel_descriptor_ecut) < 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "bessel_descriptor_ecut must >= 0");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("bessel_descriptor_tolerence");
        item.annotation = "tolerence for spherical bessel root";
        item.category = "DeePKS";
        item.type = "Real";
        item.description = "tolerance for searching the zeros of Bessel functions";
        item.default_value = "1.0e-12";
        item.unit = "";
        item.set_availability("calculation==gen_bessel");
        read_sync_double(input.bessel_descriptor_tolerence);
        this->add_item(item);
    }
    {
        Input_Item item("bessel_descriptor_rcut");
        item.annotation = "radial cutoff for spherical bessel functions(a.u.)";
        item.category = "DeePKS";
        item.type = "Real";
        item.description = "cutoff radius of Bessel functions";
        item.default_value = "6.0";
        item.unit = "Bohr";
        item.set_availability("calculation==gen_bessel");
        read_sync_double(input.bessel_descriptor_rcut);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.bessel_descriptor_rcut < 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "bessel_descriptor_rcut must >= 0");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("bessel_descriptor_smooth");
        item.annotation = "spherical bessel smooth or not";
        item.category = "DeePKS";
        item.type = "Boolean";
        item.description = "smooth the Bessel functions at radius cutoff";
        item.default_value = "False";
        item.unit = "";
        item.set_availability("calculation==gen_bessel");
        read_sync_bool(input.bessel_descriptor_smooth);
        this->add_item(item);
    }
    {
        Input_Item item("bessel_descriptor_sigma");
        item.annotation = "sphereical bessel smearing_sigma";
        item.category = "DeePKS";
        item.type = "Real";
        item.description = "smooth parameter at the cutoff radius of projectors";
        item.default_value = "0.1";
        item.unit = "Bohr";
        item.set_availability("calculation==gen_bessel");
        read_sync_double(input.bessel_descriptor_sigma);
        this->add_item(item);
    }
    {
        Input_Item item("deepks_bandgap");
        item.annotation = ">0 for bandgap label";
        item.category = "DeePKS";
        item.type = "Integer";
        item.description = R"(include bandgap label for DeePKS training
* 0: Don't include bandgap label
* 1: Include target bandgap label (see deepks_band_range for more details)
* 2: Include multiple bandgap label (see deepks_band_range for more details)
* 3: Used for systems containing H atoms. Here HOMO is defined as the max occupation except H atoms and the bandgap label is the energy between HOMO and (HOMO + 1))";
        item.default_value = "0";
        item.unit = "";
        item.set_availability("basis_type==lcao and deepks_scf==true");
        read_sync_int(input.deepks_bandgap);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.deepks_bandgap < 0 || para.input.deepks_bandgap > 3)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_bandgap must be integer in [0, 3]");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("deepks_band_range");
        item.annotation = "(int, int) range of bands for bandgap label";
        item.category = "DeePKS";
        item.type = "Integer*2";
        item.description = R"(The first value should not be larger than the second one and the meaning differs in different cases below
* deepks_bandgap is 1: Bandgap label is the energy between LUMO + deepks_band_range[0] and LUMO + deepks_band_range[1]. If not set, it will calculate energy between HOMO and LUMO states.
* deepks_bandgap is 2: Bandgap labels are energies between HOMO and all states in range [LUMO + deepks_band_range[0], LUMO + deepks_band_range[1]] (Thus there are deepks_band_range[1] - deepks_band_range[0] + 1 bandgaps in total). If HOMO is included in the setting range, it will be ignored since it will always be zero and has no valuable messages (deepks_band_range[1] - deepks_band_range[0] bandgaps in this case). NOTICE: The set range can be greater than, less than, or include the value of HOMO. In the bandgap label, we always calculate the energy of the state in the set range minus the energy of HOMO state, so the bandgap can be negative if the state is lower than HOMO.)";
        item.default_value = "-1 0";
        item.unit = "";
        item.set_availability("basis_type==lcao and deepks_scf==true and deepks_bandgap in [1, 2]");
        item.read_value = [](const Input_Item& item, Parameter& para) {
            para.input.deepks_band_range[0] = std::stod(item.str_values[0]);
            para.input.deepks_band_range[1] = std::stod(item.str_values[1]);
        };
        sync_intvec(input.deepks_band_range, 2, 0);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.deepks_bandgap == 1)
            {
                if (para.input.deepks_band_range[0] >= para.input.deepks_band_range[1])
                {
                    ModuleBase::WARNING_QUIT("ReadInput", "deepks_band_range[0] must be smaller than deepks_band_range[1] for deepks_bandgap = 1.");
                }
            }
            else if (para.input.deepks_bandgap == 2)
            {
                if (para.input.deepks_band_range[0] > para.input.deepks_band_range[1])
                {
                    ModuleBase::WARNING_QUIT("ReadInput", "deepks_band_range[0] must be no more than deepks_band_range[1] for deepks_bandgap = 2.");
                }
            }
            else
            {
                if (para.input.deepks_band_range[0] != -1 || para.input.deepks_band_range[1] != 0)
                {
                    ModuleBase::WARNING("ReadInput", "deepks_band_range is used for deepks_bandgap = 1/2. Ignore its setting for other cases.");
                }
            } 
        };
        this->add_item(item);
    }
    {
        Input_Item item("deepks_v_delta");
        item.annotation = "!=0 for v_delta/v_delta_R label. when output, 1 for vdpre, 2 for phialpha and grad_evdm (can save memory )"
                          " -1 for vdrpre, -2 for phialpha_r and gevdm (can save memory)";
        item.category = "DeePKS";
        item.type = "Integer";
        item.description = R"(Include V_delta/V_delta_R (Hamiltonian in k/real space) label for DeePKS training. When deepks_out_labels is true and deepks_v_delta > 0 (k space), ABACUS will output deepks_hbase.npy, deepks_vdelta.npy and deepks_htot.npy(htot=hbase+vdelta). When deepks_out_labels is true and deepks_v_delta < 0 (real space), ABACUS will output deepks_hrtot.csr, deepks_hrdelta.csr. All DeePKS energy and Hamiltonian labels are output in Hartree.
* deepks_v_delta = 1: deepks_vdpre.npy, which is used to calculate V_delta during DeePKS training.
* deepks_v_delta = 2: deepks_phialpha.npy and deepks_gevdm.npy, which can be used to calculate deepks_vdpre.npy. A recommanded method for memory saving.
* deepks_v_delta = -1: deepks_vdrpre.npy, which is used to calculate V_delta_R during DeePKS training.
* deepks_v_delta = -2: deepks_phialpha_r.npy and deepks_gevdm.npy, which can be used to calculate deepks_vdrpre.npy. A recommanded method for memory saving.

This parameter selects direct labels only. Leave it at 0 when deepks_grad is true; real-space Hamiltonian gradient intermediates are then selected automatically.)";
        item.default_value = "0";
        item.unit = "";
        item.set_availability("basis_type==lcao");
        read_sync_int(input.deepks_v_delta);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.deepks_v_delta < -2 || para.input.deepks_v_delta > 2)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_v_delta must be integer in [-2, 2]");
            }
            if (para.input.deepks_grad && para.input.deepks_v_delta != 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput",
                                        "deepks_v_delta selects direct labels only; use 0 with deepks_grad");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("deepks_grad");
        item.annotation = "output descriptor-gradient label intermediates for DeePKS training";
        item.category = "DeePKS";
        item.type = "Boolean";
        item.description = "Select gradient intermediates instead of direct property labels. Real-space Hamiltonian intermediates are always included: deepks_gevdm.npy, deepks_vdrpre_square.npy and deepks_dot_phialpha_hamilt.npy. Force and virial operators and Gram matrices are additionally generated when cal_force and cal_stress are enabled, including model-free runs (deepks_scf=0). Energy remains a scalar label. The projected Hamiltonian uses Hartree, consistently with direct DeePKS labels. Requires deepks_out_labels=1, nspin=1 and a numerical descriptor. Leave deepks_v_delta=0; band-gap gradient labels are not supported.";
        item.default_value = "False";
        item.unit = "";
        item.set_availability("basis_type==lcao and deepks_out_labels==1");
        read_sync_bool(input.deepks_grad);
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.deepks_grad && para.input.nspin != 1)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_grad currently supports nspin 1 only");
            }
            if (para.input.deepks_grad && para.input.deepks_out_labels != 1)
            {
                ModuleBase::WARNING_QUIT("ReadInput",
                    "deepks_grad requires deepks_out_labels 1");
            }
            if (para.input.deepks_grad && para.input.deepks_bandgap != 0)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_bandgap is not supported with deepks_grad");
            }
            if (para.input.deepks_grad && para.input.deepks_equiv)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "deepks_grad requires eigenvalue descriptors, not deepks_equiv");
            }
        };
        this->add_item(item);
    }
    {
        Input_Item item("deepks_out_unittest");
        item.annotation = "if set 1, prints intermediate quantities that shall "
                          "be used for making unit test";
        item.category = "DeePKS";
        item.type = "Boolean";
        item.description = "generate files for constructing DeePKS unit test"
                          "\n\n[NOTE] Not relevant when running actual calculations. "
                          "When set to 1, ABACUS needs to be run with only 1 process.";
        item.default_value = "False";
        item.unit = "";
        read_sync_bool(input.deepks_out_unittest);
        item.reset_value = [](const Input_Item& item, Parameter& para) {
            if (para.input.deepks_out_unittest)
            {
                para.input.deepks_out_labels = 1;
                para.input.deepks_scf = true;
            }
        };
        item.check_value = [](const Input_Item& item, const Parameter& para) {
            if (para.input.deepks_out_unittest)
            {
                if (para.input.cal_force != 1 || para.input.cal_stress != 1)
                {
                    ModuleBase::WARNING_QUIT("ReadInput",
                                             "force and stress are required in generating deepks unittest");
                }
            }
        };
        this->add_item(item);
    }
}
} // namespace ModuleIO
