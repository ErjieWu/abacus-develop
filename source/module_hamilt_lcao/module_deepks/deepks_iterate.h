#ifndef DEEPKS_ITER_H
#define DEEPKS_ITER_H

#ifdef __DEEPKS

#include "module_base/complexmatrix.h"
#include "module_base/intarray.h"
#include "module_base/matrix.h"
#include "module_base/timer.h"
#include "module_base/vector3.h"
#include "module_basis/module_ao/ORB_read.h"
#include "module_basis/module_ao/parallel_orbitals.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_cell/unitcell.h"
#include "module_parameter/parameter.h"

#include <functional>

namespace DeePKS_domain
{
//------------------------
// deepks_iterate.cpp
//------------------------

// This file contains

void iterate_ad1(const UnitCell& ucell,
                 const Grid_Driver& GridD,
                 const LCAO_Orbitals& orb,
                 const bool with_trace,
                 std::function<void(const int /*iat*/,
                                    const ModuleBase::Vector3<double>& /*tau0*/,
                                    const int /*ibt*/,
                                    const ModuleBase::Vector3<double>& /*tau1*/,
                                    const int /*start*/,
                                    const int /*nw1_tot*/,
                                    ModuleBase::Vector3<int> /*dR*/)> callback);

void iterate_ad2(const UnitCell& ucell,
                 const Grid_Driver& GridD,
                 const LCAO_Orbitals& orb,
                 const bool with_trace,
                 std::function<void(const int /*iat*/,
                                    const ModuleBase::Vector3<double>& /*tau0*/,
                                    const int /*ibt1*/,
                                    const ModuleBase::Vector3<double>& /*tau1*/,
                                    const int /*start1*/,
                                    const int /*nw1_tot*/,
                                    ModuleBase::Vector3<int> /*dR1*/,
                                    const int /*ibt2*/,
                                    const ModuleBase::Vector3<double>& /*tau2*/,
                                    const int /*start2*/,
                                    const int /*nw2_tot*/,
                                    ModuleBase::Vector3<int> /*dR2*/)> callback);
} // namespace DeePKS_domain

#endif
#endif
