#include <memory>
#include <array>
#include <set>
#include "module_parameter/parameter.h"
#include "symmetry.h"
#include "module_parameter/parameter.h"
#include "module_base/libm/libm.h"
#include "module_base/mathzone.h"
#include "module_base/constants.h"
#include "module_base/timer.h"
#include "module_io/output.h"
namespace ModuleSymmetry
{
int Symmetry::symm_flag = 0;
bool Symmetry::symm_autoclose = false;
bool Symmetry::pricell_loop = true;

void Symmetry::analy_sys(const Lattice& lat, const Statistics& st, Atom* atoms, std::ofstream& ofs_running)
{
    const double MAX_EPS = std::max(1e-3, epsilon_input * 1.001);
    const double MULT_EPS = 2.0;

    ModuleBase::TITLE("Symmetry","init");
	ModuleBase::timer::tick("Symmetry","analy_sys");

	ofs_running << "\n\n\n\n";
	ofs_running << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
	ofs_running << " |                                                                    |" << std::endl;
	ofs_running << " | Doing symmetry analysis:                                           |" << std::endl;
	ofs_running << " | We calculate the norm of 3 vectors and the angles between them,    |" << std::endl;
	ofs_running << " | the type of Bravais lattice is given. We can judge if the unticell |" << std::endl;
	ofs_running << " | is a primitive cell. Finally we give the point group operation for |" << std::endl;
	ofs_running << " | this unitcell. We use the point group operations to do symmetry |" << std::endl;
	ofs_running << " | analysis on given k-point mesh and the charge density.             |" << std::endl;
	ofs_running << " |                                                                    |" << std::endl;
	ofs_running << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
	ofs_running << "\n\n\n\n";

    // --------------------------------
    // 1. copy data and allocate memory
    // --------------------------------
    // number of total atoms
    this->nat = st.nat;
    // number of atom species
    this->ntype = st.ntype;
    this->na = new int[ntype];
    this->istart = new int[ntype];  // start number of atom.
    this->index = new int [nat + 2];   // index of atoms
    ModuleBase::GlobalFunc::ZEROS(na, ntype);
    ModuleBase::GlobalFunc::ZEROS(istart, ntype);
    ModuleBase::GlobalFunc::ZEROS(index, nat+2);

    // atom positions
    // used in checksym.
	newpos = new double[3*nat]; // positions of atoms before rotation
    rotpos = new double[3*nat]; // positions of atoms after rotation
	ModuleBase::GlobalFunc::ZEROS(newpos, 3*nat);
    ModuleBase::GlobalFunc::ZEROS(rotpos, 3*nat);

    this->a1 = lat.a1;
    this->a2 = lat.a2;
    this->a3 = lat.a3;

	ModuleBase::Matrix3 latvec1;
	latvec1.e11 = a1.x; latvec1.e12 = a1.y; latvec1.e13 = a1.z;
	latvec1.e21 = a2.x; latvec1.e22 = a2.y; latvec1.e23 = a2.z;
	latvec1.e31 = a3.x; latvec1.e32 = a3.y; latvec1.e33 = a3.z;

	output::printM3(ofs_running,"LATTICE VECTORS: (CARTESIAN COORDINATE: IN UNIT OF A0)",latvec1);

    istart[0] = 0;
    this->itmin_type = 0;
    this->itmin_start = 0;
    for (int it = 0; it < ntype; ++it)
    {
        Atom* atom = &atoms[it];
        this->na[it] = atom->na;
        if (it > 0) {
            istart[it] = istart[it-1] + na[it-1];
        }
        //std::cout << "\n istart = " << istart[it];
        if (na[it] < na[itmin_type])
        {
            this->itmin_type = it;
            this->itmin_start = istart[it];
        }
    }
    //s: input config
    s1 = a1;
    s2 = a2;
    s3 = a3;


    auto lattice_to_group = [&, this](int& nrot_out, int& nrotk_out, std::ofstream& ofs_running) -> void {
        // a: the optimized lattice vectors, output
        // s: the input lattice vectors, input
        // find the real_brav type accordiing to lattice vectors.
        this->lattice_type(this->a1, this->a2, this->a3, this->s1, this->s2, this->s3,
            this->cel_const, this->pre_const, this->real_brav, ilattname, atoms, true, this->newpos);

        ofs_running << "(for optimal symmetric configuration:)" << std::endl;
        ModuleBase::GlobalFunc::OUT(ofs_running, "BRAVAIS TYPE", real_brav);
        ModuleBase::GlobalFunc::OUT(ofs_running, "BRAVAIS LATTICE NAME", ilattname);
        ModuleBase::GlobalFunc::OUT(ofs_running, "ibrav", real_brav);
        Symm_Other::print1(real_brav, cel_const, ofs_running);

        optlat.e11 = a1.x; optlat.e12 = a1.y; optlat.e13 = a1.z;
        optlat.e21 = a2.x; optlat.e22 = a2.y; optlat.e23 = a2.z;
        optlat.e31 = a3.x; optlat.e32 = a3.y; optlat.e33 = a3.z;

        // count the number of primitive cells in the supercell
        this->pricell(this->newpos, atoms);

        test_brav = true; // output the real ibrav and point group
        
        // list all possible point group operations 
        this->setgroup(this->symop, this->nop, this->real_brav);

        // special case for AFM analysis
        // which should be loop over all atoms, f.e only loop over spin-up atoms
        // --------------------------------
        // AFM analysis Start
        if (PARAM.inp.nspin > 1) {
            pricell_loop = this->magmom_same_check(atoms);
        }

        if (!pricell_loop && PARAM.inp.nspin == 2)
        {
            this->analyze_magnetic_group(atoms, st, nrot_out, nrotk_out);
        }
        else
        {
            // get the real symmetry operations according to the input structure
            // nrot_out: the number of pure point group rotations
            // nrotk_out: the number of all space group operations
            this->getgroup(nrot_out, nrotk_out, ofs_running, this->nop, this->symop, this->gmatrix, this->gtrans,
                this->newpos, this->rotpos, this->index, this->ntype, this->itmin_type, this->itmin_start, this->istart, this->na);
        }
        };

    // --------------------------------
    // 2. analyze the symmetry
    // --------------------------------
    // 2.1 skip the symmetry analysis if the symmetry has been analyzed
    if (PARAM.inp.calculation == "cell-relax" && nrotk > 0)
    {
        std::ofstream no_out;   // to screen the output when trying new epsilon

        // For the cases where cell-relax cause the number of symmetry operations to increase
        if (this->nrotk > this->max_nrotk) {
            this->max_nrotk = this->nrotk;
        }

        int tmp_nrot, tmp_nrotk;
        lattice_to_group(tmp_nrot, tmp_nrotk, ofs_running);  // get the real symmetry operations

        // Actually, the analysis of symmetry has been done now
        // Following implementation is find the best epsilon to keep the symmetry
        // some different method to enlarge symmetry_prec
        bool eps_enlarged = false;
        auto eps_mult = [this](double mult) {epsilon *= mult;};
        auto eps_to = [this](double new_eps) {epsilon = new_eps;};

        // store the symmetry_prec and nrotk for each try
        std::vector<double> precs_try;
        std::vector<int> nrotks_try;
        // store the initial result
        precs_try.push_back(epsilon);
        nrotks_try.push_back(tmp_nrotk);
        //enlarge epsilon and regenerate pointgroup
        // Try to find the symmetry operations by increasing epsilon
        while (tmp_nrotk < this->max_nrotk && epsilon < MAX_EPS)
        {
            eps_mult(MULT_EPS);
            eps_enlarged = true;
            // lattice_to_group(tmp_nrot, tmp_nrotk, no_out);
            lattice_to_group(tmp_nrot, tmp_nrotk, no_out);
            precs_try.push_back(epsilon);
            nrotks_try.push_back(tmp_nrotk);
        }
        if (tmp_nrotk > this->nrotk)
        {
            this->nrotk = tmp_nrotk;
            ofs_running << "Find new symmtry operations during cell-relax." << std::endl;
            if (this->nrotk > this->max_nrotk) {
                this->max_nrotk = this->nrotk;
            }
        }
        if (eps_enlarged)
        {
            if (epsilon > MAX_EPS)
            {
                ofs_running << "WARNING: Symmetry cannot be kept due to the lost of accuracy with atom position during cell-relax." << std::endl;
                ofs_running << "Continue cell-relax with a lower symmetry. " << std::endl;
                // find the smallest epsilon that gives the current number of symmetry operations
                int valid_index = nrotks_try.size() - 1;
                while (valid_index > 0
                       && tmp_nrotk <= nrotks_try[valid_index - 1]) {
                    --valid_index;
                }
                eps_to(precs_try[valid_index]);
                if (valid_index > 0) {
                    ofs_running << " Enlarging `symmetry_prec` to " << epsilon
                                << " ..." << std::endl;
                } else {
                    eps_enlarged = false;
                }
                // regenerate pointgroup after change epsilon (may not give the same result)
                lattice_to_group(tmp_nrot, tmp_nrotk, ofs_running);
                this->nrotk = tmp_nrotk;
            } else {
                ofs_running << " Enlarging `symmetry_prec` to " << epsilon
                            << " ..." << std::endl;
            }
        }
        if (!eps_enlarged && epsilon > epsilon_input * 1.001)   // not "else" here. "eps_enlarged" can be set to false in the above "if"
        {   // try a smaller symmetry_prec until the number of symmetry operations decreases
            precs_try.erase(precs_try.begin() + 1, precs_try.end());
            nrotks_try.erase(nrotks_try.begin() + 1, nrotks_try.end());
            double eps_current = epsilon; // record the current symmetry_prec
            do {
                eps_mult(1 / MULT_EPS);
                lattice_to_group(tmp_nrot, tmp_nrotk, no_out);
                precs_try.push_back(epsilon);
                nrotks_try.push_back(tmp_nrotk);
            } while (tmp_nrotk >= nrotks_try[0] && epsilon > epsilon_input * 1.001 && precs_try.size() < 5);
            int valid_index = (tmp_nrotk < nrotks_try[0]) ? nrotks_try.size() - 2 : nrotks_try.size() - 1;
#ifdef __DEBUG
            assert(valid_index >= 0);
            assert(nrotks_try[valid_index] >= nrotks_try[0]);
#endif
            epsilon = precs_try[valid_index];
            // regenerate pointgroup after change epsilon
            lattice_to_group(tmp_nrot, tmp_nrotk, ofs_running);
            this->nrotk = tmp_nrotk;
            if (valid_index > 0) { // epsilon is set smaller
                ofs_running << " Narrowing `symmetry_prec` from " << eps_current
                            << " to " << epsilon << " ..." << std::endl;
            }
        }
    } else {
        lattice_to_group(this->nrot, this->nrotk, ofs_running);
    }
    // Symmetry analysis End!
    //-------------------------------------------

    // final number of symmetry operations
#ifdef __DEBUG
    ofs_running << "symmetry_prec(epsilon) in current ion step: " << this->epsilon << std::endl;
    ofs_running << "number of symmetry operations in current ion step: " << this->nrotk << std::endl;
#endif
    //----------------------------------
    // 3. output to running.log
    //----------------------------------
    // output the point group
    bool valid_group = this->pointgroup(this->nrot, this->pgnumber, this->pgname, this->gmatrix, ofs_running);
	ModuleBase::GlobalFunc::OUT(ofs_running,"POINT GROUP", this->pgname);
    // output the space group
    valid_group = this->pointgroup(this->nrotk, this->spgnumber, this->spgname, this->gmatrix, ofs_running);
    ModuleBase::GlobalFunc::OUT(ofs_running, "POINT GROUP IN SPACE GROUP", this->spgname);

    //-----------------------------
    // 4. For the case where point group is not complete due to symmetry_prec
    //-----------------------------
    if (!valid_group)
    {   // select the operations that have the inverse
        std::vector<int>invmap(this->nrotk, -1);
        this->gmatrix_invmap(this->gmatrix, this->nrotk, invmap.data());
        int nrotk_new = 0;
        for (int isym = 0;isym < this->nrotk;++isym)
        {
            if (invmap[isym] != -1)
            {
                if(nrotk_new < isym)
                {
                    this->gmatrix[nrotk_new] = this->gmatrix[isym];
                    this->gtrans[nrotk_new] = this->gtrans[isym];
                }
                ++nrotk_new;
            }
        }
        this->nrotk = nrotk_new;
    }

    // convert gmatrix to reciprocal space
    this->gmatrix_convert_int(gmatrix, kgmatrix, nrotk, optlat, lat.G);
    
    // convert the symmetry operations from the basis of optimal symmetric configuration 
    // to the basis of input configuration
    this->gmatrix_convert_int(gmatrix, gmatrix, nrotk, optlat, latvec1);
    this->gtrans_convert(gtrans, gtrans, nrotk, optlat, latvec1);

    this->set_atom_map(atoms); // find the atom mapping according to the symmetry operations

    // Do this here for debug
    if (PARAM.inp.calculation == "relax")
    {
        this->all_mbl = this->is_all_movable(atoms, st);
        if (!this->all_mbl)
        {
            std::cout << "WARNING: Symmetry cannot be kept when not all atoms are movable.\n ";
            std::cout << "Continue with symmetry=0 ... \n";
            ModuleSymmetry::Symmetry::symm_flag = 0;
        }
    }

    delete[] newpos;
    delete[] na;
    delete[] rotpos;
    delete[] index;
    delete[] istart;
    ModuleBase::timer::tick("Symmetry","analy_sys");
    return;
}

//---------------------------------------------------
// The lattice will be transformed to a 'standard
// cystallographic setting', the relation between
// 'origin' and 'transformed' lattice vectors will
// be givin in matrix form
//---------------------------------------------------
int Symmetry::standard_lat(
    ModuleBase::Vector3<double> &a,
    ModuleBase::Vector3<double> &b,
    ModuleBase::Vector3<double> &c,
    double *cel_const) const
{
    static bool first = true;
    // there are only 14 types of Bravais lattice.
    int type = 15;
	//----------------------------------------------------
    // used to calculte the volume to judge whether 
	// the lattice vectors corrispond the right-hand-sense
	//----------------------------------------------------
    double volume = 0;
    //the lattice vectors have not been changed

    const double aa = a * a;
    const double bb = b * b;
    const double cc = c * c;
    const double ab = a * b; //std::vector: a * b * cos(alpha)
    const double bc = b * c; //std::vector: b * c * cos(beta)
    const double ca = c * a; //std::vector: c * a * cos(gamma)
    double norm_a = a.norm();
    double norm_b = b.norm();
    double norm_c = c.norm();
    double gamma = ab /( norm_a * norm_b ); // cos(gamma)
    double alpha  = bc /( norm_b * norm_c ); // cos(alpha)
    double beta = ca /( norm_a * norm_c ); // cos(beta)
    double amb = sqrt( aa + bb - 2 * ab );	//amb = |a - b|
    double bmc = sqrt( bb + cc - 2 * bc );
    double cma = sqrt( cc + aa - 2 * ca );
    double apb = sqrt( aa + bb + 2 * ab );  //amb = |a + b|
    double bpc = sqrt( bb + cc + 2 * bc );
    double cpa = sqrt( cc + aa + 2 * ca );
    double apbmc = sqrt( aa + bb + cc + 2 * ab - 2 * bc - 2 * ca );	//apbmc = |a + b - c|
    double bpcma = sqrt( bb + cc + aa + 2 * bc - 2 * ca - 2 * ab );
    double cpamb = sqrt( cc + aa + bb + 2 * ca - 2 * ab - 2 * bc );
    double abc = ab + bc + ca;

	if (first)
    {
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"NORM_A",norm_a);
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"NORM_B",norm_b);
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"NORM_C",norm_c);
//		OUT(GlobalV::ofs_running,"alpha  = ", alpha );
//		OUT(GlobalV::ofs_running,"beta   = " ,beta  );
//		OUT(GlobalV::ofs_running,"gamma  = " ,gamma );
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"ALPHA (DEGREE)", acos(alpha)/ModuleBase::PI*180.0 );
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"BETA  (DEGREE)" ,acos(beta)/ModuleBase::PI*180.0  );
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"GAMMA (DEGREE)" ,acos(gamma)/ModuleBase::PI*180.0 );
        first = false;
    }

	
//	std::cout << " a=" << norm_a << std::endl; 
//	std::cout << " b=" << norm_b << std::endl; 
//	std::cout << " c=" << norm_c << std::endl; 
//	std::cout << " alpha=" << alpha << std::endl;
//	std::cout << " beta=" << beta << std::endl;
//	std::cout << " gamma=" << gamma << std::endl;
     

    Symm_Other::right_hand_sense(a, b, c);
	ModuleBase::GlobalFunc::ZEROS(cel_const, 6);
	const double small = PARAM.inp.symmetry_prec;

	//---------------------------	
	// 1. alpha == beta == gamma 
	//---------------------------	
	if( equal(alpha, gamma) && equal(alpha, beta) )
	{
		//--------------
		// a == b == c 
		//--------------
		if( equal(norm_a, norm_b) && equal(norm_b, norm_c))
		{
			//---------------------------------------
			// alpha == beta == gamma == 90 degree
			//---------------------------------------
			if ( equal(alpha,0.0) )
			{
				type=1;
				cel_const[0]=norm_a;
			}
			//----------------------------------------
			// cos(alpha) = -1.0/3.0
			//----------------------------------------
			else if( equal(alpha, -1.0/3.0) ) 
			{
				type=2;
				cel_const[0]=norm_a*2.0/sqrt(3.0);
			}
			//----------------------------------------
			// cos(alpha) = 0.5
			//----------------------------------------
			else if( equal(alpha, 0.5) ) 
			{
				type=3;
				cel_const[0]=norm_a*sqrt(2.0);
			}
			//----------------------------------------
			// cos(alpha) = all the others
			//----------------------------------------
			else
			{
				type=7;
				cel_const[0]=norm_a;
				cel_const[3]=alpha;
			}
		}
		// Crystal classes with inequal length of lattice vectors but also with
		// A1*A2=A1*A3=A2*A3:
		// Orthogonal axes:
		else if(equal(gamma,0.0)) 
		{
			// Two axes with equal lengths means simple tetragonal: (IBRAV=5)
			// Adjustment: 'c-axis' shall be the special axis.
			if (equal(norm_a, norm_b)) 
			{
				type=5;
				cel_const[0]=norm_a;
				cel_const[2]=norm_c/norm_a;
				// No axes with equal lengths means simple orthorhombic (IBRAV=8):
				// Adjustment: Sort the axis by increasing lengths:
			}
            else if(((norm_c-norm_b)>small) && ((norm_b-norm_a)>small) ) 
			{
				type=8;
				cel_const[0]=norm_a;
				cel_const[1]=norm_b/norm_a;
				cel_const[2]=norm_c/norm_a;
			}
			// Crystal classes with A1*A3=A2*A3=/A1*A2:
		}
	}//end alpha=beta=gamma
	//-----------------------
	// TWO EQUAL ANGLES
	// alpha == beta != gamma  (gamma is special)
	//------------------------
	else if (equal(alpha-beta, 0)) 
	{
		//---------------------------------------------------------
		// alpha = beta = 90 degree
		// One axis orthogonal with respect to the other two axes:
		//---------------------------------------------------------
		if (equal(alpha, 0.0)) 
		{
			//-----------------------------------------------
			// a == b 
			// Equal length of the two nonorthogonal axes:
			//-----------------------------------------------
			if (equal(norm_a, norm_b)) 
			{
				// Cosine(alpha) equal to -1/2 means hexagonal: (IBRAV=4)
				// Adjustment: 'c-axis' shall be the special axis.
				if ( equal(gamma, -0.5))   //gamma = 120 degree
				{
					type=4;
					cel_const[0]=norm_a;
					cel_const[2]=norm_c/norm_a;
					// Other angles mean base-centered orthorhombic: (IBRAV=11)
					// Adjustment: Cosine between A1 and A2 shall be lower than zero, the
					//             'c-axis' shall be the special axis.
				}
				else if(gamma<(-1.0*small)) //gamma > 90 degree
				{
					type=11;
                    cel_const[0]=apb;
                    cel_const[1]=amb/apb;
                    cel_const[2]=norm_c/apb;
                    cel_const[5]=gamma;
				}
				// Different length of the two axes means simple monoclinic (IBRAV=12):
				// Adjustment: Cosine(gamma) should be lower than zero, special axis
				//             shall be the 'b-axis'(!!!) and |A1|<|A3|:
			}
			//----------
			// a!=b!=c
			//----------
            else if( gamma<(-1.0*small) && (norm_a-norm_b)>small) 
			{
				type=12;
				cel_const[0]=norm_b;
				cel_const[1]=norm_c/norm_b;
				cel_const[2]=norm_a/norm_b;
                cel_const[4]=gamma;
                //adjust: a->c, b->a, c->b
                ModuleBase::Vector3<double> tmp=c;
				c=a;
				a=b;
				b=tmp;
			}
		}//end gamma<small
		// Arbitrary angles between the axes:
		// |A1|=|A2|=|A3| means body-centered tetragonal (IBRAV=6):
		// Further additional criterions are: (A1+A2), (A1+A3) and (A2+A3) are
		// orthogonal to one another and (adjustment//): |A1+A3|=|A2+A3|/=|A1+A2|
		else
		{
			if( equal(norm_a, norm_b) && 
				equal(norm_b, norm_c) &&
				equal(cpa, bpc) && 
				!equal(apb, cpa) &&
				equal(norm_c*norm_c+abc,0) )
			{
				type=6;
				cel_const[0]=cpa;
				cel_const[2]=apb/cpa;
			}
			// |A1|=|A2|=/|A3| means base-centered monoclinic (IBRAV=13):
			// Adjustement: The cosine between A1 and A3 as well as the cosine
			//              between A2 and A3 should be lower than zero.
			else if( equal(norm_a,norm_b) 
					&& alpha<(-1.0*small) 
					&& beta<(-1.0*small)) 
			{
				type=13;
				cel_const[0]=apb;
				cel_const[1]=amb/apb;
				cel_const[2]=norm_c/apb;
                //cos(<a+b, c>)
                cel_const[4]=(a+b)*c/apb/norm_c;
			}
		}
	} //end alpha==beta
	//-------------------------------
	// three angles are not equal
	//-------------------------------
	else 
	{
		// Crystal classes with A1*A2=/A1*A3=/A2*A3
		// |A1|=|A2|=|A3| means body-centered orthorhombic (IBRAV=9):
		// Further additional criterions are: (A1+A2), (A1+A3) and (A2+A3) are
		// orthogonal to one another and (adjustment//): |A1+A2|>|A1+A3|>|A2+A3|
		if (equal(norm_a, norm_b) &&
				equal(norm_b, norm_c) &&
				((cpa-bpc)>small) &&
				((apb-cpa)>small) && 
				equal(norm_c*norm_c+abc, 0)) 
		{
			type=9;
			cel_const[0]=bpc;
			cel_const[1]=cpa/bpc;
			cel_const[2]=apb/bpc;
		}
		// |A1|=|A2-A3| and |A2|=|A1-A3| and |A3|=|A1-A2| means face-centered
		// orthorhombic (IBRAV=10):
		// Adjustment: |A1+A2-A3|>|A1+A3-A2|>|A2+A3-A1|
		else if(equal(amb, norm_c) &&
				equal(cma, norm_b) &&
				equal(bmc, norm_a) && 
				((apbmc-cpamb)>small) &&
				((cpamb-bpcma)>small)) 
		{
			type=10;
			cel_const[0]=bpcma;
			cel_const[1]=cpamb/bpcma;
			cel_const[2]=apbmc/bpcma;
		}
		// Now there exists only one further possibility - triclinic (IBRAV=14):
		// Adjustment: All three cosines shall be greater than zero and ordered:
		else if((gamma>beta) && (beta>alpha) && (alpha>small)) 
		{
			type=14;
			cel_const[0]=norm_a;
			cel_const[1]=norm_b/norm_a;
			cel_const[2]=norm_c/norm_a;
			cel_const[3]=alpha;
			cel_const[4]=beta;
			cel_const[5]=gamma;
		}
	}
	
	return type;
}

//---------------------------------------------------
// The lattice will be transformed to a 'standard
// cystallographic setting', the relation between
// 'origin' and 'transformed' lattice vectors will
// be givin in matrix form
// must be called before symmetry analysis
// only need to called once for each ion step
//---------------------------------------------------
void Symmetry::lattice_type(
    ModuleBase::Vector3<double> &v1,
    ModuleBase::Vector3<double> &v2,
    ModuleBase::Vector3<double> &v3,
    ModuleBase::Vector3<double> &v01,
    ModuleBase::Vector3<double> &v02,
    ModuleBase::Vector3<double> &v03,
    double *cel_const,
    double *pre_const,
    int& real_brav,
    std::string& bravname,
    const Atom* atoms,
    bool convert_atoms,
    double* newpos)const
{
    ModuleBase::TITLE("Symmetry","lattice_type");
//      std::cout << "v1 = " << v1.x << " " << v1.y << " " << v1.z <<std::endl;
//      std::cout << "v2 = " << v2.x << " " << v2.y << " " << v2.z <<std::endl;
//      std::cout << "v3 = " << v3.x << " " << v3.y << " " << v3.z <<std::endl;

	//----------------------------------------------
	// (1) adjustement of the basis to right hand 
	// sense by inversion of all three lattice 
	// vectors if necessary
	//----------------------------------------------
    const bool right = Symm_Other::right_hand_sense(v1, v2, v3);
  //    std::cout << "v1 = " << v1.x << " " << v1.y << " " << v1.z <<std::endl;
  //    std::cout << "v2 = " << v2.x << " " << v2.y << " " << v2.z <<std::endl;
  //    std::cout << "v3 = " << v3.x << " " << v3.y << " " << v3.z <<std::endl;

	ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"right hand lattice",right);

	//-------------------------------------------------
	// (2) save and copy the original lattice vectors.
	//-------------------------------------------------
    v01 = v1;
    v02 = v2;
    v03 = v3;
	
	//--------------------------------------------
	// (3) calculate the 'pre_const'
	//--------------------------------------------
	ModuleBase::GlobalFunc::ZEROS(pre_const, 6);
//    std::cout << "ATTION !!!!!!" <<std::endl;
//        std::cout << "v1 = " << v1.x << " " << v1.y << " " << v1.z <<std::endl;
//        std::cout << "v2 = " << v2.x << " " << v2.y << " " << v2.z <<std::endl;
//        std::cout << "v3 = " << v3.x << " " << v3.y << " " << v3.z <<std::endl;

    int pre_brav = standard_lat(v1, v2, v3, cel_const);
//    for ( int i = 0; i < 6; ++i)
//    {
//        std::cout << "cel = "<<cel_const[i]<<" ";
//    }
//    std::cout << std::endl;

//  std::cout << "pre_brav = " << pre_brav <<std::endl;

    for ( int i = 0; i < 6; ++i)
    {
        pre_const[i] = cel_const[i];
    }
//        std::cout << "v1 = " << v1.x << " " << v1.y << " " << v1.z <<std::endl;
//        std::cout << "v2 = " << v2.x << " " << v2.y << " " << v2.z <<std::endl;
//        std::cout << "v3 = " << v3.x << " " << v3.y << " " << v3.z <<std::endl;


// find the shortest basis vectors of the lattice
    this->get_shortest_latvec(v1, v2, v3);
//        std::cout << "a1 = " << v1.x << " " << v1.y << " " << v1.z <<std::endl;
//        std::cout << "a1 = " << v2.x << " " << v2.y << " " << v2.z <<std::endl;
//        std::cout << "a1 = " << v3.x << " " << v3.y << " " << v3.z <<std::endl;

    Symm_Other::right_hand_sense(v1, v2, v3);
//        std::cout << "a1 = " << v1.x << " " << v1.y << " " << v1.z <<std::endl;
//        std::cout << "a1 = " << v2.x << " " << v2.y << " " << v2.z <<std::endl;
//        std::cout << "a1 = " << v3.x << " " << v3.y << " " << v3.z <<std::endl;

    real_brav = 15;
    double temp_const[6];

    //then we should find the best lattice vectors to make much easier the determination of the lattice symmetry
    //the method is to contrast the combination of the shortest vectors and determine their symmmetry

    ModuleBase::Vector3<double> w1, w2, w3;
    ModuleBase::Vector3<double> q1, q2, q3;
    this->get_optlat(v1, v2, v3, w1, w2, w3, real_brav, cel_const, temp_const);
//        std::cout << "a1 = " << v1.x << " " << v1.y << " " << v1.z <<std::endl;
//        std::cout << "a1 = " << v2.x << " " << v2.y << " " << v2.z <<std::endl;
//        std::cout << "a1 = " << v3.x << " " << v3.y << " " << v3.z <<std::endl;

    //now, the highest symmetry of the combination of the shortest vectors has been found
    //then we compare it with the original symmetry
	
//	GlobalV::ofs_running << " w1" << std::endl;
//	GlobalV::ofs_running << " " << std::setw(15) << w1.x << std::setw(15) << w1.y << std::setw(15) << w1.z << std::endl;
//	GlobalV::ofs_running << " " << std::setw(15) << w2.x << std::setw(15) << w2.y << std::setw(15) << w2.z << std::endl;
//	GlobalV::ofs_running << " " << std::setw(15) << w3.x << std::setw(15) << w3.y << std::setw(15) << w3.z << std::endl;
//	GlobalV::ofs_running << " pre_brav=" << pre_brav << std::endl;
//	GlobalV::ofs_running << " temp_brav=" << temp_brav << std::endl;

    bool change_flag=false;
    for (int i = 0; i < 6; ++i) {
        if(!equal(cel_const[i], pre_const[i])) 
            {change_flag=true; break;
        }
    }

    if ( real_brav < pre_brav || change_flag )
    {
        //if the symmetry of the new vectors is higher, store the new ones
        for (int i = 0; i < 6; ++i)
        {
            cel_const[i] = temp_const[i];
        }
        q1 = w1;
        q2 = w2;
        q3 = w3;
        if(convert_atoms)
        {
            GlobalV::ofs_running <<std::endl;
            GlobalV::ofs_running <<" The lattice vectors have been changed (STRU_SIMPLE.cif)"<<std::endl;
            GlobalV::ofs_running <<std::endl;
            int at=0;
            for (int it = 0; it < this->ntype; ++it)
            {
                for (int ia = 0; ia < this->na[it]; ++ia)
                    {
                    ModuleBase::Mathzone::Cartesian_to_Direct(atoms[it].tau[ia].x,
                        atoms[it].tau[ia].y,
                        atoms[it].tau[ia].z,
                                            q1.x, q1.y, q1.z,
                                            q2.x, q2.y, q2.z,
                                            q3.x, q3.y, q3.z,
                                            newpos[3*at],newpos[3*at+1],newpos[3*at+2]);

    //                        std::cout << " newpos_before = " << newpos[3*iat] << " " << newpos[3*iat+1] << " " << newpos[3*iat+2] << std::endl;
    //                      GlobalV::ofs_running << " newpos_before = " << newpos[3*iat] << " " << newpos[3*iat+1] << " " << newpos[3*iat+2] << std::endl;
                            for(int k=0; k<3; ++k)
                            {
                                    this->check_translation( newpos[at*3+k], -floor(newpos[at*3+k]));
                                    this->check_boundary( newpos[at*3+k] );
                            }
    //                        std::cout << " newpos = " << newpos[3*iat] << " " << newpos[3*iat+1] << " " << newpos[3*iat+2] << std::endl;

    //                      GlobalV::ofs_running << " newpos = " << newpos[3*iat] << " " << newpos[3*iat+1] << " " << newpos[3*iat+2] << std::endl;
                            ++at;
                    }
            }       
            /*
            std::stringstream ss;
            ss << PARAM.globalv.global_out_dir << "STRU_SIMPLE.cif";

            std::ofstream ofs( ss.str().c_str() );
            ofs << "Lattice vector  : " << std::endl;
            ofs << q1.x <<"   "<<q1.y<<"  "<<q1.z<< std::endl;
            ofs << q2.x <<"   "<<q2.y<<"  "<<q2.z<< std::endl;
            ofs << q3.x <<"   "<<q3.y<<"  "<<q3.z<< std::endl;
            ofs << std::endl;
            ofs << "Direct positions : " << " " << std::endl;
            ofs << std::endl;
            at =0;
            
            for (int it = 0; it < this->ntype; it++)
            {
                for (int ia = 0; ia < this->na[it]; ia++)
                {
                    ofs << atoms[it].label
                    << " " << newpos[3*at]
                    << " " << newpos[3*at+1]
                    << " " << newpos[3*at+2] << std::endl;
                    at++;
                }
            }
            ofs.close();
            */
        }
        // return the optimized lattice in v1, v2, v3
        v1=q1;
        v2=q2;
        v3=q3;
    }
    else
    {
        //else, store the original ones
        for (int i = 0; i < 6; ++i)
        {
            cel_const[i] = pre_const[i];
        }
        //newpos also need to be set
        if(convert_atoms)
        {
            int at=0;
            for (int it = 0; it < this->ntype; ++it)
            {
                for (int ia = 0; ia < this->na[it]; ++ia)
                {
                    ModuleBase::Mathzone::Cartesian_to_Direct(atoms[it].tau[ia].x,
                        atoms[it].tau[ia].y,
                        atoms[it].tau[ia].z,
                                    v1.x, v1.y, v1.z,
                                    v2.x, v2.y, v2.z,
                                    v3.x, v3.y, v3.z,
                                    newpos[3*at],newpos[3*at+1],newpos[3*at+2]);
                    for(int k=0; k<3; ++k)
                    {
                            this->check_translation( newpos[at*3+k], -floor(newpos[at*3+k]));
                            this->check_boundary( newpos[at*3+k] );
                    }
                    ++at;
                }
            }       
        }
    }

    /*
    bool flag3;
    if (pre_brav == temp_brav) 
	{
        flag3 = 0;
        if (!equal(temp_const[0], pre_const[0]) ||
            !equal(temp_const[1], pre_const[1]) ||
            !equal(temp_const[2], pre_const[2]) ||
            !equal(temp_const[3], pre_const[3]) ||
            !equal(temp_const[4], pre_const[4]) ||
            !equal(temp_const[5], pre_const[5])
           )
        {
            flag3 = 1;
        }
        if (flag3==0) {
            //particularly, if the symmetry of origin and new are exactly the same, we choose the original ones
            //Hey! the original vectors have been changed!!!
            v1 = s1;
            v2 = s2;
            v3 = s3;
        	change=0;
			GlobalV::ofs_running<<" The lattice vectors have been set back!"<<std::endl;
        }
    }*/
    bravname = get_brav_name(real_brav);
    return;
}


void Symmetry::getgroup(int& nrot, int& nrotk, std::ofstream& ofs_running, const int& nop,
    const ModuleBase::Matrix3* symop, ModuleBase::Matrix3* gmatrix, ModuleBase::Vector3<double>* gtrans,
    double* pos, double* rotpos, int* index, const int ntype, const int itmin_type, const int itmin_start, int* istart, int* na)const
{
    ModuleBase::TITLE("Symmetry", "getgroup");

	//--------------------------------------------------------------------------------
    //return all possible space group operators that reproduce a lattice with basis
    //out of a (maximum) pool of point group operations that is compatible with
    //the symmetry of the pure translation lattice without any basic.
	//--------------------------------------------------------------------------------

    ModuleBase::Matrix3 zero(0,0,0,0,0,0,0,0,0);
    ModuleBase::Matrix3 help[48];
    ModuleBase::Vector3<double> temp[48];

    nrot = 0;
    nrotk = 0;

	//-------------------------------------------------------------------------
    //pass through the pool of (possibly allowed) symmetry operations and
    //check each operation whether it can reproduce the lattice with basis
	//-------------------------------------------------------------------------
    //std::cout << "nop = " <<nop <<std::endl;
    for (int i = 0; i < nop; ++i)
    {
        bool s_flag = this->checksym(symop[i], gtrans[i], pos, rotpos, index, ntype, itmin_type, itmin_start, istart, na);
        if (s_flag == 1)
        {
			//------------------------------
            // this is a symmetry operation
			// with no translation vectors
            // so ,this is pure point group 
			// operations
			//------------------------------
            if ( equal(gtrans[i].x,0.0) &&
                 equal(gtrans[i].y,0.0) &&
                 equal(gtrans[i].z,0.0))
            {
                ++nrot;
                gmatrix[nrot - 1] = symop[i];
                gtrans[nrot - 1].x = 0;
                gtrans[nrot - 1].y = 0;
                gtrans[nrot - 1].z = 0;
            }
			//------------------------------
            // this is a symmetry operation
			// with translation vectors
            // so ,this is space group 
			// operations
			//------------------------------
            else
            {
                ++nrotk;
                help[nrotk - 1] = symop[i];
                temp[nrotk - 1].x = gtrans[i].x;
                temp[nrotk - 1].y = gtrans[i].y;
                temp[nrotk - 1].z = gtrans[i].z;
            }
        }
    }

	//-----------------------------------------------------
    //If there are operations with nontrivial translations
    //then store them together in the momory
	//-----------------------------------------------------
    if (nrotk > 0)
    {
        for (int i = 0; i < nrotk; ++i)
        {
            gmatrix[nrot + i] = help[i];
            gtrans[nrot + i].x = temp[i].x;
            gtrans[nrot + i].y = temp[i].y;
            gtrans[nrot + i].z = temp[i].z;
        }
    }

	//-----------------------------------------------------
    //total number of space group operations
	//-----------------------------------------------------
    nrotk += nrot;

    if(test_brav)
    {
	    ModuleBase::GlobalFunc::OUT(ofs_running,"PURE POINT GROUP OPERATIONS",nrot);
        ModuleBase::GlobalFunc::OUT(ofs_running,"SPACE GROUP OPERATIONS",nrotk);
    }

	//-----------------------------------------------------
    //fill the rest of matrices and vectors with zeros
	//-----------------------------------------------------
    if (nrotk < 48)
    {
        for (int i = nrotk; i < 48; ++i)
        {
            gmatrix[i] = zero;
            gtrans[i].x = 0;
            gtrans[i].y = 0;
            gtrans[i].z = 0;
        }
    }
    return;
}

bool Symmetry::checksym(const ModuleBase::Matrix3& s, ModuleBase::Vector3<double>& gtrans,
    double* pos, double* rotpos, int* index, const int ntype, const int itmin_type, const int itmin_start, int* istart, int* na)const
{
	//----------------------------------------------
    // checks whether a point group symmetry element 
	// is a valid symmetry operation on a supercell
	//----------------------------------------------
    // the start atom index.
    bool no_diff = false;
    ModuleBase::Vector3<double> trans(2.0, 2.0, 2.0);
    bool s_flag = false;

    for (int it = 0; it < ntype; it++)
    {
		//------------------------------------
        // impose periodic boundary condition
		// 0.5 -> -0.5
		//------------------------------------
        for (int j = istart[it]; j < istart[it] + na[it]; ++j)
        {
            this->check_boundary(pos[j*3+0]);
            this->check_boundary(pos[j*3+1]);
            this->check_boundary(pos[j*3+2]);
        }
        //order original atomic positions for current species
        this->atom_ordering_new(pos + istart[it] * 3, na[it], index + istart[it]);

        //Rotate atoms of current species
        for (int j = istart[it]; j < istart[it] + na[it]; ++j)
        {
            const int xx=j*3;
            const int yy=j*3+1;
            const int zz=j*3+2;


            rotpos[xx] = pos[xx] * s.e11
                         + pos[yy] * s.e21
                         + pos[zz] * s.e31;

            rotpos[yy] = pos[xx] * s.e12
                         + pos[yy] * s.e22
                         + pos[zz] * s.e32;

            rotpos[zz] = pos[xx] * s.e13
                         + pos[yy] * s.e23
                         + pos[zz] * s.e33;

           // std::cout << "pos = " << pos[xx] <<" "<<pos[yy] << " "<<pos[zz]<<std::endl;
           // std::cout << "rotpos = " << rotpos[xx] <<" "<<rotpos[yy] << " "<<rotpos[zz]<<std::endl;
            rotpos[xx] = fmod(rotpos[xx] + 100.5,1) - 0.5;
            rotpos[yy] = fmod(rotpos[yy] + 100.5,1) - 0.5;
            rotpos[zz] = fmod(rotpos[zz] + 100.5,1) - 0.5;
            this->check_boundary(rotpos[xx]);
            this->check_boundary(rotpos[yy]);
            this->check_boundary(rotpos[zz]);
        }
        //order rotated atomic positions for current species
        this->atom_ordering_new(rotpos + istart[it] * 3, na[it], index + istart[it]);
    }

    ModuleBase::Vector3<double> diff;

	//---------------------------------------------------------
    // itmin_start = the start atom positions of species itmin
	//---------------------------------------------------------
    // (s)tart (p)osition of atom (t)ype which has (min)inal number.
    ModuleBase::Vector3<double> sptmin(rotpos[itmin_start * 3], rotpos[itmin_start * 3 + 1], rotpos[itmin_start * 3 + 2]);

    for (int i = itmin_start; i < itmin_start + na[itmin_type]; ++i)
    {
        //set up the current test std::vector "gtrans"
        //and "gtrans" could possibly contain trivial translations:
        gtrans.x = this->get_translation_vector( sptmin.x, pos[i*3+0]);
        gtrans.y = this->get_translation_vector( sptmin.y, pos[i*3+1]);
        gtrans.z = this->get_translation_vector( sptmin.z, pos[i*3+2]);

        //If we had already detected some translation,
        //we must only look at the vectors with coordinates smaller than those
        //of the previously detected std::vector (find the smallest)
        if (gtrans.x > trans.x + epsilon ||
                gtrans.y > trans.y + epsilon ||
                gtrans.z > trans.z + epsilon
           )
        {
            continue;
        }

        //translate all the atomic coordinates BACK by "gtrans"
        for (int it = 0; it < ntype; it++)
        {
            for (int ia = istart[it]; ia < na[it] + istart[it]; ia++)
            {
                this->check_translation( rotpos[ia*3+0], gtrans.x );
                this->check_translation( rotpos[ia*3+1], gtrans.y );
                this->check_translation( rotpos[ia*3+2], gtrans.z );

                this->check_boundary( rotpos[ia*3+0] );
                this->check_boundary( rotpos[ia*3+1] );
                this->check_boundary( rotpos[ia*3+2] );
            }
            //order translated atomic positions for current species
            this->atom_ordering_new(rotpos + istart[it] * 3, na[it], index + istart[it]);
        }

        no_diff = true;
        //compare the two lattices 'one-by-one' whether they are identical
        for (int it = 0; it < ntype; it++)
        {
            for (int ia = istart[it]; ia < na[it] + istart[it]; ia++)
            {
                //take the difference of the rotated and the original coordinates
                diff.x = this->check_diff( pos[ia*3+0], rotpos[ia*3+0]);
                diff.y = this->check_diff( pos[ia*3+1], rotpos[ia*3+1]);
                diff.z = this->check_diff( pos[ia*3+2], rotpos[ia*3+2]);
                //only if all "diff" are zero vectors, flag will remain "1"
                if (	no_diff == false||
                        !equal(diff.x,0.0)||
                        !equal(diff.y,0.0)||
                        !equal(diff.z,0.0)
                   )
                {
                    no_diff = false;
                }
            }
        }
			

		//BLOCK_HERE("check symm");

        //the current test is successful
        if (no_diff == true)
        {
            s_flag = true;
            //save the detected translation std::vector temporarily
            trans.x = gtrans.x;
            trans.y = gtrans.y;
            trans.z = gtrans.z;
        }

        //restore the original rotated coordinates by subtracting "gtrans"
        for (int it = 0; it < ntype; it++)
        {
            for (int ia = istart[it]; ia < na[it] + istart[it]; ia++)
            {
                rotpos[ia*3+0] -= gtrans.x;
                rotpos[ia*3+1] -= gtrans.y;
                rotpos[ia*3+2] -= gtrans.z;
            }
        }
    }

    if (s_flag == 1)
    {
        gtrans.x = trans.x;
        gtrans.y = trans.y;
        gtrans.z = trans.z;
    }
    return s_flag;
}

void Symmetry::pricell(double* pos, const Atom* atoms)
{
    bool no_diff = false;
    ptrans.clear();

    for (int it = 0; it < ntype; it++)
    {
		//------------------------------------
        // impose periodic boundary condition
		// 0.5 -> -0.5
		//------------------------------------
        for (int j = istart[it]; j < istart[it] + na[it]; ++j)
        {
            this->check_boundary(pos[j*3+0]);
            this->check_boundary(pos[j*3+1]);
            this->check_boundary(pos[j*3+2]);
        }

        //order original atomic positions for current species
        this->atom_ordering_new(pos + istart[it] * 3, na[it], index + istart[it]);
        //copy pos to rotpos
        for (int j = istart[it]; j < istart[it] + na[it]; ++j)
        {
            const int xx=j*3;
            const int yy=j*3+1;
            const int zz=j*3+2;
            rotpos[xx] = pos[xx];
            rotpos[yy] = pos[yy];
            rotpos[zz] = pos[zz];
        }
    }

    ModuleBase::Vector3<double> diff;
    double tmp_ptrans[3];

	//---------------------------------------------------------
    // itmin_start = the start atom positions of species itmin
    //---------------------------------------------------------
    // (s)tart (p)osition of atom (t)ype which has (min)inal number.
    ModuleBase::Vector3<double> sptmin(pos[itmin_start * 3], pos[itmin_start * 3 + 1], pos[itmin_start * 3 + 2]);

    for (int i = itmin_start; i < itmin_start + na[itmin_type]; ++i)
    {
        //set up the current test std::vector "gtrans"
        //and "gtrans" could possibly contain trivial translations:
        tmp_ptrans[0] = this->get_translation_vector( pos[i*3+0], sptmin.x);
        tmp_ptrans[1] = this->get_translation_vector( pos[i*3+1], sptmin.y);
        tmp_ptrans[2] = this->get_translation_vector( pos[i*3+2], sptmin.z);
        //translate all the atomic coordinates by "gtrans"
        for (int it = 0; it < ntype; it++)
        {
            for (int ia = istart[it]; ia < na[it] + istart[it]; ia++)
            {
                this->check_translation( rotpos[ia*3+0], tmp_ptrans[0] );
                this->check_translation( rotpos[ia*3+1], tmp_ptrans[1] );
                this->check_translation( rotpos[ia*3+2], tmp_ptrans[2] );

                this->check_boundary( rotpos[ia*3+0] );
                this->check_boundary( rotpos[ia*3+1] );
                this->check_boundary( rotpos[ia*3+2] );
            }
            //order translated atomic positions for current species
            this->atom_ordering_new(rotpos + istart[it] * 3, na[it], index + istart[it]);
        }

        no_diff = true;
        //compare the two lattices 'one-by-one' whether they are identical
        for (int it = 0; it < ntype; it++)
        {
            for (int ia = istart[it]; ia < na[it] + istart[it]; ia++)
            {
                //take the difference of the rotated and the original coordinates
                diff.x = this->check_diff( pos[ia*3+0], rotpos[ia*3+0]);
                diff.y = this->check_diff( pos[ia*3+1], rotpos[ia*3+1]);
                diff.z = this->check_diff( pos[ia*3+2], rotpos[ia*3+2]);
                //only if all "diff" are zero vectors, flag will remain "1"
                if (!equal(diff.x,0.0)||
                    !equal(diff.y,0.0)||
                    !equal(diff.z,0.0))
                {
                    no_diff = false;
                    break;
                }
            }
            if (!no_diff) {
                break;
            }
        }

        //the current test is successful
        if (no_diff) {
            ptrans.push_back(ModuleBase::Vector3<double>(tmp_ptrans[0],
                                                         tmp_ptrans[1],
                                                         tmp_ptrans[2]));
        }
        //restore the original rotated coordinates by subtracting "ptrans"
        for (int it = 0; it < ntype; it++)
        {
            for (int ia = istart[it]; ia < na[it] + istart[it]; ia++)
            {
                rotpos[ia*3+0] -= tmp_ptrans[0];
                rotpos[ia*3+1] -= tmp_ptrans[1];
                rotpos[ia*3+2] -= tmp_ptrans[2];
            }
        }
    }
    int ntrans=ptrans.size();
    if (ntrans <= 1)
    {
        GlobalV::ofs_running<<"Original cell was already a primitive cell."<<std::endl;
        this->p1=this->a1;
        this->p2=this->a2;
        this->p3=this->a3;
        this->pbrav=this->real_brav;
        this->ncell=1;
        for (int i = 0; i < 6; ++i) {
            this->pcel_const[i] = this->cel_const[i];
        }
        return;
    }

    //sort ptrans:
    double* ptrans_array = new double[ntrans*3];
    for(int i=0;i<ntrans;++i)
    {
        ptrans_array[i*3]=ptrans[i].x;
        ptrans_array[i*3+1]=ptrans[i].y;
        ptrans_array[i*3+2]=ptrans[i].z;
    }
    this->atom_ordering_new(ptrans_array, ntrans, index);
    // std::cout<<"final ptrans:"<<std::endl;
    for(int i=0;i<ntrans;++i)
    {
        ptrans[i].x=ptrans_array[i*3];
        ptrans[i].y=ptrans_array[i*3+1];
        ptrans[i].z=ptrans_array[i*3+2];
        // std::cout<<ptrans[i].x<<" "<<ptrans[i].y<<" "<<ptrans[i].z<<std::endl;
    }
    delete[] ptrans_array;

    //calculate lattice vectors of pricell: 
    // find the first non-zero ptrans on all 3 directions 
    ModuleBase::Vector3<double> b1, b2, b3;
    int iplane=0, jplane=0, kplane=0;
    //1. kplane for b3
    while (kplane < ntrans
           && std::abs(ptrans[kplane].z - ptrans[0].z) < this->epsilon) {
        ++kplane;
    }
    if (kplane == ntrans) {
        kplane = 0; // a3-direction have no smaller pricell
    }
    b3=kplane>0 ? 
        ModuleBase::Vector3<double>(ptrans[kplane].x, ptrans[kplane].y, ptrans[kplane].z) : 
        ModuleBase::Vector3<double>(0, 0, 1);
    //2. jplane for b2 (not collinear with b3)
    jplane=kplane+1;
    while (jplane < ntrans
           && (std::abs(ptrans[jplane].y - ptrans[0].y) < this->epsilon
               || equal((ptrans[jplane] ^ b3).norm(), 0))) {
        ++jplane;
    }
    if (jplane == ntrans) {
        jplane = kplane; // a2-direction have no smaller pricell
    }
    b2=jplane>kplane ? 
        ModuleBase::Vector3<double>(ptrans[jplane].x, ptrans[jplane].y, ptrans[jplane].z) : 
        ModuleBase::Vector3<double>(0, 1, 0);
    //3. iplane for b1 (not coplane with <b2, b3>)
    iplane=jplane+1;
    while (iplane < ntrans
           && (std::abs(ptrans[iplane].x - ptrans[0].x) < this->epsilon
               || equal(ptrans[iplane] * (b2 ^ b3), 0))) {
        ++iplane;
    }
    b1=(iplane>jplane && iplane<ntrans)? 
        ModuleBase::Vector3<double>(ptrans[iplane].x, ptrans[iplane].y, ptrans[iplane].z) : 
        ModuleBase::Vector3<double>(1, 0, 0);    //a1-direction have no smaller pricell

    // std::cout<<"iplane="<<iplane<<std::endl;
    // std::cout<<"jplane="<<jplane<<std::endl;
    // std::cout<<"kplane="<<kplane<<std::endl;
    // std::cout<<"b1="<<b1.x<<" "<<b1.y<<" "<<b1.z<<std::endl;
    // std::cout<<"b2="<<b2.x<<" "<<b2.y<<" "<<b2.z<<std::endl;
    // std::cout<<"b3="<<b3.x<<" "<<b3.y<<" "<<b3.z<<std::endl;

    ModuleBase::Matrix3 coeff(b1.x, b1.y, b1.z, b2.x, b2.y, b2.z, b3.x, b3.y, b3.z);
    this->plat=coeff*this->optlat;

    //deal with collineation caused by default b1, b2, b3
    if(equal(plat.Det(), 0))
    {
        if(kplane==0)   //try a new b3
        {
            std::cout<<"try a new b3"<<std::endl;
            if(jplane>kplane)   // use default b2
            {
                coeff.e31=0;
                coeff.e32=1;
                coeff.e33=0;
            }
            else    //use default b1
            {
                coeff.e31=1;
                coeff.e32=0;
                coeff.e33=0;
            }
        }
        else if(jplane<=kplane)
        {
            //std::cout<<"try a new b2"<<std::endl;
            //use default b3
            coeff.e21=0;
            coeff.e22=0;
            coeff.e23=1;
        }
        else
        {
            //std::cout<<"try a new b1"<<std::endl;
            //use default b3
            coeff.e11=0;
            coeff.e12=0;
            coeff.e13=1;
        }
        this->plat=coeff*this->optlat;
        assert(!equal(plat.Det(), 0));
    }

    this->p1.x=plat.e11;
    this->p1.y=plat.e12;
    this->p1.z=plat.e13;
    this->p2.x=plat.e21;
    this->p2.y=plat.e22;
    this->p2.z=plat.e23;       
    this->p3.x=plat.e31;
    this->p3.y=plat.e32;
    this->p3.z=plat.e33;

#ifdef __DEBUG
    GlobalV::ofs_running<<"lattice vectors of primitive cell (initial):"<<std::endl;
    GlobalV::ofs_running<<p1.x<<" "<<p1.y<<" "<<p1.z<<std::endl;
    GlobalV::ofs_running<<p2.x<<" "<<p2.y<<" "<<p2.z<<std::endl;
    GlobalV::ofs_running<<p3.x<<" "<<p3.y<<" "<<p3.z<<std::endl;
#endif

    // get the optimized primitive cell
    std::string pbravname;
    ModuleBase::Vector3<double> p01=p1, p02=p2, p03=p3;
    double pcel_pre_const[6];
    for (int i = 0; i < 6; ++i) {
        pcel_pre_const[i] = pcel_const[i];
    }
    this->lattice_type(p1, p2, p3, p01, p02, p03, pcel_const, pcel_pre_const, pbrav, pbravname, atoms, false, nullptr);

    this->plat.e11=p1.x;
    this->plat.e12=p1.y;
    this->plat.e13=p1.z;
    this->plat.e21=p2.x;
    this->plat.e22=p2.y;
    this->plat.e23=p2.z;
    this->plat.e31=p3.x;
    this->plat.e32=p3.y;
    this->plat.e33=p3.z;

#ifdef __DEBUG
    GlobalV::ofs_running<<"lattice vectors of primitive cell (optimized):"<<std::endl;
    GlobalV::ofs_running<<p1.x<<" "<<p1.y<<" "<<p1.z<<std::endl;
    GlobalV::ofs_running<<p2.x<<" "<<p2.y<<" "<<p2.z<<std::endl;
    GlobalV::ofs_running<<p3.x<<" "<<p3.y<<" "<<p3.z<<std::endl;
#endif

    GlobalV::ofs_running<<"(for primitive cell:)"<<std::endl;
    Symm_Other::print1(this->pbrav, this->pcel_const, GlobalV::ofs_running);

    //count the number of pricells
    GlobalV::ofs_running<<"optimized lattice volume: "<<this->optlat.Det()<<std::endl;
    GlobalV::ofs_running<<"optimized primitive cell volume: "<<this->plat.Det()<<std::endl;
    double ncell_double = std::abs(this->optlat.Det()/this->plat.Det());
    this->ncell=floor(ncell_double+0.5);

    auto reset_pcell = [this]() -> void {
        std::cout << " Now regard the structure as a primitive cell." << std::endl;
        this->ncell = 1;
        this->ptrans = std::vector<ModuleBase::Vector3<double> >(1, ModuleBase::Vector3<double>(0, 0, 0));
        GlobalV::ofs_running << "WARNING: Original cell may have more than one primitive cells, \
        but we have to treat it as a primitive cell. Use a larger `symmetry_prec`to avoid this warning." << std::endl;
        };
    if (this->ncell != ntrans)
    {
        std::cout << " WARNING: PRICELL: NCELL != NTRANS !" << std::endl;
        std::cout << " NCELL=" << ncell << ", NTRANS=" << ntrans << std::endl;
        std::cout << " Suggest solution: Use a larger `symmetry_prec`. " << std::endl;
        reset_pcell();
        return;
    }
    if(std::abs(ncell_double-double(this->ncell)) > this->epsilon*100)
    {
        std::cout << " WARNING: THE NUMBER OF PRIMITIVE CELL IS NOT AN INTEGER !" << std::endl;
        std::cout << " NCELL(double)=" << ncell_double << ", NTRANS=" << ncell << std::endl;
        std::cout << " Suggest solution: Use a larger `symmetry_prec`. " << std::endl;
        reset_pcell();
        return;
    }
    GlobalV::ofs_running<<"Original cell was built up by "<<this->ncell<<" primitive cells."<<std::endl;

    //convert ptrans to input configuration
    ModuleBase::Matrix3 inputlat(s1.x, s1.y, s1.z, s2.x, s2.y, s2.z, s3.x, s3.y, s3.z);
    this->gtrans_convert(ptrans.data(), ptrans.data(), ntrans, this->optlat, inputlat );
    
    //how many pcell in supercell
    int n1, n2, n3;
    ModuleBase::Matrix3 nummat0=this->optlat*this->plat.Inverse();
    ModuleBase::Matrix3 nummat, transmat;
    hermite_normal_form(nummat0, nummat, transmat);
    n1=floor (nummat.e11 + epsilon);
    n2=floor (nummat.e22 + epsilon);
    n3=floor (nummat.e33 + epsilon);
    if(n1*n2*n3 != this->ncell) 
    {
        std::cout << " WARNING: Number of cells and number of vectors did not agree.";
        std::cout<<"Try to change symmetry_prec in INPUT." << std::endl;
        reset_pcell();
    }
    return;
}


//modified by shu on 2010.01.20
void Symmetry::rho_symmetry( double *rho,
                             const int &nr1, const int &nr2, const int &nr3)
{
    ModuleBase::timer::tick("Symmetry","rho_symmetry");

	// allocate flag for each FFT grid.
    bool* symflag = new bool[nr1 * nr2 * nr3];
    for (int i=0; i<nr1*nr2*nr3; i++)
    {
        symflag[i] = false;
    }

    assert(nrotk >0 );
    assert(nrotk <=48 );
    int *ri = new int[nrotk];
    int *rj = new int[nrotk];
    int *rk = new int[nrotk];

    int ci = 0;
    for (int i = 0; i< nr1; ++i)
    {
        for (int j = 0; j< nr2; ++j)
        {
            for (int k = 0; k< nr3; ++k)
            {
                if (!symflag[i * nr2 * nr3 + j * nr3 + k])
                {
                    double sum = 0;

                    for (int isym = 0; isym < nrotk; ++isym)
                    {
                        this->rotate(gmatrix[isym], gtrans[isym], i, j, k, nr1, nr2, nr3, ri[isym], rj[isym], rk[isym]);
                        const int index = ri[isym] * nr2 * nr3 + rj[isym] * nr3 + rk[isym];
                        sum += rho[ index ];
                    }
                    sum /= nrotk;

                    for (int isym = 0; isym < nrotk; ++isym)
                    {
                        const int index = ri[isym] * nr2 * nr3 + rj[isym] * nr3 + rk[isym];
                        rho[index] = sum;
                        symflag[index] = true;
                    }
                }
            }
        }
    }

    delete[] symflag;
    delete[] ri;
    delete[] rj;
    delete[] rk;
    ModuleBase::timer::tick("Symmetry","rho_symmetry");
}


void Symmetry::rhog_symmetry(std::complex<double> *rhogtot, 
    int* ixyz2ipw, const int &nx, const int &ny, const int &nz, 
    const int &fftnx, const int &fftny, const int &fftnz)
{
    ModuleBase::timer::tick("Symmetry","rhog_symmetry");
// ----------------------------------------------------------------------
// the current way is to cluster the FFT grid points into groups in advance.
// and use OpenMP to realize parallel calculation, one thread works in one group.
// ----------------------------------------------------------------------

	// allocate flag for each FFT grid.
    int* symflag = new int[fftnx*fftny*fftnz];// which group the grid belongs to
    int(*isymflag)[48] = new int[fftnx*fftny*fftnz][48];//which rotration operation the grid corresponds to
    int(*table_xyz)[48] = new int[fftnx * fftny * fftnz][48];// group information
    int* count_xyz = new int[fftnx * fftny * fftnz];// how many symmetry operations has been covered
    for (int i = 0; i < fftnx * fftny * fftnz; i++)
    {
        symflag[i] = -1;
    }
    int group_index = 0;

    assert(nrotk >0 );
    assert(nrotk <=48 );

    //map the gmatrix to inv
    std::vector<int>invmap(this->nrotk, -1);
    this->gmatrix_invmap(kgmatrix, nrotk, invmap.data());

// ---------------------------------------------------
/* This code defines a lambda function called "rotate_recip" that takes 
    a 3x3 matrix and a 3D vector as input. It performs a rotation operation 
    on the vector using the matrix and returns the rotated vector. 
    Specifically, it calculates the new coordinates of the vector after 
    the rotation and applies periodic boundary conditions to ensure that 
    the coordinates are within the FFT-grid dimensions. 
    The rotated vector is returned by modifying the input vector.
*/
// ---------------------------------------------------
    //rotate function (different from real space, without scaling gmatrix)
    auto rotate_recip = [&] (ModuleBase::Matrix3& g, ModuleBase::Vector3<int>& g0, int& ii, int& jj, int& kk) 
    {
        ii = int(g.e11 * g0.x + g.e21 * g0.y + g.e31 * g0.z) ;
        if (ii < 0)
        {
            ii += 10 * nx;
        }
        ii = ii%nx;
        jj = int(g.e12 * g0.x + g.e22 * g0.y + g.e32 * g0.z) ;
        if (jj < 0)
        {
            jj += 10 * ny;
        }
        jj = jj%ny;
        kk = int(g.e13 * g0.x + g.e23 * g0.y + g.e33 * g0.z);
        if (kk < 0)
        {
            kk += 10 * nz;
        }
        kk = kk%nz;
        return;
    };

// -------------------------------------------
/* 
    Trying to group fft grids first.
    It iterates over each FFT-grid point and checks if it is within the PW-sphere. If it is, put all the FFT-grid points connected by the 
    rotation operation into one group( the index is stored in int(*table_xyz)).
    The code marks the point as processed to avoid redundant calculations
    by using int* symflag.
*/ 
// -------------------------------------------
    ModuleBase::timer::tick("Symmetry","group_fft_grids");
    for (int i = 0; i< fftnx; ++i)
    {
        //tmp variable
        ModuleBase::Vector3<int> tmp_gdirect0(0, 0, 0);
        tmp_gdirect0.x=(i>int(nx/2)+1)?(i-nx):i;
        for (int j = 0; j< fftny; ++j)
        {
            tmp_gdirect0.y=(j>int(ny/2)+1)?(j-ny):j;
            for (int k = 0; k< fftnz; ++k)
            {
                int ixyz0=(i*fftny+j)*fftnz+k;
                if (symflag[ixyz0] == -1)
                {
                    int ipw0=ixyz2ipw[ixyz0];
                    //if a fft-grid is not in pw-sphere, just do not consider it.
                    if (ipw0 == -1) {
                        continue;
                    }
                    tmp_gdirect0.z=(k>int(nz/2)+1)?(k-nz):k;
                    int rot_count=0;
                    for (int isym = 0; isym < nrotk; ++isym)
                    {
                        if (invmap[isym] < 0 || invmap[isym] > nrotk) { continue; }
                        //tmp variables  
                        int ii, jj, kk=0;
                        rotate_recip(kgmatrix[invmap[isym]], tmp_gdirect0, ii, jj, kk);
                        if(ii>=fftnx || jj>=fftny || kk>= fftnz)
                        {
                            if(!PARAM.globalv.gamma_only_pw)
                            {
                                std::cout << " ROTATE OUT OF FFT-GRID IN RHOG_SYMMETRY !" << std::endl;
		                        ModuleBase::QUIT();
                            }
                            // for gamma_only_pw, just do not consider this rotation.
                            continue;
                        }
                        int ixyz=(ii*fftny+jj)*fftnz+kk;
                        //fft-grid index to (ip, ig)
                        int ipw=ixyz2ipw[ixyz];
                        if(ipw==-1) //not in pw-sphere
                        {
                            continue;   //else, just skip it
                        }
                        symflag[ixyz] = group_index;
                        isymflag[group_index][rot_count] = invmap[isym];
                        table_xyz[group_index][rot_count] = ixyz;
                        ++rot_count;
                        assert(rot_count <= nrotk);
                        count_xyz[group_index] = rot_count;
                    }
                group_index++;
                }
            }
        }
    }
    ModuleBase::timer::tick("Symmetry","group_fft_grids");

// -------------------------------------------
/*  This code performs symmetry operations on the reciprocal space 
    charge density using FFT-grids. It iterates over each FFT-grid 
    point in a particular group, applies a phase factor and sums the 
    charge density over the symmetry operations, and then divides by 
    the number of symmetry operations. Finally, it updates the charge
    density for each FFT-grid point using the calculated sum.
*/ 
// -------------------------------------------
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
for (int g_index = 0; g_index < group_index; g_index++)
{
    // record the index and gphase but not the final gdirect for each symm-opt
    int *ipw_record = new int[nrotk];
    int *ixyz_record = new int[nrotk];
    std::complex<double>* gphase_record = new std::complex<double> [nrotk];
    std::complex<double> sum(0, 0);
    int rot_count=0;

    for (int c_index = 0; c_index < count_xyz[g_index]; ++c_index)
    {
        int ixyz0 = table_xyz[g_index][c_index];
        int ipw0 = ixyz2ipw[ixyz0];
                if (symflag[ixyz0] == g_index)
                {
                    // note : do not use PBC after rotation. 
                    // we need a real gdirect to get the correspoding rhogtot.
                    int k = ixyz0%fftnz;
                    int j = ((ixyz0-k)/fftnz)%fftny;
                    int i = ((ixyz0-k)/fftnz-j)/fftny;
                    //fft-grid index to gdirect
                    ModuleBase::Vector3<double> tmp_gdirect_double(0.0, 0.0, 0.0);
                    tmp_gdirect_double.x=static_cast<double>((i>int(nx/2)+1)?(i-nx):i);
                    tmp_gdirect_double.y=static_cast<double>((j>int(ny/2)+1)?(j-ny):j);
                    tmp_gdirect_double.z=static_cast<double>((k>int(nz/2)+1)?(k-nz):k);
                    //calculate phase factor
                    tmp_gdirect_double = tmp_gdirect_double * ModuleBase::TWO_PI;
                    double cos_arg = 0.0, sin_arg = 0.0;
                    double arg_gtrans = tmp_gdirect_double * gtrans[isymflag[g_index][c_index]];
                    std::complex<double> phase_gtrans (ModuleBase::libm::cos(arg_gtrans), ModuleBase::libm::sin(arg_gtrans));
                    // for each pricell in supercell:
                    for (int ipt = 0;ipt < ((ModuleSymmetry::Symmetry::pricell_loop) ? this->ncell : 1);++ipt)
                    {
                        double arg = tmp_gdirect_double * ptrans[ipt];
                        double tmp_cos = 0.0, tmp_sin = 0.0;
                        ModuleBase::libm::sincos(arg, &tmp_sin, &tmp_cos);
                        cos_arg += tmp_cos;
                        sin_arg += tmp_sin;
                    }
                    // add nothing to sum, so don't consider this isym into rot_count
                    cos_arg/=static_cast<double>(ncell);
                    sin_arg/=static_cast<double>(ncell);
                    //deal with double-zero
                    if (equal(cos_arg, 0.0) && equal(sin_arg, 0.0)) {
                        continue;
                    }
                    std::complex<double> gphase(cos_arg, sin_arg);
                    gphase = phase_gtrans * gphase;
                    //deal with small difference from 1
                    if (equal(gphase.real(), 1.0) && equal(gphase.imag(), 0)) {
                        gphase = std::complex<double>(1.0, 0.0);
                    }
                    gphase_record[rot_count]=gphase;
                    sum += rhogtot[ipw0]*gphase;
                    //record
                    ipw_record[rot_count]=ipw0;
                    ixyz_record[rot_count]=ixyz0;
                    ++rot_count;
                    //assert(rot_count<=nrotk);
                }//end if section
    }//end c_index loop
    sum /= rot_count;
    for (int isym = 0; isym < rot_count; ++isym)
        {
            rhogtot[ipw_record[isym]] = sum/gphase_record[isym];
        }
    //g_index++;
    //Clean the records variables for each fft grid point
    delete[] ipw_record;
    delete[] ixyz_record;
    delete[] gphase_record;
}//end g_index loop
    delete[] symflag;
    delete[] isymflag;
    delete[] table_xyz;
    delete[] count_xyz;
    ModuleBase::timer::tick("Symmetry","rhog_symmetry");
}

void Symmetry::set_atom_map(const Atom* atoms)
{
    ModuleBase::TITLE("Symmetry", "set_atom_map");
    if (this->isym_rotiat_.size() == this->nrotk) {
        return;
    }
    this->isym_rotiat_.resize(this->nrotk);
    for (int i = 0; i < this->nrotk; ++i) {
        this->isym_rotiat_[i].resize(this->nat, -1);
    }

    double* pos = this->newpos;
    double* rotpos = this->rotpos;
    ModuleBase::GlobalFunc::ZEROS(pos, this->nat * 3);
    int iat = 0;
    for (int it = 0; it < this->ntype; it++)
    {
        for (int ia = 0; ia < this->na[it]; ia++)
        {
            pos[3 * iat] = atoms[it].taud[ia].x;
            pos[3 * iat + 1] = atoms[it].taud[ia].y;
            pos[3 * iat + 2] = atoms[it].taud[ia].z;
            for (int k = 0; k < 3; ++k)
            {
                this->check_translation(pos[iat * 3 + k], -floor(pos[iat * 3 + k]));
                this->check_boundary(pos[iat * 3 + k]);
            }
            iat++;
        }
    }
    for (int it = 0; it < this->ntype; it++)
    {
        for (int ia = istart[it]; ia < istart[it] + na[it]; ++ia)
        {
            const int xx = ia * 3; const int yy = ia * 3 + 1; const int zz = ia * 3 + 2;
            for (int k = 0;k < this->nrotk;++k)
            {
                rotpos[xx] = pos[xx] * gmatrix[k].e11 + pos[yy] * gmatrix[k].e21 + pos[zz] * gmatrix[k].e31 + gtrans[k].x;
                rotpos[yy] = pos[xx] * gmatrix[k].e12 + pos[yy] * gmatrix[k].e22 + pos[zz] * gmatrix[k].e32 + gtrans[k].y;
                rotpos[zz] = pos[xx] * gmatrix[k].e13 + pos[yy] * gmatrix[k].e23 + pos[zz] * gmatrix[k].e33 + gtrans[k].z;

                check_translation(rotpos[xx], -floor(rotpos[xx]));
                check_boundary(rotpos[xx]);
                check_translation(rotpos[yy], -floor(rotpos[yy]));
                check_boundary(rotpos[yy]);
                check_translation(rotpos[zz], -floor(rotpos[zz]));
                check_boundary(rotpos[zz]);

                for (int ja = istart[it]; ja < istart[it] + na[it]; ++ja)
                {
                    double diff1 = check_diff(pos[ja * 3], rotpos[xx]);
                    double diff2 = check_diff(pos[ja * 3 + 1], rotpos[yy]);
                    double diff3 = check_diff(pos[ja * 3 + 2], rotpos[zz]);
                    if (equal(diff1, 0.0) && equal(diff2, 0.0) && equal(diff3, 0.0))
                    {
                        this->isym_rotiat_[k][ia] = ja;

                        break;
                    }
                }
            }
        }
    }
}

void Symmetry::symmetrize_vec3_nat(double* v)const   // pengfei 2016-12-20
{
    ModuleBase::TITLE("Symmetry", "symmetrize_vec3_nat");
    double* vtot;
    int* n;
    vtot = new double[nat * 3]; ModuleBase::GlobalFunc::ZEROS(vtot, nat * 3);
    n = new int[nat]; ModuleBase::GlobalFunc::ZEROS(n, nat);

    for (int j = 0;j < nat; ++j)
    {
        const int jx = j * 3; const int jy = j * 3 + 1; const int jz = j * 3 + 2;
        for (int k = 0; k < nrotk; ++k)
        {
            int l = this->isym_rotiat_[k][j];
            if (l < 0) {
                continue;
            }
            vtot[l * 3] = vtot[l * 3] + v[jx] * gmatrix[k].e11 + v[jy] * gmatrix[k].e21 + v[jz] * gmatrix[k].e31;
            vtot[l * 3 + 1] = vtot[l * 3 + 1] + v[jx] * gmatrix[k].e12 + v[jy] * gmatrix[k].e22 + v[jz] * gmatrix[k].e32;
            vtot[l * 3 + 2] = vtot[l * 3 + 2] + v[jx] * gmatrix[k].e13 + v[jy] * gmatrix[k].e23 + v[jz] * gmatrix[k].e33;
            n[l]++;
        }
	}
    for (int j = 0;j < nat; ++j)
    {
        v[j * 3] = vtot[j * 3] / n[j];
        v[j * 3 + 1] = vtot[j * 3 + 1] / n[j];
        v[j * 3 + 2] = vtot[j * 3 + 2] / n[j];
    }
    delete[] vtot;
    delete[] n;
	return;
}

void Symmetry::symmetrize_mat3(ModuleBase::matrix& sigma, const Lattice& lat)const   //zhengdy added 2017
{
    ModuleBase::matrix A = lat.latvec.to_matrix();
    ModuleBase::matrix AT = lat.latvec.Transpose().to_matrix();
    ModuleBase::matrix invA = lat.GT.to_matrix();
    ModuleBase::matrix invAT = lat.G.to_matrix();
    ModuleBase::matrix tot_sigma(3, 3, true);
    sigma = A * sigma * AT;
    for (int k = 0; k < nrotk; ++k) {
        tot_sigma += invA * gmatrix[k].to_matrix() * sigma
                     * gmatrix[k].Transpose().to_matrix() * invAT;
    }
    sigma = tot_sigma * static_cast<double>(1.0 / nrotk);
	return;
}

void Symmetry::gmatrix_convert_int(const ModuleBase::Matrix3* sa, ModuleBase::Matrix3* sb, 
        const int n, const ModuleBase::Matrix3 &a, const ModuleBase::Matrix3 &b) const
{
    auto round = [](double x){return (x>0.0)?floor(x+0.5):ceil(x-0.5);};
    ModuleBase::Matrix3 ai = a.Inverse();
    ModuleBase::Matrix3 bi = b.Inverse();
    for (int i=0;i<n;++i)
    {
          sb[i]=b*ai*sa[i]*a*bi;
          //to int 
          sb[i].e11=round(sb[i].e11);
          sb[i].e12=round(sb[i].e12);
          sb[i].e13=round(sb[i].e13);
          sb[i].e21=round(sb[i].e21);
          sb[i].e22=round(sb[i].e22);
          sb[i].e23=round(sb[i].e23);
          sb[i].e31=round(sb[i].e31);
          sb[i].e32=round(sb[i].e32);
          sb[i].e33=round(sb[i].e33);
    }
}
void Symmetry::gmatrix_convert(const ModuleBase::Matrix3* sa, ModuleBase::Matrix3* sb, 
        const int n, const ModuleBase::Matrix3 &a, const ModuleBase::Matrix3 &b)const
{
    ModuleBase::Matrix3 ai = a.Inverse();
    ModuleBase::Matrix3 bi = b.Inverse();
    for (int i=0;i<n;++i)
    {
          sb[i]=b*ai*sa[i]*a*bi;
    }
}
void Symmetry::gtrans_convert(const ModuleBase::Vector3<double>* va, ModuleBase::Vector3<double>* vb, 
        const int n, const ModuleBase::Matrix3 &a, const ModuleBase::Matrix3 &b)const
{
    ModuleBase::Matrix3 bi = b.Inverse();
    for (int i=0;i<n;++i)
    {
          vb[i]=va[i]*a*bi;
    }
}
void Symmetry::gmatrix_invmap(const ModuleBase::Matrix3* s, const int n, int* invmap) const
{
    ModuleBase::Matrix3 eig(1, 0, 0, 0, 1, 0, 0, 0, 1);
    ModuleBase::Matrix3 tmp;
    for (int i=0;i<n;++i)
    {
        for (int j=i;j<n;++j)
        {
            tmp=s[i]*s[j];
            if(equal(tmp.e11, 1) && equal(tmp.e22, 1) && equal(tmp.e33, 1) &&
                equal(tmp.e12, 0) && equal(tmp.e21, 0) && equal(tmp.e13, 0) &&
                equal(tmp.e31, 0) && equal(tmp.e23, 0) && equal(tmp.e32, 0))
            {
                invmap[i]=j;
                invmap[j]=i;
                break;
            }
        }
    }
}

void Symmetry::get_shortest_latvec(ModuleBase::Vector3<double> &a1, 
        ModuleBase::Vector3<double> &a2, ModuleBase::Vector3<double> &a3) const
{
    double len1=a1.norm();
    double len2=a2.norm();
    double len3=a3.norm();
    bool flag=true; //at least one iter
    auto loop = [this, &flag](ModuleBase::Vector3<double> &v1, ModuleBase::Vector3<double>&v2, double &len)
    {
        bool fa=false, fb=false;
        // loop a
        double tmp_len=(v1-v2).norm();
        while (tmp_len < len-epsilon)
        {
            v1=v1-v2;
            len=v1.norm();
            tmp_len=(v1-v2).norm();
            fa=true;
        }
        // loop b
        tmp_len=(v1+v2).norm();
        while(tmp_len < len-epsilon)
        {
            assert(!fa);
            v1=v1+v2;
            len=v1.norm();
            tmp_len=(v1+v2).norm();
            fb=true;
        }
        if (fa || fb) {
            flag = true;
        }
        return;
    };
    while(flag) //iter
    {
        flag=false;
        // if any of a1, a2, a3 is updated, flag will become true.
        // which means a further search is needed.
        loop(a1, a2, len1);
        loop(a1, a3, len1);
        loop(a2, a1, len2);
        loop(a2, a3, len2);
        loop(a3, a1, len3);
        loop(a3, a2, len3);
    }
    return;
}

void Symmetry::get_optlat(ModuleBase::Vector3<double> &v1, ModuleBase::Vector3<double> &v2, 
        ModuleBase::Vector3<double> &v3, ModuleBase::Vector3<double> &w1, 
        ModuleBase::Vector3<double> &w2, ModuleBase::Vector3<double> &w3, 
        int& real_brav, double* cel_const, double* tmp_const) const
{
    ModuleBase::Vector3<double> r1, r2, r3;
    double cos1 = 1;
    double cos2 = 1;
    double cos3 = 1;
    int nif = 0;
    int ibrav;
    for (int n33 = -2; n33 < 3; ++n33)
    {
        for (int n32 = -2; n32 < 3; ++n32)
        {
            for (int n31 = -2; n31 < 3; ++n31)
            {
                for (int n23 = -2; n23 < 3; ++n23)
                {
                    for (int n22 = -2; n22 < 3; ++n22)
                    {
                        for (int n21 = -2; n21 < 3; ++n21)
                        {
                            for (int n13 = -2; n13 < 3; ++n13)
                            {
                                for (int n12 = -2; n12 < 3; ++n12)
                                {
                                    for (int n11 = -2; n11 < 3; ++n11)
                                    {
                                        ModuleBase::Matrix3 mat(n11, n12, n13, n21, n22, n23, n31, n32, n33);

                                        if (equal(mat.Det(),1.0))
                                        {
                                            r1.x = n11 * v1.x + n12 * v2.x + n13 * v3.x;
                                            r1.y = n11 * v1.y + n12 * v2.y + n13 * v3.y;
                                            r1.z = n11 * v1.z + n12 * v2.z + n13 * v3.z;
                                     
									        r2.x = n21 * v1.x + n22 * v2.x + n23 * v3.x;
                                            r2.y = n21 * v1.y + n22 * v2.y + n23 * v3.y;
                                            r2.z = n21 * v1.z + n22 * v2.z + n23 * v3.z;
                                     
									        r3.x = n31 * v1.x + n32 * v2.x + n33 * v3.x;
                                            r3.y = n31 * v1.y + n32 * v2.y + n33 * v3.y;
                                            r3.z = n31 * v1.z + n32 * v2.z + n33 * v3.z;
                                            //std::cout << "mat = " << n11 <<" " <<n12<<" "<<n13<<" "<<n21<<" "<<n22<<" "<<n23<<" "<<n31<<" "<<n32<<" "<<n33<<std::endl;
											
                                            ibrav = standard_lat(r1, r2, r3, cel_const);
//                                            if(brav == 8)
//                                            {
//                                               std::cout << "mat = " << n11 <<" " <<n12<<" "<<n13<<" "<<n21<<" "<<n22<<" "<<n23<<" "<<n31<<" "<<n32<<" "<<n33<<std::endl;
//                                            }

/*
											if(n11== 1 && n12==0 && n13==-2 && n21==2 && n22==1 && n23==-1
												&& n31==-2 && n32==-2 && n33==-1)
											{
												++nif;
												GlobalV::ofs_running << " " << std::endl;
												GlobalV::ofs_running << std::setw(8) << nif << std::setw(5) << n11 << std::setw(5) << n12
													<< std::setw(5) << n13 << std::setw(5) << n21 << std::setw(5) << n22
													<< std::setw(5) << n23 << std::setw(5) << n31 << std::setw(5) << n32
													<< std::setw(5) << n33 << std::setw(5) << ibrav << std::endl;
												GlobalV::ofs_running << " r1: " << r1.x << " " << r1.y << " " << r1.z << std::endl;
												GlobalV::ofs_running << " r2: " << r2.x << " " << r2.y << " " << r2.z << std::endl;
												GlobalV::ofs_running << " r3: " << r3.x << " " << r3.y << " " << r3.z << std::endl;
												GlobalV::ofs_running << " cel_const[3]=" << cel_const[3] << std::endl;
												GlobalV::ofs_running << " cel_const[4]=" << cel_const[4] << std::endl;
												GlobalV::ofs_running << " cel_const[5]=" << cel_const[5] << std::endl;
											}
											*/
//											if(brav == 14)
//											{
//												GlobalV::ofs_running << " ABS(CELLDM(4))=" << fabs(cel_const[3]) << std::endl;
//												GlobalV::ofs_running << " ABS(CELLDM(5))=" << fabs(cel_const[4]) << std::endl;
//												GlobalV::ofs_running << " ABS(CELLDM(6))=" << fabs(cel_const[5]) << std::endl;
//											}

                                            if ( ibrav < real_brav || ( ibrav == real_brav
                                                    && ( fabs(cel_const[3]) < (cos1-1.0e-9) )
                                                    && ( fabs(cel_const[4]) < (cos2-1.0e-9) )
                                                    && ( fabs(cel_const[5]) < (cos3-1.0e-9) )) //mohan fix bug 2012-01-15, not <=
                                               )
                                            {
												/*
												GlobalV::ofs_running << "\n IBRAV=" << brav << " ITYP=" << temp_brav << std::endl;
												GlobalV::ofs_running << " ABS(CELLDM(4))=" << fabs(cel_const[3]) << std::endl;
												GlobalV::ofs_running << " ABS(CELLDM(5))=" << fabs(cel_const[4]) << std::endl;
												GlobalV::ofs_running << " ABS(CELLDM(6))=" << fabs(cel_const[5]) << std::endl;
												GlobalV::ofs_running << " COS1=" << cos1 << std::endl;
												GlobalV::ofs_running << " COS2=" << cos2 << std::endl;
												GlobalV::ofs_running << " COS3=" << cos3 << std::endl;
												*/
												/*
												GlobalV::ofs_running << " r1: " << r1.x << " " << r1.y << " " << r1.z << std::endl;
												GlobalV::ofs_running << " r2: " << r2.x << " " << r2.y << " " << r2.z << std::endl;
												GlobalV::ofs_running << " r3: " << r3.x << " " << r3.y << " " << r3.z << std::endl;
												GlobalV::ofs_running << " N=" << n11 << " " << n12 << " " << n13 
												<< " " << n21 << " " << n22 << " " << n23 
												<< " " << n31 << " " << n32 << " " << n33 << std::endl; 
												*/
												//out.printM3("mat",mat);
                                                real_brav = ibrav;
												
                                                cos1 = fabs(cel_const[3]);
                                                cos2 = fabs(cel_const[4]);
                                                cos3 = fabs(cel_const[5]);

                                                for (int i = 0; i < 6; ++i)
                                                {
                                                    tmp_const[i] = cel_const[i];
                                                }
                                                w1 = r1;
                                                w2 = r2;
                                                w3 = r3;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}

void Symmetry::hermite_normal_form(const ModuleBase::Matrix3 &s3, ModuleBase::Matrix3 &h3, ModuleBase::Matrix3 &b3) const
{
    ModuleBase::TITLE("Symmetry","hermite_normal_form");
    // check the non-singularity and integer elements of s
#ifdef __DEBUG
    assert(!equal(s3.Det(), 0.0));
#endif
    auto near_equal = [this](double x, double y) {return fabs(x - y) < 10 * epsilon;};
    ModuleBase::matrix s = s3.to_matrix();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0;j < 3;++j)
        {
            double sij_round = std::round(s(i, j));
#ifdef __DEBUG
            assert(near_equal(s(i, j), sij_round));
#endif
            s(i, j) = sij_round;
        }
    }

    // convert Matrix3 to matrix
    ModuleBase::matrix h=s, b(3, 3, true);
    b(0, 0)=1; b(1, 1)=1; b(2, 2)=1;

    // transform H into lower triangular form
    auto max_min_index = [&h, this](int row, int &i1_to_max, int &i2_to_min)
    {
        if(fabs(h(row, i1_to_max)) < fabs(h(row, i2_to_min)) - epsilon)
        {
            int tmp = i2_to_min;
            i2_to_min = i1_to_max;
            i1_to_max = tmp;
        }
        return;
    };
    auto max_min_index_row1 = [&max_min_index, &h, this](int &imax, int &imin)
    {
        int imid=1;
        imax=0; imin=2;
        max_min_index(0, imid, imin);
        max_min_index(0, imax, imid);
        max_min_index(0, imid, imin);
        if (equal(h(0, imin), 0)) {
            imin = imid;
        } else if (equal(h(0, imax), 0)) {
            imax = imid;
        }
        return;
    };
    auto swap_col = [&h, &b](int c1, int c2)
    {
        double tmp;
        for(int r=0;r<3;++r)
        {
            tmp = h(r, c2);
            h(r, c2)=h(r, c1);
            h(r, c1)=tmp;
            tmp = b(r, c2);
            b(r, c2)=b(r, c1);
            b(r, c1)=tmp;
        }
        return;
    };
    // row 1 
    int imax, imin;
    while(int(equal(h(0, 0), 0)) + int(equal(h(0, 1), 0)) + int(equal(h(0, 2), 0)) < 2)
    {
        max_min_index_row1(imax, imin);
        double f = floor((fabs(h(0, imax) )+ epsilon)/fabs(h(0, imin)));
        if (h(0, imax) * h(0, imin) < -epsilon) {
            f *= -1;
        }
        for(int r=0;r<3;++r) {h(r, imax) -= f*h(r, imin); b(r, imax) -= f*b(r, imin); }
    }
    if (equal(h(0, 0), 0)) {
        equal(h(0, 1), 0) ? swap_col(0, 2) : swap_col(0, 1);
    }
    if (h(0, 0) < -epsilon) {
        for (int r = 0; r < 3; ++r) {
            h(r, 0) *= -1;
            b(r, 0) *= -1;
        }
    }
    //row 2
    if (equal(h(1, 1), 0)) {
        swap_col(1, 2);
    }
    while(!equal(h(1, 2), 0))
    {
        imax=1, imin=2;
        max_min_index(1, imax, imin);
        double f = floor((fabs(h(1, imax) )+ epsilon)/fabs(h(1, imin)));
        if (h(1, imax) * h(1, imin) < -epsilon) {
            f *= -1;
        }
        for(int r=0;r<3;++r) {h(r, imax) -= f*h(r, imin); b(r, imax) -= f*b(r, imin); }
        if (equal(h(1, 1), 0)) {
            swap_col(1, 2);
        }
    }
    if (h(1, 1) < -epsilon) {
        for (int r = 0; r < 3; ++r) {
            h(r, 1) *= -1;
            b(r, 1) *= -1;
        }
    }
    //row3
    if (h(2, 2) < -epsilon) {
        for (int r = 0; r < 3; ++r) {
            h(r, 2) *= -1;
            b(r, 2) *= -1;
        }
    }
    // deal with off-diagonal elements
    while (h(1, 0) > h(1, 1) - epsilon) {
        for(int r=0;r<3;++r) {h(r, 0) -= h(r, 1); b(r, 0) -= b(r, 1);
        }
    }
    while (h(1, 0) < -epsilon) {
        for(int r=0;r<3;++r) {h(r, 0) += h(r, 1); b(r, 0) += b(r, 1);
        }
    }
    for(int j=0;j<2;++j)
    {
        while (h(2, j) > h(2, 2) - epsilon) {
            for(int r=0;r<3;++r) {h(r, j) -= h(r, 2); b(r, j) -= b(r, 2);
            }
        }
        while (h(2, j) < -epsilon) {
            for(int r=0;r<3;++r) {h(r, j) += h(r, 2); b(r, j) += b(r, 2);
            }
        }
    }

    //convert matrix to Matrix3
    h3.e11=h(0, 0); h3.e12=h(0, 1); h3.e13=h(0, 2);
    h3.e21=h(1, 0); h3.e22=h(1, 1); h3.e23=h(1, 2);
    h3.e31=h(2, 0); h3.e32=h(2, 1); h3.e33=h(2, 2);
    b3.e11=b(0, 0); b3.e12=b(0, 1); b3.e13=b(0, 2);
    b3.e21=b(1, 0); b3.e22=b(1, 1); b3.e23=b(1, 2);
    b3.e31=b(2, 0); b3.e32=b(2, 1); b3.e33=b(2, 2);

    //check s*b=h
    ModuleBase::matrix check_zeros = s3.to_matrix() * b - h;
#ifdef __DEBUG
    for (int i = 0;i < 3;++i)
        for(int j=0;j<3;++j)
            assert(near_equal(check_zeros(i, j), 0));
#endif
    return;
}

bool Symmetry::magmom_same_check(const Atom* atoms)const
{
    ModuleBase::TITLE("Symmetry", "magmom_same_check");
    bool pricell_loop = true;
    for (int it = 0;it < ntype;++it)
    {
        if (pricell_loop) {
            for (int ia = 1;ia < atoms[it].na;++ia)
            {
                if (!equal(atoms[it].m_loc_[ia].x, atoms[it].m_loc_[0].x) ||
                    !equal(atoms[it].m_loc_[ia].y, atoms[it].m_loc_[0].y) ||
                    !equal(atoms[it].m_loc_[ia].z, atoms[it].m_loc_[0].z))
                {
                    pricell_loop = false;
                    break;
                }
            }
        }
    }
    return pricell_loop;
}

bool Symmetry::is_all_movable(const Atom* atoms, const Statistics& st)const
{
    bool all_mbl = true;
    for (int iat = 0;iat < st.nat;++iat)
    {
        int it = st.iat2it[iat];
        int ia = st.iat2ia[iat];
        if (!atoms[it].mbl[ia].x || !atoms[it].mbl[ia].y || !atoms[it].mbl[ia].z)
        {
            all_mbl = false;
            break;
        }
    }
    return all_mbl;
}

void Symmetry::analyze_magnetic_group(const Atom* atoms, const Statistics& st, int& nrot_out, int& nrotk_out)
{
    // 1. classify atoms with different magmom
    //  (use symmetry_prec to judge if two magmoms are the same)
    std::vector<std::set<int>> mag_type_atoms;
    for (int it = 0;it < ntype;++it)
    {
        for (int ia = 0; ia < atoms[it].na; ++ia)
        {
            bool find = false;
            for (auto& mt : mag_type_atoms)
            {
                const int mag_iat = *mt.begin();
                const int mag_it = st.iat2it[mag_iat];
                const int mag_ia = st.iat2ia[mag_iat];
                if (it == mag_it && this->equal(atoms[it].mag[ia], atoms[mag_it].mag[mag_ia]))
                {
                    mt.insert(st.itia2iat(it, ia));
                    find = true;
                    break;
                }
            }
            if (!find)
            {
                mag_type_atoms.push_back(std::set<int>({ st.itia2iat(it,ia) }));
            }
        }
    }

    // 2. get the start index, number of atoms and positions for each mag_type
    std::vector<int> mag_istart(mag_type_atoms.size());
    std::vector<int> mag_na(mag_type_atoms.size());
    std::vector<double> mag_pos;
    int mag_itmin_type = 0;
    int mag_itmin_start = 0;
    for (int mag_it = 0;mag_it < mag_type_atoms.size(); ++mag_it)
    {
        mag_na[mag_it] = mag_type_atoms.at(mag_it).size();
        if (mag_it > 0)
        {
            mag_istart[mag_it] = mag_istart[mag_it - 1] + mag_na[mag_it - 1];
        }
        if (mag_na[mag_it] < mag_na[itmin_type])
        {
            mag_itmin_type = mag_it;
            mag_itmin_start = mag_istart[mag_it];
        }
        for (auto& mag_iat : mag_type_atoms.at(mag_it))
        {
            // this->newpos have been ordered by original structure(ntype, na), it cannot be directly used here.
            // we need to reset the calculate again the coordinate of the new structure.
            const ModuleBase::Vector3<double> direct_tmp = atoms[st.iat2it[mag_iat]].tau[st.iat2ia[mag_iat]] * this->optlat.Inverse();
            std::array<double, 3> direct = { direct_tmp.x, direct_tmp.y, direct_tmp.z };
            for (int i = 0; i < 3; ++i)
            {
                this->check_translation(direct[i], -floor(direct[i]));
                this->check_boundary(direct[i]);
                mag_pos.push_back(direct[i]);
            }
        }
    }
    // 3. analyze the effective structure
    this->getgroup(nrot_out, nrotk_out, GlobalV::ofs_running, this->nop, this->symop, this->gmatrix, this->gtrans,
        mag_pos.data(), this->rotpos, this->index, mag_type_atoms.size(), mag_itmin_type, mag_itmin_start, mag_istart.data(), mag_na.data());
}
}
