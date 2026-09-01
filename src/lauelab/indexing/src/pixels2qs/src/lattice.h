#ifndef LATTICE_H
#define LATTICE_H

#ifndef CHECK_FREE
#define CHECK_FREE(A)   { if(A) free(A); (A)=NULL;}
#endif

#ifndef STD_TEMP_DEF_
#define STD_TEMP_DEF_
#define STD_TEMP		22.5			/* standard temperature (C) */
#endif
#ifndef ROOM_TEMP_DEF_
#define ROOM_TEMP_DEF_
#define ROOM_TEMP		STD_TEMP		/* room temperature (C) */
#endif

#define STRUCTURE_ATOMS_MAX 2000		/* max number of atom types in a material structure (for Si only need 1) */

//	#ifndef hc_KEV_A
//	#define	hc_KEV_A 12.3984187			/* ( h*c (keV-), from 2006 CODATA,  http://physics.nist.gov/cuu/Constants/index.html */
//	#endif
//
//	#ifndef N_Avagadro
//	#define N_Avagadro 6.02214179e23	/* Avagadro's number, from 2006 CODATA,  http://physics.nist.gov/cuu/Constants/index.html */
//	#endif


/*
 *	struct latticeParameters {			// defines a lattice 3 lengths & 3 angles		OLD
 *		double  a,b,c;					// lattice constants (Angstroms)
 *		double  alpha,beta,gamma;		// lattice angles (radian)
 *		int		structure;				// structure number from International Tables, allowed values are [1-230]
 *		double  recip[3][3];			// reciprocal lattice from constants, has the 2PI in it
 *	};
 */


struct atomTypeStructure {				/* defines one type of atom (each atom may be in multiple positions) */
	char	name[60];					/* label for this atom, usually starts with atomic symbol */
	int		Zatom;						/* Z of the atom */
	double	x;							/* fractional coord along x lattice vector */
	double	y;
	double	z;
	double	occ;						/* occupancy */
	double	Debye;						/* Debye-Waller factor */
};

struct atomStructure {					/* defines single atome position, one atom type can produce multiple atom positions */
	size_t	type;						/* index into associated atomTypeStructure for this atom */
	int		Zatom;						/* Z of the atom */
	double	x;							/* fractional coord along x lattice vector */
	double	y;
	double	z;
	double	occ;						/* occupancy */
	double	Debye;						/* Debye-Waller factor */
};

struct equivOpsStructure {				/* holds all of the information about the symmetry operations */
	int		SpaceGroup;					/* Space Group number [1,230] */
	int		N;							/* number of symmetry ops, [1,192] */
	double ***equivXYZM;				/* the matricies */
	double **equivXYZB;					/* and the vectors */
};

struct crystalStructure {				/* defines a lattice 3 lengths & 3 angles */
	char	desc[256];					/* name or decription of this crystal */
	double  a,b,c;						/* lattice constants (Angstroms) */
	double  alpha,beta,gamma;			/* lattice angles (radian) */
	double	lengthUnits;				/* units for a,b,c   conversion from meters, Angstrom->1e10, nm->1e9, micron->1e6, ... */
	int		SpaceGroup;					/* Space Group number from International Tables, allowed values are [1-230] */
	double  direct[3][3];				/* direct lattice from constants, units agree with lengthUnits, column vecs, not C vecs */
	double  recip[3][3];				/* reciprocal lattice from constants, has the 2PI in it, column vectors not C vecs */
	double	Vc;							/* Volume of unit cell */
	double	density;					/* calculated density (g/cm^3) */
	double	alphaT;						/* coef of thermal expansion, a = ao*(1+alphaT*(TempC-22.5)) */

	size_t	Ntype;						/* number of atom types described here, this needs to be provided by the user */
	struct atomTypeStructure *atomType;	/* the atom positions, space must be allocated for this */

	size_t	N;							/* number of real atom (not just types), computed from atomType */
	struct atomStructure *atom;		 	/* the atom positions, space must be allocated for this */

	struct equivOpsStructure equiv;		/* the equivalent atom position operations for thie Space Group, computed internally from Space Group */

	int		useUnconventional;			/* flag, true means use Unconventional[3][3], provided by user, usually 0 */
	double  Unconventional[3][3];		/* transform matrix for an unconventional unit cel, for conventional, this will be identity matrix */
};



#define TRICLINIC    0
#define MONOCLINIC   1
#define ORTHORHOMBIC 2
#define TETRAGONAL   3
#define TRIGONAL	 4
#define HEXAGONAL    5
#define CUBIC		 6

#define P_CENTER	 0
#define F_CENTER	 1
#define B_CENTER	 2
#define RHOMBOHEDRAL 3
#define C_CENTER	 4
#define A_CENTER	 5

#define FCC 225							/* generic structure numbers */
#define BCC 229
#define DIA 227
#define SIMPLE_CUBIC 221
#define SAPPHIRE 167


#define ALLOW_FC(H,K,L) (!(((H)+(K))%2) && !(((K)+(L))%2))		/* H,K,L all even or all odd */
#define ALLOW_BC(H,K,L) (!(((H)+(K)+(L))%2))					/* !mod(round(h+k+l),2) */
#define ALLOW_CC(H,K,L) (!((H)+(K))%2)							/* !mod(round(h+k),2) */
#define ALLOW_AC(H,K,L) (!((K)+(L))%2)							/* !mod(round(k+l),2) */
#define ALLOW_RHOM_HEX(H,K,L) (((-(H)+(K)+(L))%3)==0 || (((H)-(K)+(L))%3)==0)   /* allowed are -H+K+L=3n or H-K+L=3n */
#define ALLOW_HEXAGONAL(H,K,L) ((((H)+2*(K))%3) || !((L)%2))   /* forbidden are: H+2K=3N with L odd */


size_t Fstruct(struct crystalStructure *xtal, long h, long k, long l, double *Freal, double *Fimag);

int ForceLatticeToStructure(struct crystalStructure *xtal);
int UpdateInternalsOfCrystalStructure(struct crystalStructure *xtal);

long allowedHKL(struct crystalStructure *xtal, long h, long k, long l);
int latticeSystem(int structNum);

int setAtomXYZs(struct crystalStructure *xtal);

/* char *getSymString(int structNum, char *str); */
char *symmetryString(int SG, char *sym);
int setDirectRecip(struct crystalStructure *xtal);
//double dSpacing(struct crystalStructure *xtal, double h, double k, double l, ...);
double dSpacing(struct crystalStructure *xtal, double h, double k, double l, double T);
double densityOfCrystalStructure(struct crystalStructure *xtal);

void lowestOrderHKL(long hkl[3]);
void lowestAllowedHKL(long hkl[3], struct crystalStructure *xtal);
void copyCrystalStructure(struct crystalStructure *destLat, struct crystalStructure *sourceLat);
void InitCleanCrystalStructure(struct crystalStructure *lat);
void freeCrystalStructure(struct crystalStructure *lat);

int SetSymOpsForSpaceGroup(int SpaceGroup, struct equivOpsStructure *sym);
void print_crystalStructure(FILE *f, struct crystalStructure *xtal);
long multiplicityOfAtomType(size_t type, struct crystalStructure *xtal);
int atomicNumber(char *ele);


/* void printMat33(double m[3][3]); */
/* void printVec3(double v[3]); */
/* void printEquivOpsStructure(struct equivOpsStructure *sym); */

/* char *latticeNames[]={"Triclinic","Monoclinic","Orthorhombic","Tetragonal","Trigonal","Hexagonal","Cubic"}; */

#endif
