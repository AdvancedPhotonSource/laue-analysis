/* 
 *	Normal use of these routines is as follows:
 *
 *	struct crystalStructure xtal;				// crystal structure, declare a crystalStructure:
 *	InitCleanCrystalStructure(&xtal);			// set all structure values to clean, and initialize everything
 *	setCrystalSilicon(&xtal);					// set lattice and atoms to something, from user input somehow.
 *	ForceLatticeToStructure(&xtal);				// set the direct and recip lattices, and also set the actual atom positions
 *
 *		// xtal is now ready to use, the things that you can do are:
 *
 *	print_crystalStructure(stdout,&xtal);		// print out all the xtal information
 *	Fstruct(&xtal,4,0,0, &Fr, &Fi);				// calculate structure factor, F(400) = Fr + i*Fi
 *	d = dSpacing(&xtal,1.5,2.0,2.2);			// calculate a d-spacing (note hkl are double, not int)
 *	if (allowedHKL(&xtal, 2,2,2))				// test if (222) is allowed
 *	lowestOrderHKL(hkl);						// converts hkl[3] to the lowest integer hkl (whether or not it is allowed)
 *	lowestAllowedHKL(hkl[3]&xtal);				// converts hkl[3] to the lowest integer hkl that is allowed
 *	copyCrystalStructure(&dest, &source);		// copy one crystal struct to another
 *	freeCrystalStructure(&xtal);				// release all allocated space in xtal, and set all pointers to NULL & do an InitCleanCrystalStructure
 *
 *		// The following are available, but should not be needed
 *
 *	printf("sym = %s",symmetryString(216,*sym);	// print out the symmetry string, "F-43m"
 *	m=multiplicityOfAtomType(1,&xtal);			// find the multiplicity of an atom type, not very useful
 *	l = latticeSystem(227);						// returns lattice system number, not useful
 *	SetSymOpsForSpaceGroup(SG,&(xtal->equiv));	// re-calc sym ops, done in ForceLatticeToStructure
 *	setDirectRecip(*xtal);						// recalc direct and recip lattices, also done by ForceLatticeToStructure()
 *	setAtomXYZs(&xtal);							// should be done by ForceLatticeToStructure()
 *	rho = double densityOfCrystalStructure(&xtal);	// use xtal->density which should be set by ForceLatticeToStructure()
 */

#include <stdio.h>
#ifdef _MSC_VER									/* identifies this as a Microsoft compiler */
#define _USE_MATH_DEFINES						/* added RX2011 */
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
//#include <stdarg.h>							/* needed for the variable number of arguments in dSpacing()*/
#include "lattice.h"
#ifdef _MSC_VER									/* identifies this as a Microsoft compiler */
#include "mathUtil.h"							/* added RX2011 for "NAN" definition */
#endif


int allocateEquivAtomOps(struct equivOpsStructure *sym);
void freeEquivOpsStructure(struct equivOpsStructure *eq);

int setSymLine(int SpaceGroup,  char **str);
int ParseOneSymEquation(char *expression, double *m0, double *m1, double *m2, double *b);
int make1MatrixAndVecFromSymLine(char *symItem, double mat[3][3], double vec[3]);
int positionsOfOneAtomType(struct atomTypeStructure *atomType, struct equivOpsStructure *eq, double **xyz);


char *latticeNames[]={"Triclinic","Monoclinic","Orthorhombic","Tetragonal","Trigonal","Hexagonal","Cubic"};

#define MIN(A,B)	( (A)<(B)?(A):(B) )
#define MAX(A,B)	( (A)>(B)?(A):(B) )
#define isNAN(A) ( (A) != (A) )			/* a good test for NAN  */

#ifndef N_Avagadro
#define N_Avagadro 6.02214129e23		/* Avagadro's number, from 2006 CODATA,  http://physics.nist.gov/cuu/Constants/index.html */
#endif

long allowedHKL(
struct crystalStructure *xtal,			/* crystal structure */
long	h,
long	k,
long	l)
{
	double Fr, Fi;
	long	N;
	long	allowed;

	N = Fstruct(xtal,h,k,l,&Fr,&Fi);
	allowed = N>0 ? (Fr*Fr+Fi*Fi)/(N*N) > 0.0001 : 0;	/* allowed means more than .01 electron/atom */
	return allowed;
}



size_t Fstruct(							/* sets F(hkl), returns number of atoms in cell */
struct crystalStructure *xtal,			/* crystal structure */
long	h,
long	k,
long	l,
double  *Freal,
double  *Fimag)
{
	int		SG = xtal->SpaceGroup;
	long	N = xtal->N;				/* number of actual atoms */
	char	sym[12];					/* symmetry symbol */
	char	PFIRCA[]="PFIRCA";
	int		latticeCentering;			/* lattice centering, ie face centered, body centered, ... */
	char	*p;
	struct atomStructure *atom;			/* array of atoms */
	double	fatom;						/* atomic structure factor (real), just Z for now */
	int		hexAxes;					/* flag, using hex axes (needed for Trigonal) */
	int		system;						/* = latticeSystem(SG) */
	int		m;
	double	arg;						/* angle of a complex number */

	if (SG<1 || SG>230 || N<1 ) {
		*Freal = *Fimag = NAN;
		return 0;
	}

	system = latticeSystem(SG);
	hexAxes = (fabs(M_PI/2. - xtal->alpha)+fabs(M_PI/2. - xtal->beta)+fabs(2.*M_PI/3. - xtal->gamma))<1e-5;
	symmetryString(SG,sym);				/* get sym[] */
	p = strchr(PFIRCA,sym[0]);
	latticeCentering = p ? (p-PFIRCA) : -1;
	/*	P_CENTER	F_CENTER	B_CENTER	RHOMBOHEDRAL	C_CENTER	A_CENTER */
	*Freal = *Fimag = 0.;				/* init to forbidden */
	switch (latticeCentering) {
		case F_CENTER:
			if (!ALLOW_FC(h,k,l)) return N;
			break;
		case B_CENTER:
			if (!ALLOW_BC(h,k,l)) return N;
			break;
		case C_CENTER:
			if (!ALLOW_CC(h,k,l)) return N;
			break;
		case A_CENTER:
			if (!ALLOW_AC(h,k,l)) return N;
			break;
		case RHOMBOHEDRAL:
			if (hexAxes) {
				if (!ALLOW_RHOM_HEX(h,k,l)) return N;		/* rhombohedral cell with hexagonal axes */
			}
			break;
	}
// Hexagonal has no always rules!!
//	if (system==HEXAGONAL) {
//		if (!ALLOW_HEXAGONAL(h,k,l)) return N;
//	}
//	hexAxes = (fabs(M_PI/2. - xtal->alpha)+fabs(M_PI/2. - xtal->beta)+fabs(2.*M_PI/3. - xtal->gamma))<1e-5;
//	switch (latticeSystem(SG)) {
//		case HEXAGONAL:					/* a special test for hexagonal */
//			if (!ALLOW_HEXAGONAL(h,k,l)) return N;
//			break;
//		case TRIGONAL:					/* check for trigonal system with hexagonal axes */
//			if (hexAxes && !ALLOW_RHOM_HEX(h,k,l)) return N;
//			break;
//	}

	atom = xtal->atom;
	for (m=0;m<N;m++) {
		fatom = MAX(atom[m].Zatom,1);	/* atomic structure factor, just Z for now */
		fatom *= atom[m].occ;			/* reduce by occupancy */
		arg = 2*M_PI*(h*atom[m].x+k*atom[m].y+l*atom[m].z);
		*Freal += fatom*cos(arg);		/* 	SUM{ exp[ 2PI*i * (hx+ky+lz) ] } */
		*Fimag += fatom*sin(arg);
	}


/*	if (system==HEXAGONAL || (hexAxes && system==TRIGONAL)) { */
	if (hexAxes && system==TRIGONAL) {					/* hexagonal is simple, it is trigonal that everyone lies about */
		double Fr, Fi;
		double rr, ii;
		arg = 2*M_PI*((h+2.0*k)/3.0 + l*0.5);			/* heagnal has atoms at (0,0,0) and (1/3, 2/3, 1/2) */
		Fr=1. + cos(arg);
		Fi=sin(arg);
		rr = *Freal;
		ii = *Fimag;
		*Freal = rr*Fr - ii*Fi;
		*Fimag = rr*Fi + ii*Fr;
		//  for hexagonal:
		//	h+2k=3n,	l=even;		F = 4*f			1
		//	h+2k=3n±1,	l=odd;		F = sqrt(3)*f   sqrt(3)/4
		//	h+2k=3n±1,	l=even;		F = f			1/4
		//	h+2k=3n,	l=odd; 		F = 0			0
	}

	*Freal = fabs(*Freal)<1e-8 ? 0 : *Freal;
	*Fimag = fabs(*Fimag)<1e-8 ? 0 : *Fimag;
	return N;
}




/*	do not use, use symmetryString() instead
 */
char *symmetryString(
int		SG,								/* Space Group number [1,230] */
char	*sym)
{
	char	*symms[]={					/* there are 230 items in this array, longest string is 9 characters long, so use char s[10] */
				"P1", "P-1", "P2:b", "P21:b", "C2:b1", "Pm:b", "Pc:b1", \
				"Cm:b1", "Cc:b1", "P2/m:b", "P21/m:b", "C2/m:b1", "P2/c:b1", \
				"P21/c:b1", "C2/c:b1", "P222", "P2221", "P21212", "P212121", \
				"C2221", "C222", "F222", "I222", "I212121", "Pmm2", "Pmc21", \
				"Pcc2", "Pma2", "Pca21", "Pnc2", "Pmn21", "Pba2", "Pna21", \
				"Pnn2", "Cmm2", "Cmc21", "Ccc2", "Amm2", "Abm2", "Ama2", "Aba2", \
				"Fmm2", "Fdd2", "Imm2", "Iba2", "Ima2", "Pmmm", "Pnnn:1", "Pccm", \
				"Pban:1", "Pmma", "Pnna", "Pmna", "Pcca", "Pbam", "Pccn", "Pbcm", \
				"Pnnm", "Pmmn:1", "Pbcn", "Pbca", "Pnma", "Cmcm", "Cmca", "Cmmm", \
				"Cccm", "Cmma", "Ccca:1", "Fmmm", "Fddd:1", "Immm", "Ibam", "Ibca", \
				"Imma", "P4", "P41", "P42", "P43", "I4", "I41", "P-4", "I-4", \
				"P4/m", "P42/m", "P4/n:1", "P42/n:1", "I4/m", "I41/a:1", "P422", \
				"P4212", "P4122", "P41212", "P4222", "P42212", "P4322", "P43212", \
				"I422", "I4122", "P4mm", "P4bm", "P42cm", "P42nm", "P4cc", \
				"P4nc", "P42mc", "P42bc", "I4mm", "I4cm", "I41md", "I41cd", \
				"P-42m", "P-42c", "P-421m", "P-421c", "P-4m2", "P-4c2", "P-4b2", \
				"P-4n2", "I-4m2", "I-4c2", "I-42m", "I-42d", "P4/mmm", "P4/mcc", \
				"P4/nbm:1", "P4/nnc:1", "P4/mbm", "P4/mnc", "P4/nmm:1", "P4/ncc:1", \
				"P42/mmc", "P42/mcm", "P42/nbc:1", "P42/nnm:1", "P42/mbc", "P42/mnm", \
				"P42/nmc:1", "P42/ncm:1", "I4/mmm", "I4/mcm", "I41/amd:1", \
				"I41/acd:1", "P3", "P31", "P32", "R3:H", "P-3", "R-3:H", "P312", \
				"P321", "P3112", "P3121", "P3212", "P3221", "R32:H", "P3m1", "P31m", \
				"P3c1", "P31c", "R3m:H", "R3c:H", "P-31m", "P-31c", "P-3m1", "P-3c1", \
				"R-3m:H", "R-3c:H", "P6", "P61", "P65", "P62", "P64", "P63", "P-6", \
				"P6/m", "P63/m", "P622", "P6122", "P6522", "P6222", "P6422", "P6322", \
				"P6mm", "P6cc", "P63cm", "P63mc", "P-6m2", "P-6c2", "P-62m", "P-62c", \
				"P6/mmm", "P6/mcc", "P63/mcm", "P63/mmc", "P23", "F23", "I23", "P213", \
				"I213", "Pm-3", "Pn-3:1", "Fm-3", "Fd-3:1", "Im-3", "Pa-3", "Ia-3", \
				"P432", "P4232", "F432", "F4132", "I432", "P4332", "P4132", "I4132", \
				"P-43m", "F-43m", "I-43m", "P-43n", "F-43c", "I-43d", "Pm-3m", \
				"Pn-3n:1", "Pm-3n", "Pn-3m:1", "Fm-3m", "Fm-3c", "Fd-3m:1", "Fd-3c:1", \
				"Im-3m", "Ia-3d"
				};

	if (SG<1 || SG>230) {				/* invalid Space Group */
		sym[0] = '\0';
		return sym;
	}
	strcpy(sym,symms[SG-1]);
	return sym;
}


int latticeSystem(						/* returns number of Bravais Lattice for a Space Group */
int		SG)								/* Space Group number [1,230] */
{
	if (SG>230) return -1;				/* invalid */
	else if (SG>=195) return CUBIC;
	else if(SG>=168) return HEXAGONAL;
	else if(SG>=143) return TRIGONAL;	/* Trigonal, (using the hexagonal cell axes) */
	else if(SG>=75) return TETRAGONAL;
	else if(SG>=16) return ORTHORHOMBIC;
	else if(SG>=3) return MONOCLINIC;
	else if (SG>0) return TRICLINIC;
	return -1;							/* invalid */
}



/* changes hkl[3] to the lowest order hkl, ignores whether a reflection is allowed, just removes common factors */
void lowestOrderHKL(
long	hkl[3])
{
	int		maxDiv;						/* max possible divisor */
	int		h,k,l;
	int		i;
	h = hkl[0];
	k = hkl[1];
	l = hkl[2];
	maxDiv = MAX(abs(h),abs(k));
	maxDiv = MAX(maxDiv,abs(l));
	for (i=maxDiv;i>=2;i--) {			/* check all divisorts in range [2, maxDiv] */
		if (h%i || k%i || l%i) continue;/* i is not a factor of h, k, and l */
		h /= i;
		k /= i;
		l /= i;
	}
	hkl[0] = h;
	hkl[1] = k;
	hkl[2] = l;
}


/* changes hkl[3] to the lowest order allowed hkl (ie for FCC, 0,0,12 -> 002 not 001 */
void lowestAllowedHKL(
long	hkl[3],
struct crystalStructure *xtal)			/* crystal structure */
{
	long	hkl0[3];
	int		i;

	hkl0[0] = hkl[0];					/* save the starting point */
	hkl0[1] = hkl[1];
	hkl0[2] = hkl[2];
	lowestOrderHKL(hkl0);				/* all common factors have been removed */

	for (i=1;i<16;i++){					/* never need more than 16 to reach an allowed reflection*/
		hkl[0] = i*hkl0[0];				/* try each of the multiples to reach an allowed reflection */
		hkl[1] = i*hkl0[1];
		hkl[2] = i*hkl0[2];
		if (allowedHKL(xtal, hkl[0],hkl[1],hkl[2])) return;
	}
	return;
}









double dSpacing(						/* returns d-spacing for the hkl (nm), uses only T as an optional argument */
struct crystalStructure *xtal,			/* crystal structure */
double	h,
double	k,
double	l,
double	T)								/* Temperature (C), Temperature is referenced to ROOM_TEMP == 22.5 */
{
	double	qx,qy,qz;					/* qvec = {qx,qy,qz} */
	double	d;							/* d-spacing */

//	qx = h*(xtal->recip[0][0]) + k*(xtal->recip[1][0]) + l*(xtal->recip[2][0]);
//	qy = h*(xtal->recip[0][1]) + k*(xtal->recip[1][1]) + l*(xtal->recip[2][1]);
//	qz = h*(xtal->recip[0][2]) + k*(xtal->recip[1][2]) + l*(xtal->recip[2][2]);

	qx = h*(xtal->recip[0][0]) + k*(xtal->recip[0][1]) + l*(xtal->recip[0][2]);
	qy = h*(xtal->recip[1][0]) + k*(xtal->recip[1][1]) + l*(xtal->recip[1][2]);
	qz = h*(xtal->recip[2][0]) + k*(xtal->recip[2][1]) + l*(xtal->recip[2][2]);

	d = 2.*M_PI/sqrt(qx*qx + qy*qy + qz*qz);

	if (fabs(xtal->alphaT)<0.1 && T > -273.15) {
		d = d * ( 1 + (xtal->alphaT)*(T-ROOM_TEMP) );	/* apply temperature correction */
	}
	return d;
}










/* using the atomTypes, find all of the real atom positions, be sure to call SetSymOpsForSpaceGroup(SG,xtal->equiv) before calling this */
/* if no atoms were set in xtal->atomType, then it uses a single default atom with Z=0 at (0,0,0) */
int setAtomXYZs(
struct crystalStructure *xtal)			/* lattice */
{
	size_t	Ntype;						/* number of atom types */
	size_t	Neq;						/* number of equivalent atom operations */
	size_t	j;							/* number of actual atoms of one type */
	size_t	N;							/* total number of real atoms */
	struct atomStructure *atom;
	struct atomTypeStructure *type;
	struct atomTypeStructure defType;	/* a default atom type */
	double	**xyz=NULL;					/* temp space for all of the positions of one atom type */
	int		SG;							/* Space Group number [1-230] */
	size_t	i, m;

	SG = xtal->SpaceGroup;
	if (xtal->equiv.SpaceGroup != SG || SG<1 || SG>230 || xtal->equiv.N<1 || !(xtal->equiv.equivXYZM) || !(xtal->equiv.equivXYZB)) {
		fprintf(stderr,"the equivOpsStructure has not been set in setAtomXYZs(SG=%d, eqiv.SG=%d, eqiv.N=%d)\n",SG,xtal->equiv.SpaceGroup,xtal->equiv.N);
		exit(1);
	}
	Neq = xtal->equiv.N;				/* number of equivalent atom operations, the largest possible number of atoms of each type */

	if (xtal->Ntype > 0) {
		Ntype = xtal->Ntype;
		type = xtal->atomType;
	}
	else {								/* use the default atom type, the atoms were not given */
		Ntype = 1;						/* just one atom type in the default */
		defType.name[0] = '\0';			/* set defType so I can use it */
		defType.Zatom = 0;
		defType.x = defType.y = defType.z = 0;
		defType.occ = 1;
		defType.Debye = 0.;
		type = &defType;				/* type now points to the default, an array of length one */
	}

	xyz = calloc(Neq,sizeof(double*));
	if (!(xyz)) { fprintf(stderr,"unable to allocate xyz space for %ld atoms in setAtomXYZs\n",Neq); exit(1); }
	for (i=0;i<Neq;i++) {
		xyz[i] = calloc(3,sizeof(double));
		if (!(xyz[i])) { fprintf(stderr,"unable to allocate space for xyz[%ld] (one atom position) setAtomXYZs\n",i); exit(1); }
	}
	xtal->atom = calloc(Ntype*Neq,sizeof(*atom));	/* allocate for the max possible number of atoms */
	if (!(xtal->atom)) { fprintf(stderr,"unable to allocate space for %ld atoms in setAtomXYZs()\n",Ntype*Neq); exit(1); }
	atom = xtal->atom;						/* atom is just a local copy of xtal->atom, for convienence */

	for (m=N=0; m<Ntype; m++) {				/* loop over each atom type */
		j = positionsOfOneAtomType(&(type)[m],&(xtal->equiv),xyz);	/* fill xyz for this type, j is num of non-equiv atoms found */
		for (i=0;i<j;i++) {
			if (xtal->useUnconventional) {	/* rotate xyz[i][] by Unconventional */
				double v0,v1,v2;			/* temp value of xyz[i][] */
				v0 = xyz[i][0];  v1 = xyz[i][1];  v2 = xyz[i][2];
				xyz[i][0] = xtal->Unconventional[0][0]*v0 + xtal->Unconventional[0][1]*v1 + xtal->Unconventional[0][2]*v2;	/* xyz[i] = Unconventional x v */
				xyz[i][1] = xtal->Unconventional[1][0]*v0 + xtal->Unconventional[1][1]*v1 + xtal->Unconventional[1][2]*v2;
				xyz[i][2] = xtal->Unconventional[2][0]*v0 + xtal->Unconventional[2][1]*v1 + xtal->Unconventional[2][2]*v2;
			}
			atom[i+N].type	= m;			/* fill in these j real atom positions */
			atom[i+N].Zatom	= type[m].Zatom;
			atom[i+N].x		= xyz[i][0];
			atom[i+N].y		= xyz[i][1];
			atom[i+N].z		= xyz[i][2];
			atom[i+N].occ	= type[m].occ;
			atom[i+N].Debye	= type[m].Debye;
		}
		N += j;
	}
	atom = realloc(atom,N*sizeof(*atom));	/* remove extra space (due to duplicate atom positions) */
	if (!(atom)) { fprintf(stderr,"unable to re-allocate space for %ld atoms in setAtomXYZs()\n",N); exit(1); }
	xtal->atom = atom;						/* make sure that xtal->atom stays in sych with atom */
	xtal->N = N;

#if (DEBUG>2)
	fprintf (stdout, "in setAtomXYZs found %ld atoms at locations:\n",N);
	for (i=0;i<N;i++) fprintf(stdout,"xyz[%3lu] = (%.3f   %.3f   %.3f)\n",i,atom[i].x,atom[i].y,atom[i].z);
#endif

	for (i=0;i<Neq;i++) CHECK_FREE(xyz[i])
	CHECK_FREE(xyz)
	return 0;
}





int positionsOfOneAtomType(				/* returns xyz[][3], the positions of all non-equivalent atoms of the specified type */
struct atomTypeStructure *atomType,		/* the type of atom */
struct equivOpsStructure *eq,			/* the operations to apply to that type */
double	**xyz)							/* returned, list of all equiv posiitions for this atom in fractional coords, xyz[][3] */
{
	double	***mats,**bvecs;			/* local name, a convienence */
	size_t	Neq;						/* number of equivalent operations */
	double	r[3];
	double	in[3];						/* base postion, of the atom type */
	int		dup;						/* flag for duplicates */
	size_t	N;							/* counter, number of distinct real atoms */
	size_t	i, m;

	if ((eq->SpaceGroup)<1 || (eq->SpaceGroup)>230 || (eq->N)<1 || !(eq->equivXYZM) || !(eq->equivXYZB)) {
		fprintf(stderr,"the equivOpsStructure is wrong in positionsOfOneAtomType(SG=%d, eqiv.N=%d)\n",eq->SpaceGroup,eq->N);
		exit(1);
	}
	Neq = eq->N;
	mats = eq->equivXYZM;
	bvecs = eq->equivXYZB;
	in[0] = atomType->x;				/* base position for this atom type */
	in[1] = atomType->y;
	in[2] = atomType->z;

	for (m=N=0; m<Neq; m++) {
		r[0] = mats[m][0][0]*in[0] + mats[m][0][1]*in[1] + mats[m][0][2]*in[2]  +  bvecs[m][0];	/* r = mat x in + bv */
		r[1] = mats[m][1][0]*in[0] + mats[m][1][1]*in[1] + mats[m][1][2]*in[2]  +  bvecs[m][1];
		r[2] = mats[m][2][0]*in[0] + mats[m][2][1]*in[1] + mats[m][2][2]*in[2]  +  bvecs[m][2];

		/* r += fabs(floor(r))			// translate back into unit cell, so value in [0,1) */
		/* r = mod(r,1) */
		r[0] += fabs(floor(r[0]));		/* translate back into unit cell, so value in [0,1) */
		r[1] += fabs(floor(r[1]));
		r[2] += fabs(floor(r[2]));
		r[0] -= floor(r[0]);
		r[1] -= floor(r[1]);
		r[2] -= floor(r[2]);
#ifdef __GNUC__
#warning "the tolerance of 1e-3, is a bit arbitrary, it should use lattice constant to set it to ~.2Angstrom"
#endif
		for (i=dup=0; i<N && !dup; i++) dup=(fabs(xyz[i][0]-r[0])+fabs(xyz[i][1]-r[1])+fabs(xyz[i][2]-r[2])<1e-3);	/* reject duplicate positions */
		if (!dup) {						/* not a duplicate, so add to the list of positions */
			xyz[N][0]=r[0];		xyz[N][1]=r[1];		xyz[N][2]=r[2];
			N++;
		}
	}
	return N;
}



int ForceLatticeToStructure(
struct crystalStructure *xtal)			/* lattice */
{
	int SG = xtal->SpaceGroup;			/* local value for convienence */
	int		usingHexAxes;
	double a, b, g;

	if (SG<1 || SG>230) return 1;		/* invalid SpaceGroup, it must be in range [1,230] */
	/*	Cubic			[195,230]	//	a
	 *	Hexagonal		[168,194]	//	a,c
	 *	Trigonal		[143,167]	//	a,alpha
	 *	Tetragonal		[75,142]	//	a,c
	 *	Orthorhombic	[16,74]		//	a,b,c
	 *	Monoclinic		[3,15]		//	a,b,c,gamma
	 *	Triclinic		[1,2]		//	a,b,c,alpha,beta,gamma */

	usingHexAxes = (fabs(M_PI/2.-xtal->alpha)+fabs(M_PI/2.-xtal->beta)+fabs(2.*M_PI/3.-xtal->gamma))<1e-9;


	if (SG>=195) {						/* Cubic */
		xtal->b = xtal->c = xtal->a;
		xtal->alpha = xtal->beta = xtal->gamma = M_PI/2.;
	}
	else if (SG>=168) {					/* Hexagonal */
		xtal->b = xtal->a;
		xtal->alpha = xtal->beta = M_PI/2.;
		xtal->gamma = 2*M_PI/3.;
	}
	else if (SG>=143) {					/* Trigonal, use rhomohedral cell, unless obviously the hexagonal cell */
		if (usingHexAxes) {				/* use hexagonal */
			xtal->b = xtal->a;
			xtal->alpha = xtal->beta = M_PI/2.;
			xtal->gamma = 2*M_PI/3.;
		}
		else {							/* use rhombohedral */
			xtal->b = xtal->a;
			xtal->c = xtal->a;
			xtal->beta = xtal->alpha;
			xtal->gamma = xtal->alpha;
		}
	}
	else if (SG>=75) {					/* Tetragonal */
		xtal->b = xtal->a;
		xtal->alpha = xtal->beta = xtal->gamma = M_PI/2.;
	}
	else if (SG>=16) {					/* Orthorhombic */
		xtal->alpha = xtal->beta = xtal->gamma = M_PI/2.;
	}
	else if (SG>=3) {					/* Monoclinic */
		xtal->alpha = xtal->gamma = M_PI/2.;
	}
/*	else								// Triclinic, no constraints */

	if (xtal->a<=0 || xtal->b<=0 || xtal->c<=0 || isNAN(xtal->a+xtal->b+xtal->c)) {	/* check for valid a,b,c */
		fprintf(stderr,"invalid, (a,b,c) = (%g,%g,%g)\n",xtal->a,xtal->b,xtal->c);
		return 1;
	}
	a=xtal->alpha; b=xtal->beta; g=xtal->gamma;
	if ( !(a>0 && a<M_PI) || !(b>0 && b<M_PI) || !(g>0 && g<M_PI) ) {		/* check for valid angles */
		fprintf(stderr,"invalid, (alpha,beta,gamma) = (%g,%g,%g)\n",a,b,g);
		return 1;
	}
	UpdateInternalsOfCrystalStructure(xtal);
	return 0;
}


int UpdateInternalsOfCrystalStructure(				/* update all of the parts of the crystalStructure */
struct crystalStructure *xtal)
{
	setDirectRecip(xtal);							/* update Vc, direct and recip, (NOT density or atom positions) */
	SetSymOpsForSpaceGroup(xtal->SpaceGroup,&(xtal->equiv));	/* update eqivalent atom positions, does NOT generate the real atom positions */
	setAtomXYZs(xtal);								/* lattice */
	xtal->density = densityOfCrystalStructure(xtal);/* this needs Vc and all atom positions */
	return 0;
}


/******************************************************************************************
 * for a lattice structure, set the array recip with the reciprocal lattice based on the 
 * six constants a,b,c, alpha, beta, gamma.
 * calculation comes from:
 * International Tables B, pg 360, sect. 3.3.1 (columns of vector M)
 *  http://journals.iucr.org/iucr-top/comm/cteach/pamphlets/4/node8.html#SECTION00033000000000000000
 ******************************************************************************************/
int setDirectRecip(						/* set direct and recip lattice vectors from a,b,c,..., also calculates Vc & density */
struct crystalStructure *xtal)			/* lattice */
{
	double	a=xtal->a, b=xtal->b, c=xtal->c;
	double	sa, ca, cb, cg;
	double	phi;							/* = Vc/(a*b*c) */
	double	a0,a1,a2, b0,b1,b2, c0,c1,c2;	/* components of the direct lattice vectors */
	double	pv;								/* used for scaling */

	sa = sin(xtal->alpha);
	ca = cos(xtal->alpha);
	cb = cos(xtal->beta);
	cg = cos(xtal->gamma);
	ca = fabs(ca)<1e-14 ? 0.0 : ca;			/* remove almost zeros near 90(degree) */
	cb = fabs(cb)<1e-14 ? 0.0 : cb;
	cg = fabs(cg)<1e-14 ? 0.0 : cg;
	phi = sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
	xtal->Vc = a*b*c * phi;					/* volume of unit cell */

	pv = (2*M_PI) / (xtal->Vc);				/* used for scaling */
	a0=a*phi/sa;	a1=a*(cg-ca*cb)/sa; 	a2=a*cb;
	b0=0;			b1=b*sa;				b2=b*ca;
	c0=0;			c1=0;					c2=c;

	(xtal->direct)[0][0]=a0;	(xtal->direct)[0][1]=b0;	(xtal->direct)[0][2]=c0;	/* lattice vectors are column vectors, bad for C, but thats it */
	(xtal->direct)[1][0]=a1;	(xtal->direct)[1][1]=b1;	(xtal->direct)[1][2]=c1;
	(xtal->direct)[2][0]=a2;	(xtal->direct)[2][1]=b2;	(xtal->direct)[2][2]=c2;

	(xtal->recip)[0][0]=(b1*c2-b2*c1)*pv;	(xtal->recip)[0][1]=(c1*a2-c2*a1)*pv;	(xtal->recip)[0][2]=(a1*b2-a2*b1)*pv;	/* (b x c)*2π/Vc */
	(xtal->recip)[1][0]=(b2*c0-b0*c2)*pv;	(xtal->recip)[1][1]=(c2*a0-c0*a2)*pv;	(xtal->recip)[1][2]=(a2*b0-a0*b2)*pv;	/* (c x a)*2π/Vc */
	(xtal->recip)[2][0]=(b0*c1-b1*c0)*pv;	(xtal->recip)[2][1]=(c0*a1-c1*a0)*pv;	(xtal->recip)[2][2]=(a0*b1-a1*b0)*pv;	/* (a x b)*2π/Vc */

//	(xtal->direct)[0][0]=a0;	(xtal->direct)[0][1]=a1;	(xtal->direct)[0][2]=a2;	/* lattice vectors are row vectors !!, nice for C, but WRONG */
//	(xtal->direct)[1][0]=b0;	(xtal->direct)[1][1]=b1;	(xtal->direct)[1][2]=b2;
//	(xtal->direct)[2][0]=c0;	(xtal->direct)[2][1]=c1;	(xtal->direct)[2][2]=c2;
//
//	(xtal->recip)[0][0]=(b1*c2-b2*c1)*pv;	(xtal->recip)[0][1]=(b2*c0-b0*c2)*pv;	(xtal->recip)[0][2]=(b0*c1-b1*c0)*pv;	/* (b x c)*2π/Vc */
//	(xtal->recip)[1][0]=(c1*a2-c2*a1)*pv;	(xtal->recip)[1][1]=(c2*a0-c0*a2)*pv;	(xtal->recip)[1][2]=(c0*a1-c1*a0)*pv;	/* (c x a)*2π/Vc */
//	(xtal->recip)[2][0]=(a1*b2-a2*b1)*pv;	(xtal->recip)[2][1]=(a2*b0-a0*b2)*pv;	(xtal->recip)[2][2]=(a0*b1-a1*b0)*pv;	/* (a x b)*2π/Vc */

	return 0;
}



double densityOfCrystalStructure(		/* returns the density (g/cm^3), does NOT set density in xtal */
struct crystalStructure *xtal)			/* lattice */
{
	double	density;
	double amuList[] = {				/* atomic mass of all of the elements */
		1.00794,4.002602,6.941,9.012182,10.811,12.0107,14.0067,15.9994,18.9984032,20.1797,22.98977,24.305,
		26.981538,28.0855,30.973761,32.065,35.453,39.948,39.0983,40.078,44.95591,47.867,50.9415,51.9961,
		54.938049,55.845,58.9332,58.6934,63.546,65.409,69.723,72.64,74.9216,78.96,79.904,83.798,85.4678,87.62,
		88.90585,91.224,92.90638,95.94,98,101.07,102.9055,106.42,107.8682,112.411,114.818,118.71,121.76,
		127.6,126.90447,131.293,132.90545,137.327,138.9055,140.116,140.90765,144.24,145,150.36,151.964,157.25,
		158.92534,162.5,164.93032,167.259,168.93421,173.04,174.967,178.49,180.9479,183.84,186.207,190.23,
		192.217,195.078,196.96655,200.59,204.3833,207.2,208.98038,209,210,222,223,226,227,232.0381,231.03588,
		238.02891,237,244,243,247,247,251,252,257,258,259,262,261,262,266,264,277,268
	};
	double	amu;						/* atomic mass of all atoms in cell */
	double	N = xtal->N;				/* number of atoms */
	long	m;
	double	Vc = xtal->Vc;
	int		iZ;
	double	u;							/* conversion to cm */

	u = 1.e2 / xtal->lengthUnits;
	Vc *= u*u*u;						/* convert Vc to cm^3 */
	if (N < 1 || !(Vc>0)) return NAN;
	for (m=0,amu=0.; m<N; m++) {
		iZ = (xtal->atom)[m].Zatom - 1;
		if (iZ<0) return NAN;
		else amu += amuList[iZ] * (xtal->atom)[m].occ;
	}
	density = (amu/N_Avagadro)/Vc;		/* (grams / cm^3) */
	return density;
}


/* copy one crystalStructure to another */
void copyCrystalStructure(
struct crystalStructure *destLat,		/* destination lattice */
struct crystalStructure *sourceLat)		/* source lattice */
{
	size_t	Ntype;						/* number of atom types */
	size_t	N;							/* number of actual atoms */
	int		SG;							/* Space Group number [1,230] */
	int		Neq;						/* number of equivalent atoms, in symmetry equivalents */
	size_t	i;

	strncpy(destLat->desc,sourceLat->desc,255);
	destLat->desc[255] = '\0';			/* ensure terminating NULL */
	destLat->a			= sourceLat->a;
	destLat->b			= sourceLat->b;
	destLat->c			= sourceLat->c;
	destLat->lengthUnits= sourceLat->lengthUnits;
	destLat->alpha		= sourceLat->alpha;
	destLat->beta		= sourceLat->beta;
	destLat->gamma		= sourceLat->gamma;
	destLat->SpaceGroup = sourceLat->SpaceGroup;

	destLat->direct[0][0] = sourceLat->direct[0][0];
	destLat->direct[0][1] = sourceLat->direct[0][1];
	destLat->direct[0][2] = sourceLat->direct[0][2];

	destLat->direct[1][0] = sourceLat->direct[1][0];
	destLat->direct[1][1] = sourceLat->direct[1][1];
	destLat->direct[1][2] = sourceLat->direct[1][2];

	destLat->direct[2][0] = sourceLat->direct[2][0];
	destLat->direct[2][1] = sourceLat->direct[2][1];
	destLat->direct[2][2] = sourceLat->direct[2][2];


	destLat->recip[0][0] = sourceLat->recip[0][0];
	destLat->recip[0][1] = sourceLat->recip[0][1];
	destLat->recip[0][2] = sourceLat->recip[0][2];

	destLat->recip[1][0] = sourceLat->recip[1][0];
	destLat->recip[1][1] = sourceLat->recip[1][1];
	destLat->recip[1][2] = sourceLat->recip[1][2];

	destLat->recip[2][0] = sourceLat->recip[2][0];
	destLat->recip[2][1] = sourceLat->recip[2][1];
	destLat->recip[2][2] = sourceLat->recip[2][2];

	destLat->Vc			 = sourceLat->Vc;
	destLat->density	 = sourceLat->density;
	destLat->alphaT		 = sourceLat->alphaT;

	destLat->useUnconventional	  = sourceLat->useUnconventional;

	destLat->Unconventional[0][0] = sourceLat->Unconventional[0][0];
	destLat->Unconventional[0][1] = sourceLat->Unconventional[0][1];
	destLat->Unconventional[0][2] = sourceLat->Unconventional[0][2];

	destLat->Unconventional[1][0] = sourceLat->Unconventional[1][0];
	destLat->Unconventional[1][1] = sourceLat->Unconventional[1][1];
	destLat->Unconventional[1][2] = sourceLat->Unconventional[1][2];

	destLat->Unconventional[2][0] = sourceLat->Unconventional[2][0];
	destLat->Unconventional[2][1] = sourceLat->Unconventional[2][1];
	destLat->Unconventional[2][2] = sourceLat->Unconventional[2][2];

	destLat->Ntype		 = sourceLat->Ntype;
	destLat->N			 = sourceLat->N;
	Ntype = sourceLat->Ntype;
	N = sourceLat->N;
	if (destLat->Ntype != Ntype && destLat->atomType != NULL) {	/* free */
		CHECK_FREE(destLat->atomType)
		destLat->Ntype = 0;
	}
	if (destLat->N != N && destLat->atom != NULL) {				/* free */
		CHECK_FREE(destLat->atom)
		destLat->N =0;
	}

	if (Ntype >= 1 && sourceLat->atomType) {					/* copy all of the atom types */
		struct atomTypeStructure *atomS = sourceLat->atomType;	/* pointer to source atoms */
		struct atomTypeStructure *atomD=NULL;					/* pointer to destination atoms */
		if (destLat->atomType == NULL) {						/* allocate space for atoms */
			destLat->atomType = calloc(Ntype,sizeof(*atomS));
			if (!(destLat->atomType)) { fprintf(stderr,"unable to allocate space for %ld atom types in copyCrystalStructure\n",Ntype); exit(1); }
		}
		atomD = destLat->atomType;	/* pointer to destination atoms */
		for (i=0;i<Ntype;i++) {
			strncpy(atomD[i].name,atomS[i].name,59);
			atomD[i].name[59] = '\0';
			atomD[i].Zatom	= atomS[i].Zatom;
			atomD[i].x		= atomS[i].x;
			atomD[i].y		= atomS[i].y;
			atomD[i].z		= atomS[i].z;
			atomD[i].occ	= atomS[i].occ;
			atomD[i].Debye	= atomS[i].Debye;
		}
	}

	if (N >= 1 && sourceLat->atom) {					/* copy all of the real atom positions */
		struct atomStructure *atomS = sourceLat->atom;	/* pointer to source atoms */
		struct atomStructure *atomD=NULL;				/* pointer to destination atoms */
		if (destLat->atom == NULL) {					/* allocate space for atoms */
			destLat->atom = calloc(N,sizeof(*atomS));
			if (!(destLat->atom)) { fprintf(stderr,"unable to allocate space for %ld atoms in copyCrystalStructure\n",N); exit(1); }
		}
		atomD = destLat->atom;							/* pointer to destination atoms */
		for (i=0;i<N;i++) {
			atomD[i].type	= atomS[i].type;
			atomD[i].Zatom	= atomS[i].Zatom;
			atomD[i].x		= atomS[i].x;
			atomD[i].y		= atomS[i].y;
			atomD[i].z		= atomS[i].z;
			atomD[i].occ	= atomS[i].occ;
			atomD[i].Debye	= atomS[i].Debye;
		}
	}

	freeEquivOpsStructure(&(destLat->equiv));
	SG = destLat->equiv.SpaceGroup = sourceLat->equiv.SpaceGroup;
	Neq = destLat->equiv.N = sourceLat->equiv.N;
	if (Neq > 0 && SG >= 1 && SG <= 230) {				/* copy the equivalent atome matricies */
		size_t	m,j;
		double	***matsS, ***matsD;						/* these are for convienence */
		double	**bvecS,  **bvecD;
		allocateEquivAtomOps(&(destLat->equiv));
		matsS = sourceLat->equiv.equivXYZM;				/* mats and bvec are for convienence */
		bvecS = sourceLat->equiv.equivXYZB;
		matsD = destLat->equiv.equivXYZM;
		bvecD = destLat->equiv.equivXYZB;
		for (m=0; (int)m<Neq; m++) {
			for (j=0; j<3; j++) {
				bvecD[m][j] = bvecS[m][j];
				for (i=0; i<3; i++) matsD[m][j][i] = matsS[m][j][i];
			}
		}
	}

}


void InitCleanCrystalStructure(			/* ensure that all values are set to clean values, do NOT use this after any space has been allocated */
struct crystalStructure *lat)			/* lattice and all of the crystal structure values */
{
	lat->desc[0] = '\0';
	lat->a = lat->b = lat->c = 0.;
	lat->alpha = lat->beta = lat->gamma = 0.;
	lat->SpaceGroup = 0;
	lat->lengthUnits = 1.e9;			/* default is nm */

	(lat->direct)[1][0] = (lat->direct)[0][1] = (lat->direct)[0][2] = 0.;
	(lat->direct)[2][0] = (lat->direct)[1][1] = (lat->direct)[1][2] = 0.;
	(lat->direct)[2][0] = (lat->direct)[2][1] = (lat->direct)[2][2] = 0.;

	(lat->recip)[1][0] = (lat->recip)[0][1] = (lat->recip)[0][2] = 0.;
	(lat->recip)[2][0] = (lat->recip)[1][1] = (lat->recip)[1][2] = 0.;
	(lat->recip)[2][0] = (lat->recip)[2][1] = (lat->recip)[2][2] = 0.;

	lat->Vc = lat->density = lat->alphaT = 0.;

	lat->Ntype = 0;
	lat->atomType = NULL;				/* cannot do anything until I know Ntype */

	lat->N = 0;
	lat->atom = NULL;

	lat->equiv.SpaceGroup = 0;
	lat->equiv.N = 0;
	lat->equiv.equivXYZM = NULL;
	lat->equiv.equivXYZB = NULL;

	lat->useUnconventional = 0;
	(lat->Unconventional)[0][1] = (lat->Unconventional)[0][2] = 0.;
	(lat->Unconventional)[2][0] = (lat->Unconventional)[1][2] = 0.;
	(lat->Unconventional)[2][0] = (lat->Unconventional)[2][1] = 0.;
	(lat->Unconventional)[0][0] = (lat->Unconventional)[1][1] = (lat->Unconventional)[2][2] = 1.;
}

void freeCrystalStructure(				/* frees everything in a crystalStructure unless value is NULL */
struct crystalStructure *lat)			/* lattice and all of the crystal structure values */
{
	if (lat->Ntype > 0) CHECK_FREE(lat->atomType)
	if (lat->N > 0) CHECK_FREE(lat->atom)
	if (lat->equiv.N > 0) freeEquivOpsStructure(&(lat->equiv));
	InitCleanCrystalStructure(lat);	/* set all fixed values to the defaults */
}



int allocateEquivAtomOps(
struct equivOpsStructure *eq)			/* allocate all of the space here */
{
	double	***mats;					/* these are for convienence */
	double	**bvec;
	size_t	i, j;
	size_t	N;							/* number of equivalent atom positions, local copy */
	N = eq->N;
	if (N<1) return 0;

	if (!(eq->equivXYZM = calloc(N,sizeof(double**)))) goto errPath;
	if (!(eq->equivXYZB = calloc(N,sizeof(double*)))) goto errPath;
	mats = eq->equivXYZM;				/* mats and bvec are for convienence */
	bvec = eq->equivXYZB;
	for (i=0;i<N;i++) {
		if (!(mats[i] = calloc(3,sizeof(double*)))) goto errPath;
		if (!(bvec[i] = calloc(3,sizeof(double)))) goto errPath;
		for (j=0;j<3;j++) {
			if (!(mats[i][j] = calloc(3,sizeof(double)))) goto errPath;
		}
	}
	return 0;							/* success */

	errPath:
		fprintf(stderr,"unable to allocate space for equivXYZM and equivXYZB in allocateEquivAtomOps()\n");
		exit(1);
	return 1;
}


void freeEquivOpsStructure(
struct equivOpsStructure *eq)			/* equivalent atom operations */
{
	double	***mats;					/* these are for convienence */
	double	**bvec;
	long	i, j, N;
	mats = eq->equivXYZM;				/* mats and bvec are for convienence */
	bvec = eq->equivXYZB;
	N = eq->N;

	for (i=0;i<N;i++) {
		for (j=0;j<3;j++) CHECK_FREE(mats[i][j])
		CHECK_FREE(mats[i])
		CHECK_FREE(bvec[i])
	}
	CHECK_FREE(eq->equivXYZM)
	CHECK_FREE(eq->equivXYZB)

	eq->SpaceGroup = eq->N = 0;
}




void print_crystalStructure(			/* prints out the value in a crystalStructure */
FILE *f,
struct crystalStructure *xtal)			/* lattice and all of the crystal structure values */
{
	size_t	Ntype=xtal->Ntype;			/* number of atom types */
	size_t	N=xtal->N;					/* total number of real non-equivalent atoms */
	size_t	i;
	char	sym[12];					/* symmetry symbol */
	int		showOcc;					/* flag, used to control printing of occupancy */
	int		showDebye;					/* flag, used to control printing of Debye */
	int		showType;					/* flag, used to control printing of atom type */
	double	u;							/* converts current units to nm, for displaly */

	for (i=showOcc=showDebye=0;i<Ntype;i++) {					/* only show occupancy if all are not 1, or Debye all are not 0 */
		showOcc = showOcc || (xtal->atomType[i].occ != 1.);
		showDebye = showDebye || (xtal->atomType[i].Debye > 0.);
	}
	showType = Ntype>1;
	u = xtal->lengthUnits/1.e9;

	if (strlen(xtal->desc)) fprintf(f,"'%s'   \t",xtal->desc);
	else					fprintf(f,"\t\t\t");
	fprintf(f,"Space Group = %d   %s       Vc = %g (nm^3)",xtal->SpaceGroup, symmetryString(xtal->SpaceGroup,sym),xtal->Vc/(u*u*u));
	if (xtal->density>0) fprintf(f,"       density = %g (g/cm^3)",xtal->density);
	if (!(xtal->alphaT<=0)) fprintf(f,"     coefficient of thermal expansion is %g\n",xtal->alphaT);
	else fprintf(f,"\n");
	fprintf(f,"lattice constants  { %.15gnm, %.15gnm, %.15gnm,   %.15g°, %.15g°, %.15g° }\n",xtal->a/u,xtal->b/u,xtal->c/u,xtal->alpha*180/M_PI,xtal->beta*180/M_PI,xtal->gamma*180/M_PI);

	if (Ntype<1) fprintf(f,"No Atoms Defined\n");
	else {
		long	mult=1;
		/*		reMakeAtomXYZs(xtal) */

		fprintf(f,"\natom types:\n         Z        x         y         z     mult");
		if (showOcc) fprintf(f,"     occ");
		if (showDebye) fprintf(f,"      Debye");
		fprintf(f,"\n");
		for (i=0;i<MIN(Ntype,50);i++) {			/* print the first 50 atom positions */
			mult = multiplicityOfAtomType(xtal->atom[i].type,xtal);
			fprintf(f,"%5s  %3d    (%.5f   %.5f   %.5f)  %2ld",xtal->atomType[i].name,xtal->atomType[i].Zatom,xtal->atomType[i].x,xtal->atomType[i].y,xtal->atomType[i].z,mult);
			if (showOcc) fprintf(f,"     %5.3f",xtal->atomType[i].occ);
			if (showDebye) fprintf(f,"       %g",xtal->atom[i].Debye);
			fprintf(f,"\n");
		}
	}

	fprintf(f,"\nthere are %lu actual atoms, they are:\n  Z        x         y         z    ",N);
	if (showOcc) fprintf(f,"      occ");
	if (showType) fprintf(f,"     type");
	if (showDebye) fprintf(f,"    Debye");
	fprintf(f,"\n");
	for (i=0;i<MIN(N,50);i++) {			/* print the first 50 atom positions */
		fprintf(f,"%3d    (%.5f   %.5f   %.5f)",xtal->atom[i].Zatom,xtal->atom[i].x,xtal->atom[i].y,xtal->atom[i].z);
		if (showOcc) fprintf(f,"     %5.3f",xtal->atom[i].occ);
		if (showType) fprintf(f,"      %ld",xtal->atom[i].type);
		if (showDebye) fprintf(f,"       %g",xtal->atom[i].Debye);
		fprintf(f,"\n");
	}
	if (N>50) fprintf(f,"     ... only showing the first 50\n");

	if (xtal->useUnconventional) {					/* Unconventional exists, transform the lattice by it */
		fprintf(f,"\rusing an Uncoventional Cell, the transform is:\n");
		fprintf(f,"\t\t%+6.3f\t\t%+6.3f\t\t%+6.3f\n",		xtal->Unconventional[0][0],xtal->Unconventional[0][1],xtal->Unconventional[0][2]);
		fprintf(f,"\tT =\t%+6.3f\t\t%+6.3f\t\t%+6.3f\n",	xtal->Unconventional[1][0],xtal->Unconventional[1][1],xtal->Unconventional[1][2]);
		fprintf(f,"\t\t%+6.3f\t\t%+6.3f\t\t%+6.3f\n",		xtal->Unconventional[2][0],xtal->Unconventional[2][1],xtal->Unconventional[2][2]);
	}
	fprintf(f,"\n");
	fprintf(f,"                a               b               c   (nm)\n");
	fprintf(f,"direct     %+10.7f      %+10.7f      %+10.7f\n",xtal->direct[0][0]/u,xtal->direct[0][1]/u,xtal->direct[0][2]/u);
	fprintf(f,"lattice    %+10.7f      %+10.7f      %+10.7f\n",xtal->direct[1][0]/u,xtal->direct[1][1]/u,xtal->direct[1][2]/u);
	fprintf(f,"           %+10.7f      %+10.7f      %+10.7f\n",xtal->direct[2][0]/u,xtal->direct[2][1]/u,xtal->direct[2][2]/u);
	fprintf(f,"\n                a*              b*              c*  (1/nm)\n");
	fprintf(f,"recip     %+10.6f       %+10.6f      %+10.6f\n",xtal->recip[0][0]*u,xtal->recip[0][1]*u,xtal->recip[0][2]*u);
	fprintf(f,"lattice   %+10.6f       %+10.6f      %+10.6f\n",xtal->recip[1][0]*u,xtal->recip[1][1]*u,xtal->recip[1][2]*u);
	fprintf(f,"          %+10.6f       %+10.6f      %+10.6f\n",xtal->recip[2][0]*u,xtal->recip[2][1]*u,xtal->recip[2][2]*u);
}


long multiplicityOfAtomType(
size_t	type,							/* number of atom type, starts with 0 */
struct crystalStructure *xtal)			/* lattice and all of the crystal structure values */
{
	size_t	Ntype = xtal->Ntype;
	size_t	N = xtal->N;				/* total number of real atoms */
	size_t	i;
	size_t	m;							/* the desired multiplicity */
	if ((int)type<0 || type>=Ntype) return -1;	/* an error */
	for (i=m=0;i<N;i++) m += xtal->atom[m].type == type;
	return m;
}



/*
 *	================================================================================
 *
 *			This section is for making the matricies for the symmetry operations for Space Groups
 */

int SetSymOpsForSpaceGroup(				/* make the symmetry operations mats and vecs (if needed), returns number of operations, 0 means error */
int		SpaceGroup,
struct equivOpsStructure *sym)			/* all of the symmetry operations, to be filled here */
{
	int		N;							/* number of symmetry ops */
	char	*symOperations=NULL;		/* the list of all symmetry operations for this Space Group, the longest is 3167 bytes */
	double	***mats=NULL;
	double	**bvec=NULL;
	double	mat[3][3], vec[3];
	int		i;
	char	*s0, *s1, str[3200];

	sym->SpaceGroup = 0;				/* init to error */
	if (SpaceGroup<1 || 230<SpaceGroup) { fprintf(stderr,"Bad Space Group = %d, in SetSymOpsForSpaceGroup()",SpaceGroup); return 0; }
	sym->N = N = setSymLine(SpaceGroup,&symOperations);	/* a string like "x,y,z;-x,-y,z;-x,y,-z;x,-y,-z;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;-x+1/2,y+1/2,-z;x+1/2,-y+1/2,-z" */
	sym->SpaceGroup = SpaceGroup;
	allocateEquivAtomOps(sym);			/* allocate all of the space here */
	mats = sym->equivXYZM;				/* mats and bvec are for convienence */
	bvec = sym->equivXYZB;

	strncpy(str,symOperations,3199);
	str[3199] = '\0';
	s0 = str;
	for (i=0;i<N;i+=1) {
		s1 = strchr(s0,';');			/* find next semi-colon */
		if (s1) *s1 = '\0';				/* and change the semi-colon to a null */
		if (make1MatrixAndVecFromSymLine(s0,mat,vec)) { fprintf(stderr,"error making symmetry matricies in SetSymOpsForSpaceGroup()\n"); exit(1); }
		mats[i][0][0] = mat[0][0];		mats[i][0][1] = mat[0][1];		mats[i][0][2] = mat[0][2];
		mats[i][1][0] = mat[1][0];		mats[i][1][1] = mat[1][1];		mats[i][1][2] = mat[1][2];
		mats[i][2][0] = mat[2][0];		mats[i][2][1] = mat[2][1];		mats[i][2][2] = mat[2][2];
		bvec[i][0] = vec[0];			bvec[i][1] = vec[1];			bvec[i][2] = vec[2];
		s0 = ++s1;
	}
	if (symOperations) CHECK_FREE(symOperations)
	return N;
}



int make1MatrixAndVecFromSymLine(		/*  returns result in mat_SymItem and vec_SymItem, return value is error flag */
char	*symItem,						/* something like "-x+1/2,-y+1/2,z" */
double	mat[3][3],
double	vec[3])
{
	double	m0,m1,m2, b;
	int		i, err=0;
	char	*s0, *s1;
	char	str[256];

	strncpy(str,symItem,255);
	str[255] = '\0';
	s0 = str;
	for (i=0;i<3;i++) {
		s1 = strchr(s0,',');			/* find next comma */
		if (s1) *s1 = '\0';				/* and change the comma to a null */
		err = err || ParseOneSymEquation(s0,&m0,&m1,&m2,&b);
		mat[i][0] = m0;
		mat[i][1] = m1;
		mat[i][2] = m2;
		vec[i] = b;
		s0 = ++s1;
	}
	return err;
}
/*	err = make1MatrixAndVecFromSymLine("-x+y-2/3,x-y,-z-1/4",mat,vec);
 *	printf("\nfor symItem = '%s',    err=%d\n","-x+y-2/3,x-y,-z-1/4",err);
 *	printMat33(mat);
 *	printf("\n");
 *	printVec3(vec);
 */

/* parse one expression of form "-x+y"  or "-x", or "-x+y, etc. */
int ParseOneSymEquation(
char	*expression,
double	*m0,							/* results */
double	*m1,
double	*m2,
double	*b)
{
	double	num=NAN;					/* numerator */
	int		i;
	int		is;
	int		op;
	int		N;							/* length of expression */

	*m0 = *m1 = *m2 = *b = 0;			/* init, some may not be set in the strswitch */
	N = strlen(expression);

	for (i=0, op=1,is=1; i<N && op; i++) {
		switch(expression[i]) {
			case '-':
				is = -1;
				break;
			case '+':
				is = 1;
				break;
			case 'x':
				*m0 = is;
				is = 1;
				break;
			case 'y':
				*m1 = is;
				is = 1;
				break;
			case 'z':
				*m2 = is;
				is = 1;
				break;
			default:					/* should be a digit */
				op = 0;					/* halt loop, and set pointer back */
				num = atoi(&(expression[i]));	/* num=str2num(expression[i,Inf]) */				
		}
	}

	if (op) return 0;					/* ran out of things to do, no b */
	if ('/' != expression[i]) {fprintf(stderr,"error on expression = %s\n",expression); return 1;}
	*b = is * num/atoi(&(expression[i+1])); /* eval the constant */
	if (isNAN(*b)) {fprintf(stderr,"error on expression = %s\n",expression); return 1;}
	return 0;							/* return no-error */
}
/*	i = ParseOneSymEquation("-x+y-2/3", &m0,&m1,&m2, &b);
 *	printf("for expression = '%s',  m0=%g, m1=%g, m2=%g, b=%g\n","-x+y-2/3",m0,m1,m2,b);
 *	i = ParseOneSymEquation("x-y", &m0,&m1,&m2, &b);
 *	printf("for expression = '%s',  m0=%g, m1=%g, m2=%g, b=%g\n","x-y",m0,m1,m2,b);
 *	i = ParseOneSymEquation("-z-1/4", &m0,&m1,&m2, &b);
 *	printf("for expression = '%s',  m0=%g, m1=%g, m2=%g, b=%g\n","-z-1/4",m0,m1,m2,b);
 */


/*
 *	void printMat33(double m[3][3])
 *	{
 *		printf("%+3f.0   %+3f.0   %+3f.0\n",m[0][0],m[0][1],m[0][2]);
 *		printf("%+3f.0   %+3f.0   %+3f.0\n",m[1][0],m[1][1],m[1][2]);
 *		printf("%+3f.0   %+3f.0   %+3f.0\n",m[2][0],m[2][1],m[2][2]);
 *	}
 *	void printVec3(double v[3])
 *	{
 *		printf("%+3f.0   %+3f.0   %+3f.0\n",v[0],v[1],v[2]);
 *	}
 *
 *
 *	void printEquivOpsStructure(
 *	struct equivOpsStructure *sym)				// all of the symmetry operations
 *	{
 *
 * 		double **m,*v;
 *		printf("for SpaceGroup = %d,  there are %d symmetry operations\n",sym->SpaceGroup,sym->N);
 *
 *		m = (sym->equivXYZM)[0];
 *		v = (sym->equivXYZB)[0];
 *		printf("\nthe first is:\n");
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[0][0],m[0][1],m[0][2],v[0]);
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[1][0],m[1][1],m[1][2],v[1]);
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[2][0],m[2][1],m[2][2],v[2]);
 *
 *		m = (sym->equivXYZM)[1];
 *		v = (sym->equivXYZB)[1];
 *		printf("\nthe second is:\n");
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[0][0],m[0][1],m[0][2],v[0]);
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[1][0],m[1][1],m[1][2],v[1]);
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[2][0],m[2][1],m[2][2],v[2]);
 *
 *		m = (sym->equivXYZM)[sym->N - 2];
 *		v = (sym->equivXYZB)[sym->N - 2];
 *		printf("\nthe 2nd from last is:\n");
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[0][0],m[0][1],m[0][2],v[0]);
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[1][0],m[1][1],m[1][2],v[1]);
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[2][0],m[2][1],m[2][2],v[2]);
 *
 *		m = (sym->equivXYZM)[sym->N - 1];
 *		v = (sym->equivXYZB)[sym->N - 1];
 *		printf("\nthe last is:\n");
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[0][0],m[0][1],m[0][2],v[0]);
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[1][0],m[1][1],m[1][2],v[1]);
 *		printf("%+3f.0   %+3f.0   %+3f.0         %+3f.0\n",m[2][0],m[2][1],m[2][2],v[2]);
 *	}
 */



int atomicNumber(										/* returns atomic number from name of element */
char	*ele)
{
	char	symb[4];									/* atomic symbol with trailing semi-colon*/
	char	*end;										/* position of symb in symbols */
	char	*p;											/* pointer into symbols */
	int		Z;											/* atomic number */
	char	atomicSymbols[]="H;He;Li;Be;B;C;N;O;F;Ne;Na;Mg;Al;Si;P;S;Cl;Ar;K;Ca;Sc;Ti;V;Cr;Mn;Fe;Co;Ni;Cu;Zn;Ga;Ge;As;\
Se;Br;Kr;Rb;Sr;Y;Zr;Nb;Mo;Tc;Ru;Rh;Pd;Ag;Cd;In;Sn;Sb;Te;I;Xe;Cs;Ba;La;Ce;Pr;Nd;Pm;Sm;Eu;Gd;Tb;Dy;Ho;Er;Tm;Yb;Lu;\
Hf;Ta;W;Re;Os;Ir;Pt;Au;Hg;Tl;Pb;Bi;Po;At;Rn;Fr;Ra;Ac;Th;Pa;U;Np;Pu;Am;Cm;Bk;Cf;Es;Fm;Md;No;Lr;Rf;Db;Sg;Bh;Hs;Mt;Ds;";

	if (!ele) return 0;									/* bad pointer */
	symb[0] = toupper(ele[0]);							/* symb is proper atomic symbol */
	if (ele[0]) { symb[1] = isalpha(ele[1]) ? tolower(ele[1]) : '\0'; }
	symb[2] = symb[3] = '\0';
	symb[strlen(symb)] = ';';							/* symb is now has sem-colon at end of atomic symbol */
	end = strstr(atomicSymbols,symb); 
	if (!end) return 0;									/* failed to find symb */
	for(Z=1,p=strchr(atomicSymbols,';'); p<=end; p=strchr(++p,';')) Z++;
	return Z;
}



int		setSymLine(						/* returns the number of symmetry operations in str */
int		SpaceGroup,						/* Space Group number [1,230] */
char	**str)							/* pointer to result, allocated here */
{
	size_t	len;						/* length of result */
	int		Nops=0;						/* number of symmetry operations */
	char	*p;
	char	*symLines[]={				/* there are 230 items in this array */
				"x,y,z",				/* longest is 3167 bytes in SpaceGroups 227 & 228 */
				"x,y,z;-x,-y,-z",
				"x,y,z;-x,y,-z",
				"x,y,z;-x,y+1/2,-z",
				"x,y,z;-x,y,-z;x+1/2,y+1/2,z;-x+1/2,y+1/2,-z",
				"x,y,z;x,-y,z",
				"x,y,z;x,-y,z+1/2",
				"x,y,z;x,-y,z;x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;x,-y,z+1/2;x+1/2,y+1/2,z;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,y,-z;-x,-y,-z;x,-y,z",
				"x,y,z;-x,y+1/2,-z;-x,-y,-z;x,-y+1/2,z",
				"x,y,z;-x,y,-z;-x,-y,-z;x,-y,z;x+1/2,y+1/2,z;-x+1/2,y+1/2,-z;-x+1/2,-y+1/2,-z;x+1/2,-y+1/2,z",
				"x,y,z;-x,y,-z+1/2;-x,-y,-z;x,-y,z+1/2",
				"x,y,z;-x,y+1/2,-z+1/2;-x,-y,-z;x,-y+1/2,z+1/2",
				"x,y,z;-x,y,-z+1/2;-x,-y,-z;x,-y,z+1/2;x+1/2,y+1/2,z;-x+1/2,y+1/2,-z+1/2;-x+1/2,-y+1/2,-z;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;x,-y,-z;-x,y,-z",
				"x,y,z;-x,-y,z+1/2;x,-y,-z;-x,y,-z+1/2",
				"x,y,z;-x,-y,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z",
				"x,y,z;-x+1/2,-y,z+1/2;x+1/2,-y+1/2,-z;-x,y+1/2,-z+1/2",
				"x,y,z;-x,-y,z+1/2;x,-y,-z;-x,y,-z+1/2;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z+1/2;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z+1/2",
				"x,y,z;-x,-y,z;x,-y,-z;-x,y,-z;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z",
				"x,y,z;-x,-y,z;x,-y,-z;-x,y,-z;x,y+1/2,z+1/2;-x,-y+1/2,z+1/2;x,-y+1/2,-z+1/2;-x,y+1/2,-z+1/2;x+1/2,y,z+1/2;-x+1/2,-y,z+1/2;\
x+1/2,-y,-z+1/2;-x+1/2,y,-z+1/2;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z",
				"x,y,z;-x,-y,z;x,-y,-z;-x,y,-z;x+1/2,y+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2",
				"x,y,z;-x,-y+1/2,z;x,-y,-z+1/2;-x+1/2,y,-z;x+1/2,y+1/2,z+1/2;-x+1/2,-y,z+1/2;x+1/2,-y+1/2,-z;-x,y+1/2,-z+1/2",
				"x,y,z;-x,-y,z;-x,y,z;x,-y,z",
				"x,y,z;-x,-y,z+1/2;-x,y,z;x,-y,z+1/2",
				"x,y,z;-x,-y,z;-x,y,z+1/2;x,-y,z+1/2",
				"x,y,z;-x,-y,z;-x+1/2,y,z;x+1/2,-y,z",
				"x,y,z;-x,-y,z+1/2;-x+1/2,y,z+1/2;x+1/2,-y,z",
				"x,y,z;-x,-y,z;-x,y+1/2,z+1/2;x,-y+1/2,z+1/2",
				"x,y,z;-x+1/2,-y,z+1/2;-x,y,z;x+1/2,-y,z+1/2",
				"x,y,z;-x,-y,z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x,-y,z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z",
				"x,y,z;-x,-y,z;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;-x,y,z;x,-y,z;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x,-y,z+1/2;-x,y,z;x,-y,z+1/2;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;-x,y,z+1/2;x,-y,z+1/2;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;-x,y,z;x,-y,z;x,y+1/2,z+1/2;-x,-y+1/2,z+1/2;-x,y+1/2,z+1/2;x,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;-x,y,z+1/2;x,-y,z+1/2;x,y+1/2,z+1/2;-x,-y+1/2,z+1/2;-x,y+1/2,z;x,-y+1/2,z",
				"x,y,z;-x,-y,z;-x+1/2,y,z;x+1/2,-y,z;x,y+1/2,z+1/2;-x,-y+1/2,z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;-x+1/2,y,z+1/2;x+1/2,-y,z+1/2;x,y+1/2,z+1/2;-x,-y+1/2,z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x,-y,z;-x,y,z;x,-y,z;x,y+1/2,z+1/2;-x,-y+1/2,z+1/2;-x,y+1/2,z+1/2;x,-y+1/2,z+1/2;x+1/2,y,z+1/2;-x+1/2,-y,z+1/2;\
-x+1/2,y,z+1/2;x+1/2,-y,z+1/2;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x,-y,z;-x+1/4,y+1/4,z+1/4;x+1/4,-y+1/4,z+1/4;x,y+1/2,z+1/2;-x,-y+1/2,z+1/2;-x+1/4,y+3/4,z+3/4;x+1/4,-y+3/4,z+3/4;\
x+1/2,y,z+1/2;-x+1/2,-y,z+1/2;-x+3/4,y+1/4,z+3/4;x+3/4,-y+1/4,z+3/4;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;-x+3/4,y+3/4,z+1/4;\
x+3/4,-y+3/4,z+1/4",
				"x,y,z;-x,-y,z;-x,y,z;x,-y,z;x+1/2,y+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;-x,y,z+1/2;x,-y,z+1/2;x+1/2,y+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x,-y,z;-x+1/2,y,z;x+1/2,-y,z;x+1/2,y+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;-x,y+1/2,z+1/2;x,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;x,-y,-z;-x,y,-z;-x,-y,-z;x,y,-z;-x,y,z;x,-y,z",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-x,-y,z;x,-y,-z;-x,y,-z;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;x,-y,-z+1/2;-x,y,-z+1/2;-x,-y,-z;x,y,-z;-x,y,z+1/2;x,-y,z+1/2",
				"x,y,z;-x+1/2,-y+1/2,-z;-x,-y,z;x,-y,-z;-x,y,-z;x+1/2,y+1/2,-z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x+1/2,-y,z;x+1/2,-y,-z;-x,y,-z;-x,-y,-z;x+1/2,y,-z;-x+1/2,y,z;x,-y,z",
				"x,y,z;-x+1/2,-y,z;x,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;-x,-y,-z;x+1/2,y,-z;-x,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x+1/2,-y,z+1/2;x,-y,-z;-x+1/2,y,-z+1/2;-x,-y,-z;x+1/2,y,-z+1/2;-x,y,z;x+1/2,-y,z+1/2",
				"x,y,z;-x+1/2,-y,z;x+1/2,-y,-z+1/2;-x,y,-z+1/2;-x,-y,-z;x+1/2,y,-z;-x+1/2,y,z+1/2;x,-y,z+1/2",
				"x,y,z;-x,-y,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;-x,-y,-z;x,y,-z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x+1/2,-y+1/2,z;x+1/2,-y,-z+1/2;-x,y+1/2,-z+1/2;-x,-y,-z;x+1/2,y+1/2,-z;-x+1/2,y,z+1/2;x,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z+1/2;x,-y+1/2,-z;-x,y+1/2,-z+1/2;-x,-y,-z;x,y,-z+1/2;-x,y+1/2,z;x,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;-x,-y,-z;x,y,-z;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x+1/2,-y+1/2,-z;-x,-y,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;x+1/2,y+1/2,-z;-x,y,z;x,-y,z",
				"x,y,z;-x+1/2,-y+1/2,z+1/2;x+1/2,-y+1/2,-z;-x,y,-z+1/2;-x,-y,-z;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z;x,-y,z+1/2",
				"x,y,z;-x+1/2,-y,z+1/2;x+1/2,-y+1/2,-z;-x,y+1/2,-z+1/2;-x,-y,-z;x+1/2,y,-z+1/2;-x+1/2,y+1/2,z;x,-y+1/2,z+1/2",
				"x,y,z;-x+1/2,-y,z+1/2;x+1/2,-y+1/2,-z+1/2;-x,y+1/2,-z;-x,-y,-z;x+1/2,y,-z+1/2;-x+1/2,y+1/2,z+1/2;x,-y+1/2,z",
				"x,y,z;-x,-y,z+1/2;x,-y,-z;-x,y,-z+1/2;-x,-y,-z;x,y,-z+1/2;-x,y,z;x,-y,z+1/2;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z+1/2;x+1/2,-y+1/2,-z;\
-x+1/2,y+1/2,-z+1/2;-x+1/2,-y+1/2,-z;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y+1/2,z+1/2;x,-y,-z;-x,y+1/2,-z+1/2;-x,-y,-z;x,y+1/2,-z+1/2;-x,y,z;x,-y+1/2,z+1/2;x+1/2,y+1/2,z;-x+1/2,-y,z+1/2;\
x+1/2,-y+1/2,-z;-x+1/2,y,-z+1/2;-x+1/2,-y+1/2,-z;x+1/2,y,-z+1/2;-x+1/2,y+1/2,z;x+1/2,-y,z+1/2",
				"x,y,z;-x,-y,z;x,-y,-z;-x,y,-z;-x,-y,-z;x,y,-z;-x,y,z;x,-y,z;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;\
-x+1/2,-y+1/2,-z;x+1/2,y+1/2,-z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x,-y,z;x,-y,-z+1/2;-x,y,-z+1/2;-x,-y,-z;x,y,-z;-x,y,z+1/2;x,-y,z+1/2;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;x+1/2,-y+1/2,-z+1/2;\
-x+1/2,y+1/2,-z+1/2;-x+1/2,-y+1/2,-z;x+1/2,y+1/2,-z;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y+1/2,z;x,-y,-z;-x,y+1/2,-z;-x,-y,-z;x,y+1/2,-z;-x,y,z;x,-y+1/2,z;x+1/2,y+1/2,z;-x+1/2,-y,z;x+1/2,-y+1/2,-z;-x+1/2,y,-z;\
-x+1/2,-y+1/2,-z;x+1/2,y,-z;-x+1/2,y+1/2,z;x+1/2,-y,z",
				"x,y,z;-x,-y+1/2,-z+1/2;-x,-y,z;x,-y,-z;-x,y,-z;x,y+1/2,-z+1/2;-x,y+1/2,z+1/2;x,-y+1/2,z+1/2;x+1/2,y+1/2,z;-x+1/2,-y,-z+1/2;\
-x+1/2,-y+1/2,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;x+1/2,y,-z+1/2;-x+1/2,y,z+1/2;x+1/2,-y,z+1/2",
				"x,y,z;-x,-y,z;x,-y,-z;-x,y,-z;-x,-y,-z;x,y,-z;-x,y,z;x,-y,z;x,y+1/2,z+1/2;-x,-y+1/2,z+1/2;x,-y+1/2,-z+1/2;-x,y+1/2,-z+1/2;\
-x,-y+1/2,-z+1/2;x,y+1/2,-z+1/2;-x,y+1/2,z+1/2;x,-y+1/2,z+1/2;x+1/2,y,z+1/2;-x+1/2,-y,z+1/2;x+1/2,-y,-z+1/2;-x+1/2,y,-z+1/2;\
-x+1/2,-y,-z+1/2;x+1/2,y,-z+1/2;-x+1/2,y,z+1/2;x+1/2,-y,z+1/2;x+1/2,y+1/2,z;-x+1/2,-y+1/2,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;\
-x+1/2,-y+1/2,-z;x+1/2,y+1/2,-z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x+1/4,-y+1/4,-z+1/4;-x,-y,z;x,-y,-z;-x,y,-z;x+1/4,y+1/4,-z+1/4;-x+1/4,y+1/4,z+1/4;x+1/4,-y+1/4,z+1/4;x,y+1/2,z+1/2;\
-x+1/4,-y+3/4,-z+3/4;-x,-y+1/2,z+1/2;x,-y+1/2,-z+1/2;-x,y+1/2,-z+1/2;x+1/4,y+3/4,-z+3/4;-x+1/4,y+3/4,z+3/4;x+1/4,-y+3/4,z+3/4;\
x+1/2,y,z+1/2;-x+3/4,-y+1/4,-z+3/4;-x+1/2,-y,z+1/2;x+1/2,-y,-z+1/2;-x+1/2,y,-z+1/2;x+3/4,y+1/4,-z+3/4;-x+3/4,y+1/4,z+3/4;\
x+3/4,-y+1/4,z+3/4;x+1/2,y+1/2,z;-x+3/4,-y+3/4,-z+1/4;-x+1/2,-y+1/2,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;x+3/4,y+3/4,-z+1/4;\
-x+3/4,y+3/4,z+1/4;x+3/4,-y+3/4,z+1/4",
				"x,y,z;-x,-y,z;x,-y,-z;-x,y,-z;-x,-y,-z;x,y,-z;-x,y,z;x,-y,z;x+1/2,y+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;x+1/2,-y+1/2,-z+1/2;\
-x+1/2,y+1/2,-z+1/2;-x+1/2,-y+1/2,-z+1/2;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;-x,-y,z;x,-y,-z+1/2;-x,y,-z+1/2;-x,-y,-z;x,y,-z;-x,y,z+1/2;x,-y,z+1/2;x+1/2,y+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;x+1/2,-y+1/2,-z;\
-x+1/2,y+1/2,-z;-x+1/2,-y+1/2,-z+1/2;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x,-y+1/2,z;x,-y,-z+1/2;-x+1/2,y,-z;-x,-y,-z;x,y+1/2,-z;-x,y,z+1/2;x+1/2,-y,z;x+1/2,y+1/2,z+1/2;-x+1/2,-y,z+1/2;\
x+1/2,-y+1/2,-z;-x,y+1/2,-z+1/2;-x+1/2,-y+1/2,-z+1/2;x+1/2,y,-z+1/2;-x+1/2,y+1/2,z;x,-y+1/2,z+1/2",
				"x,y,z;-x,-y+1/2,z;x,-y,-z;-x,y+1/2,-z;-x,-y,-z;x,y+1/2,-z;-x,y,z;x,-y+1/2,z;x+1/2,y+1/2,z+1/2;-x+1/2,-y,z+1/2;x+1/2,-y+1/2,-z+1/2;\
-x+1/2,y,-z+1/2;-x+1/2,-y+1/2,-z+1/2;x+1/2,y,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y,z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z",
				"x,y,z;-y,x,z+1/4;-x,-y,z+1/2;y,-x,z+3/4",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2",
				"x,y,z;-y,x,z+3/4;-x,-y,z+1/2;y,-x,z+1/4",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x+1/2,y+1/2,z+1/2;-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;y+1/2,-x+1/2,z+1/2",
				"x,y,z;-y,x+1/2,z+1/4;-x,-y,z;y,-x+1/2,z+1/4;x+1/2,y+1/2,z+1/2;-y+1/2,x,z+3/4;-x+1/2,-y+1/2,z+1/2;y+1/2,-x,z+3/4",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;x+1/2,y+1/2,z+1/2;y+1/2,-x+1/2,-z+1/2;-x+1/2,-y+1/2,z+1/2;-y+1/2,x+1/2,-z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;-x,-y,-z;y,-x,-z+1/2;x,y,-z;-y,x,-z+1/2",
				"x,y,z;-x+1/2,-y+1/2,-z;-y+1/2,x+1/2,z;-x,-y,z;y+1/2,-x+1/2,z;y,-x,-z;-y,x,-z;x+1/2,y+1/2,-z",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;y,-x,-z;-y,x,-z;x+1/2,y+1/2,-z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;x+1/2,y+1/2,z+1/2;-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;\
y+1/2,-x+1/2,z+1/2;-x+1/2,-y+1/2,-z+1/2;y+1/2,-x+1/2,-z+1/2;x+1/2,y+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2",
				"x,y,z;-x,-y+1/2,-z+1/4;-y,x+1/2,z+1/4;-x,-y,z;y,-x+1/2,z+1/4;y,-x,-z;-y,x,-z;x,y+1/2,-z+1/4;x+1/2,y+1/2,z+1/2;-x+1/2,-y,-z+3/4;\
-y+1/2,x,z+3/4;-x+1/2,-y+1/2,z+1/2;y+1/2,-x,z+3/4;y+1/2,-x+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2;x+1/2,y,-z+3/4",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-y,-z;-x,y,-z;y,x,-z;-y,-x,-z",
				"x,y,z;-y+1/2,x+1/2,z;-x,-y,z;y+1/2,-x+1/2,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;y,x,-z;-y,-x,-z",
				"x,y,z;-y,x,z+1/4;-x,-y,z+1/2;y,-x,z+3/4;x,-y,-z+1/2;-x,y,-z;y,x,-z+3/4;-y,-x,-z+1/4",
				"x,y,z;-y+1/2,x+1/2,z+1/4;-x,-y,z+1/2;y+1/2,-x+1/2,z+3/4;x+1/2,-y+1/2,-z+3/4;-x+1/2,y+1/2,-z+1/4;y,x,-z;-y,-x,-z+1/2",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;x,-y,-z;-x,y,-z;y,x,-z+1/2;-y,-x,-z+1/2",
				"x,y,z;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;y,x,-z;-y,-x,-z",
				"x,y,z;-y,x,z+3/4;-x,-y,z+1/2;y,-x,z+1/4;x,-y,-z+1/2;-x,y,-z;y,x,-z+1/4;-y,-x,-z+3/4",
				"x,y,z;-y+1/2,x+1/2,z+3/4;-x,-y,z+1/2;y+1/2,-x+1/2,z+1/4;x+1/2,-y+1/2,-z+1/4;-x+1/2,y+1/2,-z+3/4;y,x,-z;-y,-x,-z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-y,-z;-x,y,-z;y,x,-z;-y,-x,-z;x+1/2,y+1/2,z+1/2;-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;\
y+1/2,-x+1/2,z+1/2;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2",
				"x,y,z;-y,x+1/2,z+1/4;-x,-y,z;y,-x+1/2,z+1/4;x,-y+1/2,-z+1/4;-x,y+1/2,-z+1/4;y,x,-z;-y,-x,-z;x+1/2,y+1/2,z+1/2;-y+1/2,x,z+3/4;\
-x+1/2,-y+1/2,z+1/2;y+1/2,-x,z+3/4;x+1/2,-y,-z+3/4;-x+1/2,y,-z+3/4;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;-x,y,z;x,-y,z;-y,-x,z;y,x,z",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;-x,y,z+1/2;x,-y,z+1/2;-y,-x,z;y,x,z",
				"x,y,z;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y,-x,z;y,x,z",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;-x,y,z+1/2;x,-y,z+1/2;-y,-x,z+1/2;y,x,z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;-x,y,z;x,-y,z;-y,-x,z+1/2;y,x,z+1/2",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;-x,y,z;x,-y,z;-y,-x,z;y,x,z;x+1/2,y+1/2,z+1/2;-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;\
y+1/2,-x+1/2,z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;-x,y,z+1/2;x,-y,z+1/2;-y,-x,z+1/2;y,x,z+1/2;x+1/2,y+1/2,z+1/2;-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;\
y+1/2,-x+1/2,z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z",
				"x,y,z;-y,x+1/2,z+1/4;-x,-y,z;y,-x+1/2,z+1/4;-x,y,z;x,-y,z;-y,-x+1/2,z+1/4;y,x+1/2,z+1/4;x+1/2,y+1/2,z+1/2;-y+1/2,x,z+3/4;\
-x+1/2,-y+1/2,z+1/2;y+1/2,-x,z+3/4;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y+1/2,-x,z+3/4;y+1/2,x,z+3/4",
				"x,y,z;-y,x+1/2,z+1/4;-x,-y,z;y,-x+1/2,z+1/4;-x,y,z+1/2;x,-y,z+1/2;-y+1/2,-x,z+1/4;y+1/2,x,z+1/4;x+1/2,y+1/2,z+1/2;-y+1/2,x,z+3/4;\
-x+1/2,-y+1/2,z+1/2;y+1/2,-x,z+3/4;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y,-x+1/2,z+3/4;y,x+1/2,z+3/4",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;x,-y,-z;-x,y,-z;-y,-x,z;y,x,z",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;x,-y,-z+1/2;-x,y,-z+1/2;-y,-x,z+1/2;y,x,z+1/2",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;y,x,-z;-y,-x,-z;-x,y,z;x,-y,z",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;y,x,-z+1/2;-y,-x,-z+1/2;-x,y,z+1/2;x,-y,z+1/2",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;y+1/2,x+1/2,-z;-y+1/2,-x+1/2,-z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;y,x,-z;-y,-x,-z;-x,y,z;x,-y,z;x+1/2,y+1/2,z+1/2;y+1/2,-x+1/2,-z+1/2;-x+1/2,-y+1/2,z+1/2;\
-y+1/2,x+1/2,-z+1/2;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;y,x,-z+1/2;-y,-x,-z+1/2;-x,y,z+1/2;x,-y,z+1/2;x+1/2,y+1/2,z+1/2;y+1/2,-x+1/2,-z+1/2;\
-x+1/2,-y+1/2,z+1/2;-y+1/2,x+1/2,-z+1/2;y+1/2,x+1/2,-z;-y+1/2,-x+1/2,-z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;x,-y,-z;-x,y,-z;-y,-x,z;y,x,z;x+1/2,y+1/2,z+1/2;y+1/2,-x+1/2,-z+1/2;-x+1/2,-y+1/2,z+1/2;\
-y+1/2,x+1/2,-z+1/2;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;x,-y+1/2,-z+1/4;-x,y+1/2,-z+1/4;-y,-x+1/2,z+1/4;y,x+1/2,z+1/4;x+1/2,y+1/2,z+1/2;y+1/2,-x+1/2,-z+1/2;\
-x+1/2,-y+1/2,z+1/2;-y+1/2,x+1/2,-z+1/2;x+1/2,-y,-z+3/4;-x+1/2,y,-z+3/4;-y+1/2,-x,z+3/4;y+1/2,x,z+3/4",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-y,-z;-x,y,-z;y,x,-z;-y,-x,-z;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;-x,y,z;x,-y,z;-y,-x,z;y,x,z",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-y,-z+1/2;-x,y,-z+1/2;y,x,-z+1/2;-y,-x,-z+1/2;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;-x,y,z+1/2;x,-y,z+1/2;\
-y,-x,z+1/2;y,x,z+1/2",
				"x,y,z;-x+1/2,-y+1/2,-z;-y,x,z;-x,-y,z;y,-x,z;y+1/2,-x+1/2,-z;-y+1/2,x+1/2,-z;x,-y,-z;-x,y,-z;y,x,-z;-y,-x,-z;x+1/2,y+1/2,-z;\
-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-y,x,z;-x,-y,z;y,-x,z;y+1/2,-x+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2;x,-y,-z;-x,y,-z;y,x,-z;-y,-x,-z;\
x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;y+1/2,x+1/2,-z;-y+1/2,-x+1/2,-z;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;\
-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;-x,-y,-z;y,-x,-z;\
x,y,-z;-y,x,-z;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-x+1/2,-y+1/2,-z;-y+1/2,x+1/2,z;-x,-y,z;y+1/2,-x+1/2,z;y,-x,-z;-y,x,-z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;y,x,-z;-y,-x,-z;\
x+1/2,y+1/2,-z;-x,y,z;x,-y,z;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z",
				"x,y,z;-x+1/2,-y+1/2,-z;-y+1/2,x+1/2,z;-x,-y,z;y+1/2,-x+1/2,z;y,-x,-z;-y,x,-z;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;y,x,-z+1/2;\
-y,-x,-z+1/2;x+1/2,y+1/2,-z;-x,y,z+1/2;x,-y,z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;x,-y,-z;-x,y,-z;y,x,-z+1/2;-y,-x,-z+1/2;-x,-y,-z;y,-x,-z+1/2;x,y,-z;-y,x,-z+1/2;-x,y,z;x,-y,z;\
-y,-x,z+1/2;y,x,z+1/2",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;x,-y,-z+1/2;-x,y,-z+1/2;y,x,-z;-y,-x,-z;-x,-y,-z;y,-x,-z+1/2;x,y,-z;-y,x,-z+1/2;-x,y,z+1/2;\
x,-y,z+1/2;-y,-x,z;y,x,z",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;y,-x,-z;-y,x,-z;x,-y,-z+1/2;-x,y,-z+1/2;y+1/2,x+1/2,-z;\
-y+1/2,-x+1/2,-z;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y,-x,z+1/2;y,x,z+1/2",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;y,-x,-z;-y,x,-z;x,-y,-z;-x,y,-z;y+1/2,x+1/2,-z+1/2;\
-y+1/2,-x+1/2,-z+1/2;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y,-x,z;y,x,z",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;-x,-y,-z;y,-x,-z+1/2;\
x,y,-z;-y,x,-z+1/2;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;y,x,-z;-y,-x,-z;-x,-y,-z;\
y+1/2,-x+1/2,-z+1/2;x,y,-z;-y+1/2,x+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y,-x,z;y,x,z",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;y,-x,-z;-y,x,-z;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;\
y,x,-z;-y,-x,-z;x+1/2,y+1/2,-z+1/2;-x,y,z;x,-y,z;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;y,-x,-z;-y,x,-z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;\
y,x,-z+1/2;-y,-x,-z+1/2;x+1/2,y+1/2,-z+1/2;-x,y,z+1/2;x,-y,z+1/2;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-y,-z;-x,y,-z;y,x,-z;-y,-x,-z;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;-x,y,z;x,-y,z;-y,-x,z;y,x,z;\
x+1/2,y+1/2,z+1/2;-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;y+1/2,-x+1/2,z+1/2;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;\
y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;-x+1/2,-y+1/2,-z+1/2;y+1/2,-x+1/2,-z+1/2;x+1/2,y+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2;\
-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-y,-z+1/2;-x,y,-z+1/2;y,x,-z+1/2;-y,-x,-z+1/2;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;-x,y,z+1/2;x,-y,z+1/2;\
-y,-x,z+1/2;y,x,z+1/2;x+1/2,y+1/2,z+1/2;-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;y+1/2,-x+1/2,z+1/2;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;\
y+1/2,x+1/2,-z;-y+1/2,-x+1/2,-z;-x+1/2,-y+1/2,-z+1/2;y+1/2,-x+1/2,-z+1/2;x+1/2,y+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2;-x+1/2,y+1/2,z;\
x+1/2,-y+1/2,z;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z",
				"x,y,z;-x,-y+1/2,-z+1/4;-y,x+1/2,z+1/4;-x,-y,z;y,-x+1/2,z+1/4;y,-x,-z;-y,x,-z;x,-y+1/2,-z+1/4;-x,y+1/2,-z+1/4;y,x,-z;-y,-x,-z;\
x,y+1/2,-z+1/4;-x,y,z;x,-y,z;-y,-x+1/2,z+1/4;y,x+1/2,z+1/4;x+1/2,y+1/2,z+1/2;-x+1/2,-y,-z+3/4;-y+1/2,x,z+3/4;-x+1/2,-y+1/2,z+1/2;\
y+1/2,-x,z+3/4;y+1/2,-x+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2;x+1/2,-y,-z+3/4;-x+1/2,y,-z+3/4;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;\
x+1/2,y,-z+3/4;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y+1/2,-x,z+3/4;y+1/2,x,z+3/4",
				"x,y,z;-x,-y+1/2,-z+1/4;-y,x+1/2,z+1/4;-x,-y,z;y,-x+1/2,z+1/4;y,-x,-z;-y,x,-z;x+1/2,-y,-z+1/4;-x+1/2,y,-z+1/4;y,x,-z+1/2;\
-y,-x,-z+1/2;x,y+1/2,-z+1/4;-x,y,z+1/2;x,-y,z+1/2;-y+1/2,-x,z+1/4;y+1/2,x,z+1/4;x+1/2,y+1/2,z+1/2;-x+1/2,-y,-z+3/4;-y+1/2,x,z+3/4;\
-x+1/2,-y+1/2,z+1/2;y+1/2,-x,z+3/4;y+1/2,-x+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2;x,-y+1/2,-z+3/4;-x,y+1/2,-z+3/4;y+1/2,x+1/2,-z;\
-y+1/2,-x+1/2,-z;x+1/2,y,-z+3/4;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z;-y,-x+1/2,z+3/4;y,x+1/2,z+3/4",
				"x,y,z;-y,x-y,z;-x+y,-x,z",
				"x,y,z;-y,x-y,z+1/3;-x+y,-x,z+2/3",
				"x,y,z;-y,x-y,z+2/3;-x+y,-x,z+1/3",
				"x,y,z;-y,x-y,z;-x+y,-x,z;x+2/3,y+1/3,z+1/3;-y+2/3,x-y+1/3,z+1/3;-x+y+2/3,-x+1/3,z+1/3;x+1/3,y+2/3,z+2/3;-y+1/3,x-y+2/3,z+2/3;\
-x+y+1/3,-x+2/3,z+2/3",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-x,-y,-z;y,-x+y,-z;x-y,x,-z",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-x,-y,-z;y,-x+y,-z;x-y,x,-z;x+2/3,y+1/3,z+1/3;-y+2/3,x-y+1/3,z+1/3;-x+y+2/3,-x+1/3,z+1/3;\
-x+2/3,-y+1/3,-z+1/3;y+2/3,-x+y+1/3,-z+1/3;x-y+2/3,x+1/3,-z+1/3;x+1/3,y+2/3,z+2/3;-y+1/3,x-y+2/3,z+2/3;-x+y+1/3,-x+2/3,z+2/3;\
-x+1/3,-y+2/3,-z+2/3;y+1/3,-x+y+2/3,-z+2/3;x-y+1/3,x+2/3,-z+2/3",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-y,-x,-z;-x+y,y,-z;x,x-y,-z",
				"x,y,z;-y,x-y,z;-x+y,-x,z;x-y,-y,-z;-x,-x+y,-z;y,x,-z",
				"x,y,z;-y,x-y,z+1/3;-x+y,-x,z+2/3;-y,-x,-z+2/3;-x+y,y,-z+1/3;x,x-y,-z",
				"x,y,z;-y,x-y,z+1/3;-x+y,-x,z+2/3;x-y,-y,-z+2/3;-x,-x+y,-z+1/3;y,x,-z",
				"x,y,z;-y,x-y,z+2/3;-x+y,-x,z+1/3;-y,-x,-z+1/3;-x+y,y,-z+2/3;x,x-y,-z",
				"x,y,z;-y,x-y,z+2/3;-x+y,-x,z+1/3;x-y,-y,-z+1/3;-x,-x+y,-z+2/3;y,x,-z",
				"x,y,z;-y,x-y,z;-x+y,-x,z;x-y,-y,-z;-x,-x+y,-z;y,x,-z;x+2/3,y+1/3,z+1/3;-y+2/3,x-y+1/3,z+1/3;-x+y+2/3,-x+1/3,z+1/3;\
x-y+2/3,-y+1/3,-z+1/3;-x+2/3,-x+y+1/3,-z+1/3;y+2/3,x+1/3,-z+1/3;x+1/3,y+2/3,z+2/3;-y+1/3,x-y+2/3,z+2/3;-x+y+1/3,-x+2/3,z+2/3;\
x-y+1/3,-y+2/3,-z+2/3;-x+1/3,-x+y+2/3,-z+2/3;y+1/3,x+2/3,-z+2/3",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-x+y,y,z;x,x-y,z;-y,-x,z",
				"x,y,z;-y,x-y,z;-x+y,-x,z;y,x,z;x-y,-y,z;-x,-x+y,z",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2",
				"x,y,z;-y,x-y,z;-x+y,-x,z;y,x,z+1/2;x-y,-y,z+1/2;-x,-x+y,z+1/2",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-x+y,y,z;x,x-y,z;-y,-x,z;x+2/3,y+1/3,z+1/3;-y+2/3,x-y+1/3,z+1/3;-x+y+2/3,-x+1/3,z+1/3;\
-x+y+2/3,y+1/3,z+1/3;x+2/3,x-y+1/3,z+1/3;-y+2/3,-x+1/3,z+1/3;x+1/3,y+2/3,z+2/3;-y+1/3,x-y+2/3,z+2/3;-x+y+1/3,-x+2/3,z+2/3;\
-x+y+1/3,y+2/3,z+2/3;x+1/3,x-y+2/3,z+2/3;-y+1/3,-x+2/3,z+2/3",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2;x+2/3,y+1/3,z+1/3;-y+2/3,x-y+1/3,z+1/3;-x+y+2/3,-x+1/3,z+1/3;\
-x+y+2/3,y+1/3,z+5/6;x+2/3,x-y+1/3,z+5/6;-y+2/3,-x+1/3,z+5/6;x+1/3,y+2/3,z+2/3;-y+1/3,x-y+2/3,z+2/3;-x+y+1/3,-x+2/3,z+2/3;\
-x+y+1/3,y+2/3,z+1/6;x+1/3,x-y+2/3,z+1/6;-y+1/3,-x+2/3,z+1/6",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-y,-x,-z;-x+y,y,-z;x,x-y,-z;-x,-y,-z;y,-x+y,-z;x-y,x,-z;y,x,z;x-y,-y,z;-x,-x+y,z",
				"x,y,z;-y,x-y,z;-x+y,-x,z;-y,-x,-z+1/2;-x+y,y,-z+1/2;x,x-y,-z+1/2;-x,-y,-z;y,-x+y,-z;x-y,x,-z;y,x,z+1/2;x-y,-y,z+1/2;-x,-x+y,z+1/2",
				"x,y,z;-y,x-y,z;-x+y,-x,z;x-y,-y,-z;-x,-x+y,-z;y,x,-z;-x,-y,-z;y,-x+y,-z;x-y,x,-z;-x+y,y,z;x,x-y,z;-y,-x,z",
				"x,y,z;-y,x-y,z;-x+y,-x,z;x-y,-y,-z+1/2;-x,-x+y,-z+1/2;y,x,-z+1/2;-x,-y,-z;y,-x+y,-z;x-y,x,-z;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2",
				"x,y,z;-y,x-y,z;-x+y,-x,z;x-y,-y,-z;-x,-x+y,-z;y,x,-z;-x,-y,-z;y,-x+y,-z;x-y,x,-z;-x+y,y,z;x,x-y,z;-y,-x,z;x+2/3,y+1/3,z+1/3;\
-y+2/3,x-y+1/3,z+1/3;-x+y+2/3,-x+1/3,z+1/3;x-y+2/3,-y+1/3,-z+1/3;-x+2/3,-x+y+1/3,-z+1/3;y+2/3,x+1/3,-z+1/3;-x+2/3,-y+1/3,-z+1/3;\
y+2/3,-x+y+1/3,-z+1/3;x-y+2/3,x+1/3,-z+1/3;-x+y+2/3,y+1/3,z+1/3;x+2/3,x-y+1/3,z+1/3;-y+2/3,-x+1/3,z+1/3;x+1/3,y+2/3,z+2/3;\
-y+1/3,x-y+2/3,z+2/3;-x+y+1/3,-x+2/3,z+2/3;x-y+1/3,-y+2/3,-z+2/3;-x+1/3,-x+y+2/3,-z+2/3;y+1/3,x+2/3,-z+2/3;-x+1/3,-y+2/3,-z+2/3;\
y+1/3,-x+y+2/3,-z+2/3;x-y+1/3,x+2/3,-z+2/3;-x+y+1/3,y+2/3,z+2/3;x+1/3,x-y+2/3,z+2/3;-y+1/3,-x+2/3,z+2/3",
				"x,y,z;-y,x-y,z;-x+y,-x,z;x-y,-y,-z+1/2;-x,-x+y,-z+1/2;y,x,-z+1/2;-x,-y,-z;y,-x+y,-z;x-y,x,-z;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2;\
x+2/3,y+1/3,z+1/3;-y+2/3,x-y+1/3,z+1/3;-x+y+2/3,-x+1/3,z+1/3;x-y+2/3,-y+1/3,-z+5/6;-x+2/3,-x+y+1/3,-z+5/6;y+2/3,x+1/3,-z+5/6;\
-x+2/3,-y+1/3,-z+1/3;y+2/3,-x+y+1/3,-z+1/3;x-y+2/3,x+1/3,-z+1/3;-x+y+2/3,y+1/3,z+5/6;x+2/3,x-y+1/3,z+5/6;-y+2/3,-x+1/3,z+5/6;\
x+1/3,y+2/3,z+2/3;-y+1/3,x-y+2/3,z+2/3;-x+y+1/3,-x+2/3,z+2/3;x-y+1/3,-y+2/3,-z+1/6;-x+1/3,-x+y+2/3,-z+1/6;y+1/3,x+2/3,-z+1/6;\
-x+1/3,-y+2/3,-z+2/3;y+1/3,-x+y+2/3,-z+2/3;x-y+1/3,x+2/3,-z+2/3;-x+y+1/3,y+2/3,z+1/6;x+1/3,x-y+2/3,z+1/6;-y+1/3,-x+2/3,z+1/6",
				"x,y,z;x-y,x,z;-y,x-y,z;-x,-y,z;-x+y,-x,z;y,-x+y,z",
				"x,y,z;x-y,x,z+1/6;-y,x-y,z+1/3;-x,-y,z+1/2;-x+y,-x,z+2/3;y,-x+y,z+5/6",
				"x,y,z;x-y,x,z+5/6;-y,x-y,z+2/3;-x,-y,z+1/2;-x+y,-x,z+1/3;y,-x+y,z+1/6",
				"x,y,z;x-y,x,z+1/3;-y,x-y,z+2/3;-x,-y,z;-x+y,-x,z+1/3;y,-x+y,z+2/3",
				"x,y,z;x-y,x,z+2/3;-y,x-y,z+1/3;-x,-y,z;-x+y,-x,z+2/3;y,-x+y,z+1/3",
				"x,y,z;x-y,x,z+1/2;-y,x-y,z;-x,-y,z+1/2;-x+y,-x,z;y,-x+y,z+1/2",
				"x,y,z;-x+y,-x,-z;-y,x-y,z;x,y,-z;-x+y,-x,z;-y,x-y,-z",
				"x,y,z;x-y,x,z;-y,x-y,z;-x,-y,z;-x+y,-x,z;y,-x+y,z;-x,-y,-z;-x+y,-x,-z;y,-x+y,-z;x,y,-z;x-y,x,-z;-y,x-y,-z",
				"x,y,z;x-y,x,z+1/2;-y,x-y,z;-x,-y,z+1/2;-x+y,-x,z;y,-x+y,z+1/2;-x,-y,-z;-x+y,-x,-z+1/2;y,-x+y,-z;x,y,-z+1/2;x-y,x,-z;-y,x-y,-z+1/2",
				"x,y,z;x-y,x,z;-y,x-y,z;-x,-y,z;-x+y,-x,z;y,-x+y,z;x-y,-y,-z;-x,-x+y,-z;y,x,-z;-y,-x,-z;-x+y,y,-z;x,x-y,-z",
				"x,y,z;x-y,x,z+1/6;-y,x-y,z+1/3;-x,-y,z+1/2;-x+y,-x,z+2/3;y,-x+y,z+5/6;x-y,-y,-z;-x,-x+y,-z+2/3;y,x,-z+1/3;-y,-x,-z+5/6;\
-x+y,y,-z+1/2;x,x-y,-z+1/6",
				"x,y,z;x-y,x,z+5/6;-y,x-y,z+2/3;-x,-y,z+1/2;-x+y,-x,z+1/3;y,-x+y,z+1/6;x-y,-y,-z;-x,-x+y,-z+1/3;y,x,-z+2/3;-y,-x,-z+1/6;\
-x+y,y,-z+1/2;x,x-y,-z+5/6",
				"x,y,z;x-y,x,z+1/3;-y,x-y,z+2/3;-x,-y,z;-x+y,-x,z+1/3;y,-x+y,z+2/3;x-y,-y,-z;-x,-x+y,-z+1/3;y,x,-z+2/3;-y,-x,-z+2/3;-x+y,y,-z;\
x,x-y,-z+1/3",
				"x,y,z;x-y,x,z+2/3;-y,x-y,z+1/3;-x,-y,z;-x+y,-x,z+2/3;y,-x+y,z+1/3;x-y,-y,-z;-x,-x+y,-z+2/3;y,x,-z+1/3;-y,-x,-z+1/3;-x+y,y,-z;\
x,x-y,-z+2/3",
				"x,y,z;x-y,x,z+1/2;-y,x-y,z;-x,-y,z+1/2;-x+y,-x,z;y,-x+y,z+1/2;x-y,-y,-z;-x,-x+y,-z;y,x,-z;-y,-x,-z+1/2;-x+y,y,-z+1/2;x,x-y,-z+1/2",
				"x,y,z;x-y,x,z;-y,x-y,z;-x,-y,z;-x+y,-x,z;y,-x+y,z;-x+y,y,z;x,x-y,z;-y,-x,z;y,x,z;x-y,-y,z;-x,-x+y,z",
				"x,y,z;x-y,x,z;-y,x-y,z;-x,-y,z;-x+y,-x,z;y,-x+y,z;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2;y,x,z+1/2;x-y,-y,z+1/2;-x,-x+y,z+1/2",
				"x,y,z;x-y,x,z+1/2;-y,x-y,z;-x,-y,z+1/2;-x+y,-x,z;y,-x+y,z+1/2;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2;y,x,z;x-y,-y,z;-x,-x+y,z",
				"x,y,z;x-y,x,z+1/2;-y,x-y,z;-x,-y,z+1/2;-x+y,-x,z;y,-x+y,z+1/2;-x+y,y,z;x,x-y,z;-y,-x,z;y,x,z+1/2;x-y,-y,z+1/2;-x,-x+y,z+1/2",
				"x,y,z;-x+y,-x,-z;-y,x-y,z;x,y,-z;-x+y,-x,z;-y,x-y,-z;-y,-x,-z;-x+y,y,-z;x,x-y,-z;-x+y,y,z;x,x-y,z;-y,-x,z",
				"x,y,z;-x+y,-x,-z+1/2;-y,x-y,z;x,y,-z+1/2;-x+y,-x,z;-y,x-y,-z+1/2;-y,-x,-z;-x+y,y,-z;x,x-y,-z;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2",
				"x,y,z;-x+y,-x,-z;-y,x-y,z;x,y,-z;-x+y,-x,z;-y,x-y,-z;x-y,-y,-z;-x,-x+y,-z;y,x,-z;y,x,z;x-y,-y,z;-x,-x+y,z",
				"x,y,z;-x+y,-x,-z+1/2;-y,x-y,z;x,y,-z+1/2;-x+y,-x,z;-y,x-y,-z+1/2;x-y,-y,-z;-x,-x+y,-z;y,x,-z;y,x,z+1/2;x-y,-y,z+1/2;-x,-x+y,z+1/2",
				"x,y,z;x-y,x,z;-y,x-y,z;-x,-y,z;-x+y,-x,z;y,-x+y,z;x-y,-y,-z;-x,-x+y,-z;y,x,-z;-y,-x,-z;-x+y,y,-z;x,x-y,-z;-x,-y,-z;-x+y,-x,-z;\
y,-x+y,-z;x,y,-z;x-y,x,-z;-y,x-y,-z;-x+y,y,z;x,x-y,z;-y,-x,z;y,x,z;x-y,-y,z;-x,-x+y,z",
				"x,y,z;x-y,x,z;-y,x-y,z;-x,-y,z;-x+y,-x,z;y,-x+y,z;x-y,-y,-z+1/2;-x,-x+y,-z+1/2;y,x,-z+1/2;-y,-x,-z+1/2;-x+y,y,-z+1/2;x,x-y,-z+1/2;\
-x,-y,-z;-x+y,-x,-z;y,-x+y,-z;x,y,-z;x-y,x,-z;-y,x-y,-z;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2;y,x,z+1/2;x-y,-y,z+1/2;-x,-x+y,z+1/2",
				"x,y,z;x-y,x,z+1/2;-y,x-y,z;-x,-y,z+1/2;-x+y,-x,z;y,-x+y,z+1/2;x-y,-y,-z+1/2;-x,-x+y,-z+1/2;y,x,-z+1/2;-y,-x,-z;-x+y,y,-z;x,x-y,-z;\
-x,-y,-z;-x+y,-x,-z+1/2;y,-x+y,-z;x,y,-z+1/2;x-y,x,-z;-y,x-y,-z+1/2;-x+y,y,z+1/2;x,x-y,z+1/2;-y,-x,z+1/2;y,x,z;x-y,-y,z;-x,-x+y,z",
				"x,y,z;x-y,x,z+1/2;-y,x-y,z;-x,-y,z+1/2;-x+y,-x,z;y,-x+y,z+1/2;x-y,-y,-z;-x,-x+y,-z;y,x,-z;-y,-x,-z+1/2;-x+y,y,-z+1/2;x,x-y,-z+1/2;\
-x,-y,-z;-x+y,-x,-z+1/2;y,-x+y,-z;x,y,-z+1/2;x-y,x,-z;-y,x-y,-z+1/2;-x+y,y,z;x,x-y,z;-y,-x,z;y,x,z+1/2;x-y,-y,z+1/2;-x,-x+y,z+1/2",
				"x,y,z;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-x,-y,z;x,-y,-z;-x,y,-z",
				"x,y,z;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-x,-y,z;x,-y,-z;-x,y,-z;x,y+1/2,z+1/2;z,x+1/2,y+1/2;\
y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;-x,-y+1/2,z+1/2;\
x,-y+1/2,-z+1/2;-x,y+1/2,-z+1/2;x+1/2,y,z+1/2;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;\
-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;-x+1/2,-y,z+1/2;x+1/2,-y,-z+1/2;-x+1/2,y,-z+1/2;x+1/2,y+1/2,z;z+1/2,x+1/2,y;\
y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;-x+1/2,-y+1/2,z;\
x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z",
				"x,y,z;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-x,-y,z;x,-y,-z;-x,y,-z;x+1/2,y+1/2,z+1/2;z+1/2,x+1/2,y+1/2;\
y+1/2,z+1/2,x+1/2;-y+1/2,-z+1/2,x+1/2;z+1/2,-x+1/2,-y+1/2;-y+1/2,z+1/2,-x+1/2;-z+1/2,-x+1/2,y+1/2;-z+1/2,x+1/2,-y+1/2;\
y+1/2,-z+1/2,-x+1/2;-x+1/2,-y+1/2,z+1/2;x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2",
				"x,y,z;z,x,y;y,z,x;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;-x+1/2,-y,z+1/2;\
x+1/2,-y+1/2,-z;-x,y+1/2,-z+1/2",
				"x,y,z;z,x,y;y,z,x;-y,-z+1/2,x;z,-x,-y+1/2;-y+1/2,z,-x;-z,-x+1/2,y;-z+1/2,x,-y;y,-z,-x+1/2;-x,-y+1/2,z;x,-y,-z+1/2;-x+1/2,y,-z;\
x+1/2,y+1/2,z+1/2;z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;\
-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;-x+1/2,-y,z+1/2;x+1/2,-y+1/2,-z;-x,y+1/2,-z+1/2",
				"x,y,z;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-x,-y,z;x,-y,-z;-x,y,-z;-x,-y,-z;-z,-x,-y;-y,-z,-x;y,z,-x;-z,x,y;\
y,-z,x;z,x,-y;z,-x,y;-y,z,x;x,y,-z;-x,y,z;x,-y,z",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-z+1/2,-x+1/2,-y+1/2;-y+1/2,-z+1/2,-x+1/2;\
y+1/2,z+1/2,-x+1/2;-z+1/2,x+1/2,y+1/2;y+1/2,-z+1/2,x+1/2;z+1/2,x+1/2,-y+1/2;z+1/2,-x+1/2,y+1/2;-y+1/2,z+1/2,x+1/2;-x,-y,z;x,-y,-z;\
-x,y,-z;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-x,-y,z;x,-y,-z;-x,y,-z;-x,-y,-z;-z,-x,-y;-y,-z,-x;y,z,-x;-z,x,y;\
y,-z,x;z,x,-y;z,-x,y;-y,z,x;x,y,-z;-x,y,z;x,-y,z;x,y+1/2,z+1/2;z,x+1/2,y+1/2;y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;z,-x+1/2,-y+1/2;\
-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;-x,-y+1/2,z+1/2;x,-y+1/2,-z+1/2;-x,y+1/2,-z+1/2;-x,-y+1/2,-z+1/2;\
-z,-x+1/2,-y+1/2;-y,-z+1/2,-x+1/2;y,z+1/2,-x+1/2;-z,x+1/2,y+1/2;y,-z+1/2,x+1/2;z,x+1/2,-y+1/2;z,-x+1/2,y+1/2;-y,z+1/2,x+1/2;\
x,y+1/2,-z+1/2;-x,y+1/2,z+1/2;x,-y+1/2,z+1/2;x+1/2,y,z+1/2;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x,-y+1/2;\
-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;-x+1/2,-y,z+1/2;x+1/2,-y,-z+1/2;-x+1/2,y,-z+1/2;-x+1/2,-y,-z+1/2;\
-z+1/2,-x,-y+1/2;-y+1/2,-z,-x+1/2;y+1/2,z,-x+1/2;-z+1/2,x,y+1/2;y+1/2,-z,x+1/2;z+1/2,x,-y+1/2;z+1/2,-x,y+1/2;-y+1/2,z,x+1/2;\
x+1/2,y,-z+1/2;-x+1/2,y,z+1/2;x+1/2,-y,z+1/2;x+1/2,y+1/2,z;z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;z+1/2,-x+1/2,-y;\
-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;-x+1/2,-y+1/2,z;x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;-x+1/2,-y+1/2,-z;\
-z+1/2,-x+1/2,-y;-y+1/2,-z+1/2,-x;y+1/2,z+1/2,-x;-z+1/2,x+1/2,y;y+1/2,-z+1/2,x;z+1/2,x+1/2,-y;z+1/2,-x+1/2,y;-y+1/2,z+1/2,x;\
x+1/2,y+1/2,-z;-x+1/2,y+1/2,z;x+1/2,-y+1/2,z",
				"x,y,z;-x+1/4,-y+1/4,-z+1/4;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-z+1/4,-x+1/4,-y+1/4;-y+1/4,-z+1/4,-x+1/4;\
y+1/4,z+1/4,-x+1/4;-z+1/4,x+1/4,y+1/4;y+1/4,-z+1/4,x+1/4;z+1/4,x+1/4,-y+1/4;z+1/4,-x+1/4,y+1/4;-y+1/4,z+1/4,x+1/4;-x,-y,z;x,-y,-z;\
-x,y,-z;x+1/4,y+1/4,-z+1/4;-x+1/4,y+1/4,z+1/4;x+1/4,-y+1/4,z+1/4;x,y+1/2,z+1/2;-x+1/4,-y+3/4,-z+3/4;z,x+1/2,y+1/2;y,z+1/2,x+1/2;\
-y,-z+1/2,x+1/2;z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;-z+1/4,-x+3/4,-y+3/4;\
-y+1/4,-z+3/4,-x+3/4;y+1/4,z+3/4,-x+3/4;-z+1/4,x+3/4,y+3/4;y+1/4,-z+3/4,x+3/4;z+1/4,x+3/4,-y+3/4;z+1/4,-x+3/4,y+3/4;\
-y+1/4,z+3/4,x+3/4;-x,-y+1/2,z+1/2;x,-y+1/2,-z+1/2;-x,y+1/2,-z+1/2;x+1/4,y+3/4,-z+3/4;-x+1/4,y+3/4,z+3/4;x+1/4,-y+3/4,z+3/4;\
x+1/2,y,z+1/2;-x+3/4,-y+1/4,-z+3/4;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;\
-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;-z+3/4,-x+1/4,-y+3/4;-y+3/4,-z+1/4,-x+3/4;y+3/4,z+1/4,-x+3/4;-z+3/4,x+1/4,y+3/4;y+3/4,-z+1/4,x+3/4;\
z+3/4,x+1/4,-y+3/4;z+3/4,-x+1/4,y+3/4;-y+3/4,z+1/4,x+3/4;-x+1/2,-y,z+1/2;x+1/2,-y,-z+1/2;-x+1/2,y,-z+1/2;x+3/4,y+1/4,-z+3/4;\
-x+3/4,y+1/4,z+3/4;x+3/4,-y+1/4,z+3/4;x+1/2,y+1/2,z;-x+3/4,-y+3/4,-z+1/4;z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;\
z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;-z+3/4,-x+3/4,-y+1/4;-y+3/4,-z+3/4,-x+1/4;\
y+3/4,z+3/4,-x+1/4;-z+3/4,x+3/4,y+1/4;y+3/4,-z+3/4,x+1/4;z+3/4,x+3/4,-y+1/4;z+3/4,-x+3/4,y+1/4;-y+3/4,z+3/4,x+1/4;-x+1/2,-y+1/2,z;\
x+1/2,-y+1/2,-z;-x+1/2,y+1/2,-z;x+3/4,y+3/4,-z+1/4;-x+3/4,y+3/4,z+1/4;x+3/4,-y+3/4,z+1/4",
				"x,y,z;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-x,-y,z;x,-y,-z;-x,y,-z;-x,-y,-z;-z,-x,-y;-y,-z,-x;y,z,-x;-z,x,y;\
y,-z,x;z,x,-y;z,-x,y;-y,z,x;x,y,-z;-x,y,z;x,-y,z;x+1/2,y+1/2,z+1/2;z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z+1/2,x+1/2;\
z+1/2,-x+1/2,-y+1/2;-y+1/2,z+1/2,-x+1/2;-z+1/2,-x+1/2,y+1/2;-z+1/2,x+1/2,-y+1/2;y+1/2,-z+1/2,-x+1/2;-x+1/2,-y+1/2,z+1/2;\
x+1/2,-y+1/2,-z+1/2;-x+1/2,y+1/2,-z+1/2;-x+1/2,-y+1/2,-z+1/2;-z+1/2,-x+1/2,-y+1/2;-y+1/2,-z+1/2,-x+1/2;y+1/2,z+1/2,-x+1/2;\
-z+1/2,x+1/2,y+1/2;y+1/2,-z+1/2,x+1/2;z+1/2,x+1/2,-y+1/2;z+1/2,-x+1/2,y+1/2;-y+1/2,z+1/2,x+1/2;x+1/2,y+1/2,-z+1/2;\
-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2",
				"x,y,z;z,x,y;y,z,x;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;-x+1/2,-y,z+1/2;\
x+1/2,-y+1/2,-z;-x,y+1/2,-z+1/2;-x,-y,-z;-z,-x,-y;-y,-z,-x;y+1/2,z,-x+1/2;-z+1/2,x+1/2,y;y,-z+1/2,x+1/2;z+1/2,x,-y+1/2;\
z,-x+1/2,y+1/2;-y+1/2,z+1/2,x;x+1/2,y,-z+1/2;-x+1/2,y+1/2,z;x,-y+1/2,z+1/2",
				"x,y,z;z,x,y;y,z,x;-y,-z+1/2,x;z,-x,-y+1/2;-y+1/2,z,-x;-z,-x+1/2,y;-z+1/2,x,-y;y,-z,-x+1/2;-x,-y+1/2,z;x,-y,-z+1/2;-x+1/2,y,-z;\
-x,-y,-z;-z,-x,-y;-y,-z,-x;y,z+1/2,-x;-z,x,y+1/2;y+1/2,-z,x;z,x+1/2,-y;z+1/2,-x,y;-y,z,x+1/2;x,y+1/2,-z;-x,y,z+1/2;x+1/2,-y,z;\
x+1/2,y+1/2,z+1/2;z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;\
-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;-x+1/2,-y,z+1/2;x+1/2,-y+1/2,-z;-x,y+1/2,-z+1/2;-x+1/2,-y+1/2,-z+1/2;-z+1/2,-x+1/2,-y+1/2;\
-y+1/2,-z+1/2,-x+1/2;y+1/2,z,-x+1/2;-z+1/2,x+1/2,y;y,-z+1/2,x+1/2;z+1/2,x,-y+1/2;z,-x+1/2,y+1/2;-y+1/2,z+1/2,x;x+1/2,y,-z+1/2;\
-x+1/2,y+1/2,z;x,-y+1/2,z+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-z,y;x,-y,-z;x,z,-y;z,y,-x;-x,y,-z;-z,y,x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;y,x,-z;-y,-x,-z;-x,z,y;-x,-z,-y;z,-y,x;-z,-y,-x",
				"x,y,z;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;x+1/2,-z+1/2,y+1/2;x,-y,-z;x+1/2,z+1/2,-y+1/2;z+1/2,y+1/2,-x+1/2;-x,y,-z;\
-z+1/2,y+1/2,x+1/2;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;\
-x+1/2,z+1/2,y+1/2;-x+1/2,-z+1/2,-y+1/2;z+1/2,-y+1/2,x+1/2;-z+1/2,-y+1/2,-x+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-z,y;x,-y,-z;x,z,-y;z,y,-x;-x,y,-z;-z,y,x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;y,x,-z;-y,-x,-z;-x,z,y;-x,-z,-y;z,-y,x;-z,-y,-x;x,y+1/2,z+1/2;-y,x+1/2,z+1/2;-x,-y+1/2,z+1/2;y,-x+1/2,z+1/2;x,-z+1/2,y+1/2;\
x,-y+1/2,-z+1/2;x,z+1/2,-y+1/2;z,y+1/2,-x+1/2;-x,y+1/2,-z+1/2;-z,y+1/2,x+1/2;z,x+1/2,y+1/2;y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;\
z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;y,x+1/2,-z+1/2;-y,-x+1/2,-z+1/2;-x,z+1/2,y+1/2;\
-x,-z+1/2,-y+1/2;z,-y+1/2,x+1/2;-z,-y+1/2,-x+1/2;x+1/2,y,z+1/2;-y+1/2,x,z+1/2;-x+1/2,-y,z+1/2;y+1/2,-x,z+1/2;x+1/2,-z,y+1/2;\
x+1/2,-y,-z+1/2;x+1/2,z,-y+1/2;z+1/2,y,-x+1/2;-x+1/2,y,-z+1/2;-z+1/2,y,x+1/2;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;\
z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;y+1/2,x,-z+1/2;-y+1/2,-x,-z+1/2;-x+1/2,z,y+1/2;\
-x+1/2,-z,-y+1/2;z+1/2,-y,x+1/2;-z+1/2,-y,-x+1/2;x+1/2,y+1/2,z;-y+1/2,x+1/2,z;-x+1/2,-y+1/2,z;y+1/2,-x+1/2,z;x+1/2,-z+1/2,y;\
x+1/2,-y+1/2,-z;x+1/2,z+1/2,-y;z+1/2,y+1/2,-x;-x+1/2,y+1/2,-z;-z+1/2,y+1/2,x;z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;\
z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;y+1/2,x+1/2,-z;-y+1/2,-x+1/2,-z;-x+1/2,z+1/2,y;\
-x+1/2,-z+1/2,-y;z+1/2,-y+1/2,x;-z+1/2,-y+1/2,-x",
				"x,y,z;-y+1/4,x+1/4,z+1/4;-x,-y,z;y+1/4,-x+1/4,z+1/4;x+1/4,-z+1/4,y+1/4;x,-y,-z;x+1/4,z+1/4,-y+1/4;z+1/4,y+1/4,-x+1/4;-x,y,-z;\
-z+1/4,y+1/4,x+1/4;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;y+1/4,x+1/4,-z+1/4;-y+1/4,-x+1/4,-z+1/4;\
-x+1/4,z+1/4,y+1/4;-x+1/4,-z+1/4,-y+1/4;z+1/4,-y+1/4,x+1/4;-z+1/4,-y+1/4,-x+1/4;x,y+1/2,z+1/2;-y+1/4,x+3/4,z+3/4;-x,-y+1/2,z+1/2;\
y+1/4,-x+3/4,z+3/4;x+1/4,-z+3/4,y+3/4;x,-y+1/2,-z+1/2;x+1/4,z+3/4,-y+3/4;z+1/4,y+3/4,-x+3/4;-x,y+1/2,-z+1/2;-z+1/4,y+3/4,x+3/4;\
z,x+1/2,y+1/2;y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;\
y+1/4,x+3/4,-z+3/4;-y+1/4,-x+3/4,-z+3/4;-x+1/4,z+3/4,y+3/4;-x+1/4,-z+3/4,-y+3/4;z+1/4,-y+3/4,x+3/4;-z+1/4,-y+3/4,-x+3/4;\
x+1/2,y,z+1/2;-y+3/4,x+1/4,z+3/4;-x+1/2,-y,z+1/2;y+3/4,-x+1/4,z+3/4;x+3/4,-z+1/4,y+3/4;x+1/2,-y,-z+1/2;x+3/4,z+1/4,-y+3/4;\
z+3/4,y+1/4,-x+3/4;-x+1/2,y,-z+1/2;-z+3/4,y+1/4,x+3/4;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;\
-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;y+3/4,x+1/4,-z+3/4;-y+3/4,-x+1/4,-z+3/4;-x+3/4,z+1/4,y+3/4;-x+3/4,-z+1/4,-y+3/4;\
z+3/4,-y+1/4,x+3/4;-z+3/4,-y+1/4,-x+3/4;x+1/2,y+1/2,z;-y+3/4,x+3/4,z+1/4;-x+1/2,-y+1/2,z;y+3/4,-x+3/4,z+1/4;x+3/4,-z+3/4,y+1/4;\
x+1/2,-y+1/2,-z;x+3/4,z+3/4,-y+1/4;z+3/4,y+3/4,-x+1/4;-x+1/2,y+1/2,-z;-z+3/4,y+3/4,x+1/4;z+1/2,x+1/2,y;y+1/2,z+1/2,x;\
-y+1/2,-z+1/2,x;z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;y+3/4,x+3/4,-z+1/4;\
-y+3/4,-x+3/4,-z+1/4;-x+3/4,z+3/4,y+1/4;-x+3/4,-z+3/4,-y+1/4;z+3/4,-y+3/4,x+1/4;-z+3/4,-y+3/4,-x+1/4",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-z,y;x,-y,-z;x,z,-y;z,y,-x;-x,y,-z;-z,y,x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;y,x,-z;-y,-x,-z;-x,z,y;-x,-z,-y;z,-y,x;-z,-y,-x;x+1/2,y+1/2,z+1/2;-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;\
y+1/2,-x+1/2,z+1/2;x+1/2,-z+1/2,y+1/2;x+1/2,-y+1/2,-z+1/2;x+1/2,z+1/2,-y+1/2;z+1/2,y+1/2,-x+1/2;-x+1/2,y+1/2,-z+1/2;\
-z+1/2,y+1/2,x+1/2;z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z+1/2,x+1/2;z+1/2,-x+1/2,-y+1/2;-y+1/2,z+1/2,-x+1/2;\
-z+1/2,-x+1/2,y+1/2;-z+1/2,x+1/2,-y+1/2;y+1/2,-z+1/2,-x+1/2;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;-x+1/2,z+1/2,y+1/2;\
-x+1/2,-z+1/2,-y+1/2;z+1/2,-y+1/2,x+1/2;-z+1/2,-y+1/2,-x+1/2",
				"x,y,z;-y+3/4,x+1/4,z+3/4;-x+1/2,-y,z+1/2;y+3/4,-x+3/4,z+1/4;x+3/4,-z+3/4,y+1/4;x+1/2,-y+1/2,-z;x+1/4,z+3/4,-y+3/4;\
z+1/4,y+3/4,-x+3/4;-x,y+1/2,-z+1/2;-z+3/4,y+1/4,x+3/4;z,x,y;y,z,x;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;\
-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;y+1/4,x+3/4,-z+3/4;-y+1/4,-x+1/4,-z+1/4;-x+3/4,z+1/4,y+3/4;-x+1/4,-z+1/4,-y+1/4;z+3/4,-y+3/4,x+1/4;\
-z+1/4,-y+1/4,-x+1/4",
				"x,y,z;-y+1/4,x+3/4,z+1/4;-x+1/2,-y,z+1/2;y+1/4,-x+1/4,z+3/4;x+1/4,-z+1/4,y+3/4;x+1/2,-y+1/2,-z;x+3/4,z+1/4,-y+1/4;\
z+3/4,y+1/4,-x+1/4;-x,y+1/2,-z+1/2;-z+1/4,y+3/4,x+1/4;z,x,y;y,z,x;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;\
-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;y+3/4,x+1/4,-z+1/4;-y+3/4,-x+3/4,-z+3/4;-x+1/4,z+3/4,y+1/4;-x+3/4,-z+3/4,-y+3/4;z+1/4,-y+1/4,x+3/4;\
-z+3/4,-y+3/4,-x+3/4",
				"x,y,z;-y+1/4,x+3/4,z+1/4;-x,-y+1/2,z;y+1/4,-x+1/4,z+3/4;x+1/4,-z+1/4,y+3/4;x,-y,-z+1/2;x+3/4,z+1/4,-y+1/4;z+3/4,y+1/4,-x+1/4;\
-x+1/2,y,-z;-z+1/4,y+3/4,x+1/4;z,x,y;y,z,x;-y,-z+1/2,x;z,-x,-y+1/2;-y+1/2,z,-x;-z,-x+1/2,y;-z+1/2,x,-y;y,-z,-x+1/2;\
y+3/4,x+1/4,-z+1/4;-y+1/4,-x+1/4,-z+1/4;-x+1/4,z+3/4,y+1/4;-x+1/4,-z+1/4,-y+1/4;z+1/4,-y+1/4,x+3/4;-z+1/4,-y+1/4,-x+1/4;\
x+1/2,y+1/2,z+1/2;-y+3/4,x+1/4,z+3/4;-x+1/2,-y,z+1/2;y+3/4,-x+3/4,z+1/4;x+3/4,-z+3/4,y+1/4;x+1/2,-y+1/2,-z;x+1/4,z+3/4,-y+3/4;\
z+1/4,y+3/4,-x+3/4;-x,y+1/2,-z+1/2;-z+3/4,y+1/4,x+3/4;z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;\
-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;y+1/4,x+3/4,-z+3/4;-y+3/4,-x+3/4,-z+3/4;-x+3/4,z+1/4,y+3/4;\
-x+3/4,-z+3/4,-y+3/4;z+3/4,-y+3/4,x+1/4;-z+3/4,-y+3/4,-x+3/4",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;-x,z,-y;x,-y,-z;-x,-z,y;-z,-y,x;-x,y,-z;z,-y,-x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;-y,-x,z;y,x,z;x,-z,-y;x,z,y;-z,y,-x;z,y,x",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;-x,z,-y;x,-y,-z;-x,-z,y;-z,-y,x;-x,y,-z;z,-y,-x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;-y,-x,z;y,x,z;x,-z,-y;x,z,y;-z,y,-x;z,y,x;x,y+1/2,z+1/2;y,-x+1/2,-z+1/2;-x,-y+1/2,z+1/2;-y,x+1/2,-z+1/2;-x,z+1/2,-y+1/2;\
x,-y+1/2,-z+1/2;-x,-z+1/2,y+1/2;-z,-y+1/2,x+1/2;-x,y+1/2,-z+1/2;z,-y+1/2,-x+1/2;z,x+1/2,y+1/2;y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;\
z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;-y,-x+1/2,z+1/2;y,x+1/2,z+1/2;x,-z+1/2,-y+1/2;\
x,z+1/2,y+1/2;-z,y+1/2,-x+1/2;z,y+1/2,x+1/2;x+1/2,y,z+1/2;y+1/2,-x,-z+1/2;-x+1/2,-y,z+1/2;-y+1/2,x,-z+1/2;-x+1/2,z,-y+1/2;\
x+1/2,-y,-z+1/2;-x+1/2,-z,y+1/2;-z+1/2,-y,x+1/2;-x+1/2,y,-z+1/2;z+1/2,-y,-x+1/2;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;\
z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;-y+1/2,-x,z+1/2;y+1/2,x,z+1/2;x+1/2,-z,-y+1/2;\
x+1/2,z,y+1/2;-z+1/2,y,-x+1/2;z+1/2,y,x+1/2;x+1/2,y+1/2,z;y+1/2,-x+1/2,-z;-x+1/2,-y+1/2,z;-y+1/2,x+1/2,-z;-x+1/2,z+1/2,-y;\
x+1/2,-y+1/2,-z;-x+1/2,-z+1/2,y;-z+1/2,-y+1/2,x;-x+1/2,y+1/2,-z;z+1/2,-y+1/2,-x;z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;\
z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z;x+1/2,-z+1/2,-y;\
x+1/2,z+1/2,y;-z+1/2,y+1/2,-x;z+1/2,y+1/2,x",
				"x,y,z;y,-x,-z;-x,-y,z;-y,x,-z;-x,z,-y;x,-y,-z;-x,-z,y;-z,-y,x;-x,y,-z;z,-y,-x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;-y,-x,z;y,x,z;x,-z,-y;x,z,y;-z,y,-x;z,y,x;x+1/2,y+1/2,z+1/2;y+1/2,-x+1/2,-z+1/2;-x+1/2,-y+1/2,z+1/2;-y+1/2,x+1/2,-z+1/2;\
-x+1/2,z+1/2,-y+1/2;x+1/2,-y+1/2,-z+1/2;-x+1/2,-z+1/2,y+1/2;-z+1/2,-y+1/2,x+1/2;-x+1/2,y+1/2,-z+1/2;z+1/2,-y+1/2,-x+1/2;\
z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z+1/2,x+1/2;z+1/2,-x+1/2,-y+1/2;-y+1/2,z+1/2,-x+1/2;-z+1/2,-x+1/2,y+1/2;\
-z+1/2,x+1/2,-y+1/2;y+1/2,-z+1/2,-x+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2;x+1/2,-z+1/2,-y+1/2;x+1/2,z+1/2,y+1/2;\
-z+1/2,y+1/2,-x+1/2;z+1/2,y+1/2,x+1/2",
				"x,y,z;y+1/2,-x+1/2,-z+1/2;-x,-y,z;-y+1/2,x+1/2,-z+1/2;-x+1/2,z+1/2,-y+1/2;x,-y,-z;-x+1/2,-z+1/2,y+1/2;-z+1/2,-y+1/2,x+1/2;-x,y,-z;\
z+1/2,-y+1/2,-x+1/2;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2;\
x+1/2,-z+1/2,-y+1/2;x+1/2,z+1/2,y+1/2;-z+1/2,y+1/2,-x+1/2;z+1/2,y+1/2,x+1/2",
				"x,y,z;y,-x,-z+1/2;-x,-y,z;-y,x,-z+1/2;-x,z,-y+1/2;x,-y,-z;-x,-z,y+1/2;-z,-y,x+1/2;-x,y,-z;z,-y,-x+1/2;z,x,y;y,z,x;-y,-z,x;z,-x,-y;\
-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-y,-x,z+1/2;y,x,z+1/2;x,-z,-y+1/2;x,z,y+1/2;-z,y,-x+1/2;z,y,x+1/2;x,y+1/2,z+1/2;y,-x+1/2,-z;\
-x,-y+1/2,z+1/2;-y,x+1/2,-z;-x,z+1/2,-y;x,-y+1/2,-z+1/2;-x,-z+1/2,y;-z,-y+1/2,x;-x,y+1/2,-z+1/2;z,-y+1/2,-x;z,x+1/2,y+1/2;\
y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;-y,-x+1/2,z;\
y,x+1/2,z;x,-z+1/2,-y;x,z+1/2,y;-z,y+1/2,-x;z,y+1/2,x;x+1/2,y,z+1/2;y+1/2,-x,-z;-x+1/2,-y,z+1/2;-y+1/2,x,-z;-x+1/2,z,-y;\
x+1/2,-y,-z+1/2;-x+1/2,-z,y;-z+1/2,-y,x;-x+1/2,y,-z+1/2;z+1/2,-y,-x;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x,-y+1/2;\
-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;-y+1/2,-x,z;y+1/2,x,z;x+1/2,-z,-y;x+1/2,z,y;-z+1/2,y,-x;z+1/2,y,x;\
x+1/2,y+1/2,z;y+1/2,-x+1/2,-z+1/2;-x+1/2,-y+1/2,z;-y+1/2,x+1/2,-z+1/2;-x+1/2,z+1/2,-y+1/2;x+1/2,-y+1/2,-z;-x+1/2,-z+1/2,y+1/2;\
-z+1/2,-y+1/2,x+1/2;-x+1/2,y+1/2,-z;z+1/2,-y+1/2,-x+1/2;z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;z+1/2,-x+1/2,-y;\
-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2;x+1/2,-z+1/2,-y+1/2;\
x+1/2,z+1/2,y+1/2;-z+1/2,y+1/2,-x+1/2;z+1/2,y+1/2,x+1/2",
				"x,y,z;y+1/4,-x+3/4,-z+1/4;-x,-y+1/2,z;-y+1/4,x+1/4,-z+3/4;-x+1/4,z+1/4,-y+3/4;x,-y,-z+1/2;-x+3/4,-z+1/4,y+1/4;-z+3/4,-y+1/4,x+1/4;\
-x+1/2,y,-z;z+1/4,-y+3/4,-x+1/4;z,x,y;y,z,x;-y,-z+1/2,x;z,-x,-y+1/2;-y+1/2,z,-x;-z,-x+1/2,y;-z+1/2,x,-y;y,-z,-x+1/2;\
-y+3/4,-x+1/4,z+1/4;y+1/4,x+1/4,z+1/4;x+1/4,-z+3/4,-y+1/4;x+1/4,z+1/4,y+1/4;-z+1/4,y+1/4,-x+3/4;z+1/4,y+1/4,x+1/4;\
x+1/2,y+1/2,z+1/2;y+3/4,-x+1/4,-z+3/4;-x+1/2,-y,z+1/2;-y+3/4,x+3/4,-z+1/4;-x+3/4,z+3/4,-y+1/4;x+1/2,-y+1/2,-z;-x+1/4,-z+3/4,y+3/4;\
-z+1/4,-y+3/4,x+3/4;-x,y+1/2,-z+1/2;z+3/4,-y+1/4,-x+3/4;z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;\
-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;-y+1/4,-x+3/4,z+3/4;y+3/4,x+3/4,z+3/4;x+3/4,-z+1/4,-y+3/4;\
x+3/4,z+3/4,y+3/4;-z+3/4,y+3/4,-x+1/4;z+3/4,y+3/4,x+3/4",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-z,y;x,-y,-z;x,z,-y;z,y,-x;-x,y,-z;-z,y,x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;y,x,-z;-y,-x,-z;-x,z,y;-x,-z,-y;z,-y,x;-z,-y,-x;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;-x,z,-y;-x,y,z;-x,-z,y;-z,-y,x;x,-y,z;\
z,-y,-x;-z,-x,-y;-y,-z,-x;y,z,-x;-z,x,y;y,-z,x;z,x,-y;z,-x,y;-y,z,x;-y,-x,z;y,x,z;x,-z,-y;x,z,y;-z,y,-x;z,y,x",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-y,x,z;-x,-y,z;y,-x,z;x,-z,y;x,-y,-z;x,z,-y;z,y,-x;-x,y,-z;-z,y,x;y+1/2,-x+1/2,-z+1/2;\
-y+1/2,x+1/2,-z+1/2;-x+1/2,z+1/2,-y+1/2;-x+1/2,-z+1/2,y+1/2;-z+1/2,-y+1/2,x+1/2;z+1/2,-y+1/2,-x+1/2;z,x,y;y,z,x;-y,-z,x;z,-x,-y;\
-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-z+1/2,-x+1/2,-y+1/2;-y+1/2,-z+1/2,-x+1/2;y+1/2,z+1/2,-x+1/2;-z+1/2,x+1/2,y+1/2;y+1/2,-z+1/2,x+1/2;\
z+1/2,x+1/2,-y+1/2;z+1/2,-x+1/2,y+1/2;-y+1/2,z+1/2,x+1/2;y,x,-z;-y,-x,-z;-x,z,y;-x,-z,-y;z,-y,x;-z,-y,-x;x+1/2,y+1/2,-z+1/2;\
-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2;x+1/2,-z+1/2,-y+1/2;x+1/2,z+1/2,y+1/2;\
-z+1/2,y+1/2,-x+1/2;z+1/2,y+1/2,x+1/2",
				"x,y,z;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;x+1/2,-z+1/2,y+1/2;x,-y,-z;x+1/2,z+1/2,-y+1/2;z+1/2,y+1/2,-x+1/2;-x,y,-z;\
-z+1/2,y+1/2,x+1/2;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;\
-x+1/2,z+1/2,y+1/2;-x+1/2,-z+1/2,-y+1/2;z+1/2,-y+1/2,x+1/2;-z+1/2,-y+1/2,-x+1/2;-x,-y,-z;y+1/2,-x+1/2,-z+1/2;x,y,-z;\
-y+1/2,x+1/2,-z+1/2;-x+1/2,z+1/2,-y+1/2;-x,y,z;-x+1/2,-z+1/2,y+1/2;-z+1/2,-y+1/2,x+1/2;x,-y,z;z+1/2,-y+1/2,-x+1/2;-z,-x,-y;\
-y,-z,-x;y,z,-x;-z,x,y;y,-z,x;z,x,-y;z,-x,y;-y,z,x;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2;x+1/2,-z+1/2,-y+1/2;x+1/2,z+1/2,y+1/2;\
-z+1/2,y+1/2,-x+1/2;z+1/2,y+1/2,x+1/2",
				"x,y,z;-x+1/2,-y+1/2,-z+1/2;-y+1/2,x+1/2,z+1/2;-x,-y,z;y+1/2,-x+1/2,z+1/2;x+1/2,-z+1/2,y+1/2;x,-y,-z;x+1/2,z+1/2,-y+1/2;\
z+1/2,y+1/2,-x+1/2;-x,y,-z;-z+1/2,y+1/2,x+1/2;y,-x,-z;-y,x,-z;-x,z,-y;-x,-z,y;-z,-y,x;z,-y,-x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;\
-z,-x,y;-z,x,-y;y,-z,-x;-z+1/2,-x+1/2,-y+1/2;-y+1/2,-z+1/2,-x+1/2;y+1/2,z+1/2,-x+1/2;-z+1/2,x+1/2,y+1/2;y+1/2,-z+1/2,x+1/2;\
z+1/2,x+1/2,-y+1/2;z+1/2,-x+1/2,y+1/2;-y+1/2,z+1/2,x+1/2;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;-x+1/2,z+1/2,y+1/2;\
-x+1/2,-z+1/2,-y+1/2;z+1/2,-y+1/2,x+1/2;-z+1/2,-y+1/2,-x+1/2;x+1/2,y+1/2,-z+1/2;-x+1/2,y+1/2,z+1/2;x+1/2,-y+1/2,z+1/2;-y,-x,z;\
y,x,z;x,-z,-y;x,z,y;-z,y,-x;z,y,x",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-z,y;x,-y,-z;x,z,-y;z,y,-x;-x,y,-z;-z,y,x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;y,x,-z;-y,-x,-z;-x,z,y;-x,-z,-y;z,-y,x;-z,-y,-x;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;-x,z,-y;-x,y,z;-x,-z,y;-z,-y,x;x,-y,z;\
z,-y,-x;-z,-x,-y;-y,-z,-x;y,z,-x;-z,x,y;y,-z,x;z,x,-y;z,-x,y;-y,z,x;-y,-x,z;y,x,z;x,-z,-y;x,z,y;-z,y,-x;z,y,x;x,y+1/2,z+1/2;\
-y,x+1/2,z+1/2;-x,-y+1/2,z+1/2;y,-x+1/2,z+1/2;x,-z+1/2,y+1/2;x,-y+1/2,-z+1/2;x,z+1/2,-y+1/2;z,y+1/2,-x+1/2;-x,y+1/2,-z+1/2;\
-z,y+1/2,x+1/2;z,x+1/2,y+1/2;y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;\
y,-z+1/2,-x+1/2;y,x+1/2,-z+1/2;-y,-x+1/2,-z+1/2;-x,z+1/2,y+1/2;-x,-z+1/2,-y+1/2;z,-y+1/2,x+1/2;-z,-y+1/2,-x+1/2;-x,-y+1/2,-z+1/2;\
y,-x+1/2,-z+1/2;x,y+1/2,-z+1/2;-y,x+1/2,-z+1/2;-x,z+1/2,-y+1/2;-x,y+1/2,z+1/2;-x,-z+1/2,y+1/2;-z,-y+1/2,x+1/2;x,-y+1/2,z+1/2;\
z,-y+1/2,-x+1/2;-z,-x+1/2,-y+1/2;-y,-z+1/2,-x+1/2;y,z+1/2,-x+1/2;-z,x+1/2,y+1/2;y,-z+1/2,x+1/2;z,x+1/2,-y+1/2;z,-x+1/2,y+1/2;\
-y,z+1/2,x+1/2;-y,-x+1/2,z+1/2;y,x+1/2,z+1/2;x,-z+1/2,-y+1/2;x,z+1/2,y+1/2;-z,y+1/2,-x+1/2;z,y+1/2,x+1/2;x+1/2,y,z+1/2;\
-y+1/2,x,z+1/2;-x+1/2,-y,z+1/2;y+1/2,-x,z+1/2;x+1/2,-z,y+1/2;x+1/2,-y,-z+1/2;x+1/2,z,-y+1/2;z+1/2,y,-x+1/2;-x+1/2,y,-z+1/2;\
-z+1/2,y,x+1/2;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;\
y+1/2,-z,-x+1/2;y+1/2,x,-z+1/2;-y+1/2,-x,-z+1/2;-x+1/2,z,y+1/2;-x+1/2,-z,-y+1/2;z+1/2,-y,x+1/2;-z+1/2,-y,-x+1/2;-x+1/2,-y,-z+1/2;\
y+1/2,-x,-z+1/2;x+1/2,y,-z+1/2;-y+1/2,x,-z+1/2;-x+1/2,z,-y+1/2;-x+1/2,y,z+1/2;-x+1/2,-z,y+1/2;-z+1/2,-y,x+1/2;x+1/2,-y,z+1/2;\
z+1/2,-y,-x+1/2;-z+1/2,-x,-y+1/2;-y+1/2,-z,-x+1/2;y+1/2,z,-x+1/2;-z+1/2,x,y+1/2;y+1/2,-z,x+1/2;z+1/2,x,-y+1/2;z+1/2,-x,y+1/2;\
-y+1/2,z,x+1/2;-y+1/2,-x,z+1/2;y+1/2,x,z+1/2;x+1/2,-z,-y+1/2;x+1/2,z,y+1/2;-z+1/2,y,-x+1/2;z+1/2,y,x+1/2;x+1/2,y+1/2,z;\
-y+1/2,x+1/2,z;-x+1/2,-y+1/2,z;y+1/2,-x+1/2,z;x+1/2,-z+1/2,y;x+1/2,-y+1/2,-z;x+1/2,z+1/2,-y;z+1/2,y+1/2,-x;-x+1/2,y+1/2,-z;\
-z+1/2,y+1/2,x;z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;\
y+1/2,-z+1/2,-x;y+1/2,x+1/2,-z;-y+1/2,-x+1/2,-z;-x+1/2,z+1/2,y;-x+1/2,-z+1/2,-y;z+1/2,-y+1/2,x;-z+1/2,-y+1/2,-x;-x+1/2,-y+1/2,-z;\
y+1/2,-x+1/2,-z;x+1/2,y+1/2,-z;-y+1/2,x+1/2,-z;-x+1/2,z+1/2,-y;-x+1/2,y+1/2,z;-x+1/2,-z+1/2,y;-z+1/2,-y+1/2,x;x+1/2,-y+1/2,z;\
z+1/2,-y+1/2,-x;-z+1/2,-x+1/2,-y;-y+1/2,-z+1/2,-x;y+1/2,z+1/2,-x;-z+1/2,x+1/2,y;y+1/2,-z+1/2,x;z+1/2,x+1/2,-y;z+1/2,-x+1/2,y;\
-y+1/2,z+1/2,x;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z;x+1/2,-z+1/2,-y;x+1/2,z+1/2,y;-z+1/2,y+1/2,-x;z+1/2,y+1/2,x",
				"x,y,z;-y,x,z+1/2;-x,-y,z;y,-x,z+1/2;x,-z,y+1/2;x,-y,-z;x,z,-y+1/2;z,y,-x+1/2;-x,y,-z;-z,y,x+1/2;z,x,y;y,z,x;-y,-z,x;z,-x,-y;\
-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;y,x,-z+1/2;-y,-x,-z+1/2;-x,z,y+1/2;-x,-z,-y+1/2;z,-y,x+1/2;-z,-y,-x+1/2;-x,-y,-z;y,-x,-z+1/2;\
x,y,-z;-y,x,-z+1/2;-x,z,-y+1/2;-x,y,z;-x,-z,y+1/2;-z,-y,x+1/2;x,-y,z;z,-y,-x+1/2;-z,-x,-y;-y,-z,-x;y,z,-x;-z,x,y;y,-z,x;z,x,-y;\
z,-x,y;-y,z,x;-y,-x,z+1/2;y,x,z+1/2;x,-z,-y+1/2;x,z,y+1/2;-z,y,-x+1/2;z,y,x+1/2;x,y+1/2,z+1/2;-y,x+1/2,z;-x,-y+1/2,z+1/2;\
y,-x+1/2,z;x,-z+1/2,y;x,-y+1/2,-z+1/2;x,z+1/2,-y;z,y+1/2,-x;-x,y+1/2,-z+1/2;-z,y+1/2,x;z,x+1/2,y+1/2;y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;\
z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;y,x+1/2,-z;-y,-x+1/2,-z;-x,z+1/2,y;-x,-z+1/2,-y;\
z,-y+1/2,x;-z,-y+1/2,-x;-x,-y+1/2,-z+1/2;y,-x+1/2,-z;x,y+1/2,-z+1/2;-y,x+1/2,-z;-x,z+1/2,-y;-x,y+1/2,z+1/2;-x,-z+1/2,y;-z,-y+1/2,x;\
x,-y+1/2,z+1/2;z,-y+1/2,-x;-z,-x+1/2,-y+1/2;-y,-z+1/2,-x+1/2;y,z+1/2,-x+1/2;-z,x+1/2,y+1/2;y,-z+1/2,x+1/2;z,x+1/2,-y+1/2;\
z,-x+1/2,y+1/2;-y,z+1/2,x+1/2;-y,-x+1/2,z;y,x+1/2,z;x,-z+1/2,-y;x,z+1/2,y;-z,y+1/2,-x;z,y+1/2,x;x+1/2,y,z+1/2;-y+1/2,x,z;\
-x+1/2,-y,z+1/2;y+1/2,-x,z;x+1/2,-z,y;x+1/2,-y,-z+1/2;x+1/2,z,-y;z+1/2,y,-x;-x+1/2,y,-z+1/2;-z+1/2,y,x;z+1/2,x,y+1/2;y+1/2,z,x+1/2;\
-y+1/2,-z,x+1/2;z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;y+1/2,x,-z;-y+1/2,-x,-z;-x+1/2,z,y;\
-x+1/2,-z,-y;z+1/2,-y,x;-z+1/2,-y,-x;-x+1/2,-y,-z+1/2;y+1/2,-x,-z;x+1/2,y,-z+1/2;-y+1/2,x,-z;-x+1/2,z,-y;-x+1/2,y,z+1/2;\
-x+1/2,-z,y;-z+1/2,-y,x;x+1/2,-y,z+1/2;z+1/2,-y,-x;-z+1/2,-x,-y+1/2;-y+1/2,-z,-x+1/2;y+1/2,z,-x+1/2;-z+1/2,x,y+1/2;y+1/2,-z,x+1/2;\
z+1/2,x,-y+1/2;z+1/2,-x,y+1/2;-y+1/2,z,x+1/2;-y+1/2,-x,z;y+1/2,x,z;x+1/2,-z,-y;x+1/2,z,y;-z+1/2,y,-x;z+1/2,y,x;x+1/2,y+1/2,z;\
-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z;y+1/2,-x+1/2,z+1/2;x+1/2,-z+1/2,y+1/2;x+1/2,-y+1/2,-z;x+1/2,z+1/2,-y+1/2;z+1/2,y+1/2,-x+1/2;\
-x+1/2,y+1/2,-z;-z+1/2,y+1/2,x+1/2;z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;\
-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;y+1/2,x+1/2,-z+1/2;-y+1/2,-x+1/2,-z+1/2;-x+1/2,z+1/2,y+1/2;-x+1/2,-z+1/2,-y+1/2;z+1/2,-y+1/2,x+1/2;\
-z+1/2,-y+1/2,-x+1/2;-x+1/2,-y+1/2,-z;y+1/2,-x+1/2,-z+1/2;x+1/2,y+1/2,-z;-y+1/2,x+1/2,-z+1/2;-x+1/2,z+1/2,-y+1/2;-x+1/2,y+1/2,z;\
-x+1/2,-z+1/2,y+1/2;-z+1/2,-y+1/2,x+1/2;x+1/2,-y+1/2,z;z+1/2,-y+1/2,-x+1/2;-z+1/2,-x+1/2,-y;-y+1/2,-z+1/2,-x;y+1/2,z+1/2,-x;\
-z+1/2,x+1/2,y;y+1/2,-z+1/2,x;z+1/2,x+1/2,-y;z+1/2,-x+1/2,y;-y+1/2,z+1/2,x;-y+1/2,-x+1/2,z+1/2;y+1/2,x+1/2,z+1/2;\
x+1/2,-z+1/2,-y+1/2;x+1/2,z+1/2,y+1/2;-z+1/2,y+1/2,-x+1/2;z+1/2,y+1/2,x+1/2",
				"x,y,z;-x+1/4,-y+1/4,-z+1/4;-y+1/4,x+1/4,z+1/4;-x,-y,z;y+1/4,-x+1/4,z+1/4;x+1/4,-z+1/4,y+1/4;x,-y,-z;x+1/4,z+1/4,-y+1/4;\
z+1/4,y+1/4,-x+1/4;-x,y,-z;-z+1/4,y+1/4,x+1/4;y,-x,-z;-y,x,-z;-x,z,-y;-x,-z,y;-z,-y,x;z,-y,-x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;\
-z,-x,y;-z,x,-y;y,-z,-x;-z+1/4,-x+1/4,-y+1/4;-y+1/4,-z+1/4,-x+1/4;y+1/4,z+1/4,-x+1/4;-z+1/4,x+1/4,y+1/4;y+1/4,-z+1/4,x+1/4;\
z+1/4,x+1/4,-y+1/4;z+1/4,-x+1/4,y+1/4;-y+1/4,z+1/4,x+1/4;y+1/4,x+1/4,-z+1/4;-y+1/4,-x+1/4,-z+1/4;-x+1/4,z+1/4,y+1/4;\
-x+1/4,-z+1/4,-y+1/4;z+1/4,-y+1/4,x+1/4;-z+1/4,-y+1/4,-x+1/4;x+1/4,y+1/4,-z+1/4;-x+1/4,y+1/4,z+1/4;x+1/4,-y+1/4,z+1/4;-y,-x,z;\
y,x,z;x,-z,-y;x,z,y;-z,y,-x;z,y,x;x,y+1/2,z+1/2;-x+1/4,-y+3/4,-z+3/4;-y+1/4,x+3/4,z+3/4;-x,-y+1/2,z+1/2;y+1/4,-x+3/4,z+3/4;\
x+1/4,-z+3/4,y+3/4;x,-y+1/2,-z+1/2;x+1/4,z+3/4,-y+3/4;z+1/4,y+3/4,-x+3/4;-x,y+1/2,-z+1/2;-z+1/4,y+3/4,x+3/4;y,-x+1/2,-z+1/2;\
-y,x+1/2,-z+1/2;-x,z+1/2,-y+1/2;-x,-z+1/2,y+1/2;-z,-y+1/2,x+1/2;z,-y+1/2,-x+1/2;z,x+1/2,y+1/2;y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;\
z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;-z+1/4,-x+3/4,-y+3/4;-y+1/4,-z+3/4,-x+3/4;\
y+1/4,z+3/4,-x+3/4;-z+1/4,x+3/4,y+3/4;y+1/4,-z+3/4,x+3/4;z+1/4,x+3/4,-y+3/4;z+1/4,-x+3/4,y+3/4;-y+1/4,z+3/4,x+3/4;\
y+1/4,x+3/4,-z+3/4;-y+1/4,-x+3/4,-z+3/4;-x+1/4,z+3/4,y+3/4;-x+1/4,-z+3/4,-y+3/4;z+1/4,-y+3/4,x+3/4;-z+1/4,-y+3/4,-x+3/4;\
x+1/4,y+3/4,-z+3/4;-x+1/4,y+3/4,z+3/4;x+1/4,-y+3/4,z+3/4;-y,-x+1/2,z+1/2;y,x+1/2,z+1/2;x,-z+1/2,-y+1/2;x,z+1/2,y+1/2;\
-z,y+1/2,-x+1/2;z,y+1/2,x+1/2;x+1/2,y,z+1/2;-x+3/4,-y+1/4,-z+3/4;-y+3/4,x+1/4,z+3/4;-x+1/2,-y,z+1/2;y+3/4,-x+1/4,z+3/4;\
x+3/4,-z+1/4,y+3/4;x+1/2,-y,-z+1/2;x+3/4,z+1/4,-y+3/4;z+3/4,y+1/4,-x+3/4;-x+1/2,y,-z+1/2;-z+3/4,y+1/4,x+3/4;y+1/2,-x,-z+1/2;\
-y+1/2,x,-z+1/2;-x+1/2,z,-y+1/2;-x+1/2,-z,y+1/2;-z+1/2,-y,x+1/2;z+1/2,-y,-x+1/2;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;\
z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;-z+3/4,-x+1/4,-y+3/4;-y+3/4,-z+1/4,-x+3/4;\
y+3/4,z+1/4,-x+3/4;-z+3/4,x+1/4,y+3/4;y+3/4,-z+1/4,x+3/4;z+3/4,x+1/4,-y+3/4;z+3/4,-x+1/4,y+3/4;-y+3/4,z+1/4,x+3/4;\
y+3/4,x+1/4,-z+3/4;-y+3/4,-x+1/4,-z+3/4;-x+3/4,z+1/4,y+3/4;-x+3/4,-z+1/4,-y+3/4;z+3/4,-y+1/4,x+3/4;-z+3/4,-y+1/4,-x+3/4;\
x+3/4,y+1/4,-z+3/4;-x+3/4,y+1/4,z+3/4;x+3/4,-y+1/4,z+3/4;-y+1/2,-x,z+1/2;y+1/2,x,z+1/2;x+1/2,-z,-y+1/2;x+1/2,z,y+1/2;\
-z+1/2,y,-x+1/2;z+1/2,y,x+1/2;x+1/2,y+1/2,z;-x+3/4,-y+3/4,-z+1/4;-y+3/4,x+3/4,z+1/4;-x+1/2,-y+1/2,z;y+3/4,-x+3/4,z+1/4;\
x+3/4,-z+3/4,y+1/4;x+1/2,-y+1/2,-z;x+3/4,z+3/4,-y+1/4;z+3/4,y+3/4,-x+1/4;-x+1/2,y+1/2,-z;-z+3/4,y+3/4,x+1/4;y+1/2,-x+1/2,-z;\
-y+1/2,x+1/2,-z;-x+1/2,z+1/2,-y;-x+1/2,-z+1/2,y;-z+1/2,-y+1/2,x;z+1/2,-y+1/2,-x;z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;\
z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;-z+3/4,-x+3/4,-y+1/4;-y+3/4,-z+3/4,-x+1/4;\
y+3/4,z+3/4,-x+1/4;-z+3/4,x+3/4,y+1/4;y+3/4,-z+3/4,x+1/4;z+3/4,x+3/4,-y+1/4;z+3/4,-x+3/4,y+1/4;-y+3/4,z+3/4,x+1/4;\
y+3/4,x+3/4,-z+1/4;-y+3/4,-x+3/4,-z+1/4;-x+3/4,z+3/4,y+1/4;-x+3/4,-z+3/4,-y+1/4;z+3/4,-y+3/4,x+1/4;-z+3/4,-y+3/4,-x+1/4;\
x+3/4,y+3/4,-z+1/4;-x+3/4,y+3/4,z+1/4;x+3/4,-y+3/4,z+1/4;-y+1/2,-x+1/2,z;y+1/2,x+1/2,z;x+1/2,-z+1/2,-y;x+1/2,z+1/2,y;\
-z+1/2,y+1/2,-x;z+1/2,y+1/2,x",
				"x,y,z;-x+1/4,-y+1/4,-z+3/4;-y+1/4,x+1/4,z+1/4;-x,-y,z;y+1/4,-x+1/4,z+1/4;x+1/4,-z+1/4,y+1/4;x,-y,-z;x+1/4,z+1/4,-y+1/4;\
z+1/4,y+1/4,-x+1/4;-x,y,-z;-z+1/4,y+1/4,x+1/4;y,-x,-z+1/2;-y,x,-z+1/2;-x,z,-y+1/2;-x,-z,y+1/2;-z,-y,x+1/2;z,-y,-x+1/2;z,x,y;y,z,x;\
-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;y,-z,-x;-z+1/4,-x+1/4,-y+3/4;-y+1/4,-z+1/4,-x+3/4;y+1/4,z+1/4,-x+3/4;-z+1/4,x+1/4,y+3/4;\
y+1/4,-z+1/4,x+3/4;z+1/4,x+1/4,-y+3/4;z+1/4,-x+1/4,y+3/4;-y+1/4,z+1/4,x+3/4;y+1/4,x+1/4,-z+1/4;-y+1/4,-x+1/4,-z+1/4;\
-x+1/4,z+1/4,y+1/4;-x+1/4,-z+1/4,-y+1/4;z+1/4,-y+1/4,x+1/4;-z+1/4,-y+1/4,-x+1/4;x+1/4,y+1/4,-z+3/4;-x+1/4,y+1/4,z+3/4;\
x+1/4,-y+1/4,z+3/4;-y,-x,z+1/2;y,x,z+1/2;x,-z,-y+1/2;x,z,y+1/2;-z,y,-x+1/2;z,y,x+1/2;x,y+1/2,z+1/2;-x+1/4,-y+3/4,-z+1/4;\
-y+1/4,x+3/4,z+3/4;-x,-y+1/2,z+1/2;y+1/4,-x+3/4,z+3/4;x+1/4,-z+3/4,y+3/4;x,-y+1/2,-z+1/2;x+1/4,z+3/4,-y+3/4;z+1/4,y+3/4,-x+3/4;\
-x,y+1/2,-z+1/2;-z+1/4,y+3/4,x+3/4;y,-x+1/2,-z;-y,x+1/2,-z;-x,z+1/2,-y;-x,-z+1/2,y;-z,-y+1/2,x;z,-y+1/2,-x;z,x+1/2,y+1/2;\
y,z+1/2,x+1/2;-y,-z+1/2,x+1/2;z,-x+1/2,-y+1/2;-y,z+1/2,-x+1/2;-z,-x+1/2,y+1/2;-z,x+1/2,-y+1/2;y,-z+1/2,-x+1/2;-z+1/4,-x+3/4,-y+1/4;\
-y+1/4,-z+3/4,-x+1/4;y+1/4,z+3/4,-x+1/4;-z+1/4,x+3/4,y+1/4;y+1/4,-z+3/4,x+1/4;z+1/4,x+3/4,-y+1/4;z+1/4,-x+3/4,y+1/4;\
-y+1/4,z+3/4,x+1/4;y+1/4,x+3/4,-z+3/4;-y+1/4,-x+3/4,-z+3/4;-x+1/4,z+3/4,y+3/4;-x+1/4,-z+3/4,-y+3/4;z+1/4,-y+3/4,x+3/4;\
-z+1/4,-y+3/4,-x+3/4;x+1/4,y+3/4,-z+1/4;-x+1/4,y+3/4,z+1/4;x+1/4,-y+3/4,z+1/4;-y,-x+1/2,z;y,x+1/2,z;x,-z+1/2,-y;x,z+1/2,y;\
-z,y+1/2,-x;z,y+1/2,x;x+1/2,y,z+1/2;-x+3/4,-y+1/4,-z+1/4;-y+3/4,x+1/4,z+3/4;-x+1/2,-y,z+1/2;y+3/4,-x+1/4,z+3/4;x+3/4,-z+1/4,y+3/4;\
x+1/2,-y,-z+1/2;x+3/4,z+1/4,-y+3/4;z+3/4,y+1/4,-x+3/4;-x+1/2,y,-z+1/2;-z+3/4,y+1/4,x+3/4;y+1/2,-x,-z;-y+1/2,x,-z;-x+1/2,z,-y;\
-x+1/2,-z,y;-z+1/2,-y,x;z+1/2,-y,-x;z+1/2,x,y+1/2;y+1/2,z,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x,-y+1/2;-y+1/2,z,-x+1/2;-z+1/2,-x,y+1/2;\
-z+1/2,x,-y+1/2;y+1/2,-z,-x+1/2;-z+3/4,-x+1/4,-y+1/4;-y+3/4,-z+1/4,-x+1/4;y+3/4,z+1/4,-x+1/4;-z+3/4,x+1/4,y+1/4;y+3/4,-z+1/4,x+1/4;\
z+3/4,x+1/4,-y+1/4;z+3/4,-x+1/4,y+1/4;-y+3/4,z+1/4,x+1/4;y+3/4,x+1/4,-z+3/4;-y+3/4,-x+1/4,-z+3/4;-x+3/4,z+1/4,y+3/4;\
-x+3/4,-z+1/4,-y+3/4;z+3/4,-y+1/4,x+3/4;-z+3/4,-y+1/4,-x+3/4;x+3/4,y+1/4,-z+1/4;-x+3/4,y+1/4,z+1/4;x+3/4,-y+1/4,z+1/4;-y+1/2,-x,z;\
y+1/2,x,z;x+1/2,-z,-y;x+1/2,z,y;-z+1/2,y,-x;z+1/2,y,x;x+1/2,y+1/2,z;-x+3/4,-y+3/4,-z+3/4;-y+3/4,x+3/4,z+1/4;-x+1/2,-y+1/2,z;\
y+3/4,-x+3/4,z+1/4;x+3/4,-z+3/4,y+1/4;x+1/2,-y+1/2,-z;x+3/4,z+3/4,-y+1/4;z+3/4,y+3/4,-x+1/4;-x+1/2,y+1/2,-z;-z+3/4,y+3/4,x+1/4;\
y+1/2,-x+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2;-x+1/2,z+1/2,-y+1/2;-x+1/2,-z+1/2,y+1/2;-z+1/2,-y+1/2,x+1/2;z+1/2,-y+1/2,-x+1/2;\
z+1/2,x+1/2,y;y+1/2,z+1/2,x;-y+1/2,-z+1/2,x;z+1/2,-x+1/2,-y;-y+1/2,z+1/2,-x;-z+1/2,-x+1/2,y;-z+1/2,x+1/2,-y;y+1/2,-z+1/2,-x;\
-z+3/4,-x+3/4,-y+3/4;-y+3/4,-z+3/4,-x+3/4;y+3/4,z+3/4,-x+3/4;-z+3/4,x+3/4,y+3/4;y+3/4,-z+3/4,x+3/4;z+3/4,x+3/4,-y+3/4;\
z+3/4,-x+3/4,y+3/4;-y+3/4,z+3/4,x+3/4;y+3/4,x+3/4,-z+1/4;-y+3/4,-x+3/4,-z+1/4;-x+3/4,z+3/4,y+1/4;-x+3/4,-z+3/4,-y+1/4;\
z+3/4,-y+3/4,x+1/4;-z+3/4,-y+3/4,-x+1/4;x+3/4,y+3/4,-z+3/4;-x+3/4,y+3/4,z+3/4;x+3/4,-y+3/4,z+3/4;-y+1/2,-x+1/2,z+1/2;\
y+1/2,x+1/2,z+1/2;x+1/2,-z+1/2,-y+1/2;x+1/2,z+1/2,y+1/2;-z+1/2,y+1/2,-x+1/2;z+1/2,y+1/2,x+1/2",
				"x,y,z;-y,x,z;-x,-y,z;y,-x,z;x,-z,y;x,-y,-z;x,z,-y;z,y,-x;-x,y,-z;-z,y,x;z,x,y;y,z,x;-y,-z,x;z,-x,-y;-y,z,-x;-z,-x,y;-z,x,-y;\
y,-z,-x;y,x,-z;-y,-x,-z;-x,z,y;-x,-z,-y;z,-y,x;-z,-y,-x;-x,-y,-z;y,-x,-z;x,y,-z;-y,x,-z;-x,z,-y;-x,y,z;-x,-z,y;-z,-y,x;x,-y,z;\
z,-y,-x;-z,-x,-y;-y,-z,-x;y,z,-x;-z,x,y;y,-z,x;z,x,-y;z,-x,y;-y,z,x;-y,-x,z;y,x,z;x,-z,-y;x,z,y;-z,y,-x;z,y,x;x+1/2,y+1/2,z+1/2;\
-y+1/2,x+1/2,z+1/2;-x+1/2,-y+1/2,z+1/2;y+1/2,-x+1/2,z+1/2;x+1/2,-z+1/2,y+1/2;x+1/2,-y+1/2,-z+1/2;x+1/2,z+1/2,-y+1/2;\
z+1/2,y+1/2,-x+1/2;-x+1/2,y+1/2,-z+1/2;-z+1/2,y+1/2,x+1/2;z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z+1/2,x+1/2;\
z+1/2,-x+1/2,-y+1/2;-y+1/2,z+1/2,-x+1/2;-z+1/2,-x+1/2,y+1/2;-z+1/2,x+1/2,-y+1/2;y+1/2,-z+1/2,-x+1/2;y+1/2,x+1/2,-z+1/2;\
-y+1/2,-x+1/2,-z+1/2;-x+1/2,z+1/2,y+1/2;-x+1/2,-z+1/2,-y+1/2;z+1/2,-y+1/2,x+1/2;-z+1/2,-y+1/2,-x+1/2;-x+1/2,-y+1/2,-z+1/2;\
y+1/2,-x+1/2,-z+1/2;x+1/2,y+1/2,-z+1/2;-y+1/2,x+1/2,-z+1/2;-x+1/2,z+1/2,-y+1/2;-x+1/2,y+1/2,z+1/2;-x+1/2,-z+1/2,y+1/2;\
-z+1/2,-y+1/2,x+1/2;x+1/2,-y+1/2,z+1/2;z+1/2,-y+1/2,-x+1/2;-z+1/2,-x+1/2,-y+1/2;-y+1/2,-z+1/2,-x+1/2;y+1/2,z+1/2,-x+1/2;\
-z+1/2,x+1/2,y+1/2;y+1/2,-z+1/2,x+1/2;z+1/2,x+1/2,-y+1/2;z+1/2,-x+1/2,y+1/2;-y+1/2,z+1/2,x+1/2;-y+1/2,-x+1/2,z+1/2;\
y+1/2,x+1/2,z+1/2;x+1/2,-z+1/2,-y+1/2;x+1/2,z+1/2,y+1/2;-z+1/2,y+1/2,-x+1/2;z+1/2,y+1/2,x+1/2",
				"x,y,z;-y+1/4,x+3/4,z+1/4;-x,-y+1/2,z;y+1/4,-x+1/4,z+3/4;x+1/4,-z+1/4,y+3/4;x,-y,-z+1/2;x+3/4,z+1/4,-y+1/4;z+3/4,y+1/4,-x+1/4;\
-x+1/2,y,-z;-z+1/4,y+3/4,x+1/4;z,x,y;y,z,x;-y,-z+1/2,x;z,-x,-y+1/2;-y+1/2,z,-x;-z,-x+1/2,y;-z+1/2,x,-y;y,-z,-x+1/2;\
y+3/4,x+1/4,-z+1/4;-y+1/4,-x+1/4,-z+1/4;-x+1/4,z+3/4,y+1/4;-x+1/4,-z+1/4,-y+1/4;z+1/4,-y+1/4,x+3/4;-z+1/4,-y+1/4,-x+1/4;-x,-y,-z;\
y+3/4,-x+1/4,-z+3/4;x,y+1/2,-z;-y+3/4,x+3/4,-z+1/4;-x+3/4,z+3/4,-y+1/4;-x,y,z+1/2;-x+1/4,-z+3/4,y+3/4;-z+1/4,-y+3/4,x+3/4;\
x+1/2,-y,z;z+3/4,-y+1/4,-x+3/4;-z,-x,-y;-y,-z,-x;y,z+1/2,-x;-z,x,y+1/2;y+1/2,-z,x;z,x+1/2,-y;z+1/2,-x,y;-y,z,x+1/2;\
-y+1/4,-x+3/4,z+3/4;y+3/4,x+3/4,z+3/4;x+3/4,-z+1/4,-y+3/4;x+3/4,z+3/4,y+3/4;-z+3/4,y+3/4,-x+1/4;z+3/4,y+3/4,x+3/4;\
x+1/2,y+1/2,z+1/2;-y+3/4,x+1/4,z+3/4;-x+1/2,-y,z+1/2;y+3/4,-x+3/4,z+1/4;x+3/4,-z+3/4,y+1/4;x+1/2,-y+1/2,-z;x+1/4,z+3/4,-y+3/4;\
z+1/4,y+3/4,-x+3/4;-x,y+1/2,-z+1/2;-z+3/4,y+1/4,x+3/4;z+1/2,x+1/2,y+1/2;y+1/2,z+1/2,x+1/2;-y+1/2,-z,x+1/2;z+1/2,-x+1/2,-y;\
-y,z+1/2,-x+1/2;-z+1/2,-x,y+1/2;-z,x+1/2,-y+1/2;y+1/2,-z+1/2,-x;y+1/4,x+3/4,-z+3/4;-y+3/4,-x+3/4,-z+3/4;-x+3/4,z+1/4,y+3/4;\
-x+3/4,-z+3/4,-y+3/4;z+3/4,-y+3/4,x+1/4;-z+3/4,-y+3/4,-x+3/4;-x+1/2,-y+1/2,-z+1/2;y+1/4,-x+3/4,-z+1/4;x+1/2,y,-z+1/2;\
-y+1/4,x+1/4,-z+3/4;-x+1/4,z+1/4,-y+3/4;-x+1/2,y+1/2,z;-x+3/4,-z+1/4,y+1/4;-z+3/4,-y+1/4,x+1/4;x,-y+1/2,z+1/2;z+1/4,-y+3/4,-x+1/4;\
-z+1/2,-x+1/2,-y+1/2;-y+1/2,-z+1/2,-x+1/2;y+1/2,z,-x+1/2;-z+1/2,x+1/2,y;y,-z+1/2,x+1/2;z+1/2,x,-y+1/2;z,-x+1/2,y+1/2;\
-y+1/2,z+1/2,x;-y+3/4,-x+1/4,z+1/4;y+1/4,x+1/4,z+1/4;x+1/4,-z+3/4,-y+1/4;x+1/4,z+1/4,y+1/4;-z+1/4,y+1/4,-x+3/4;z+1/4,y+1/4,x+1/4"
			};

	if (SpaceGroup<1 || SpaceGroup>230) {*str=NULL; return 0;}		/* invalid SpaceGroup, not in [1,230] */

	len = strlen(symLines[SpaceGroup-1]);
	*str = (char *)calloc(len+1,sizeof(char));
	if (!(*str)) { fprintf(stderr,"unable to allocate space for symLines[%d] in setSymLine()\n",SpaceGroup); exit(1); }
	strncpy(*str,symLines[SpaceGroup-1],len);
	Nops = 0;
	p = *str;
	while(p) {
		p = strchr(++p,';'); 
		Nops++;
	}
	return Nops;
}
