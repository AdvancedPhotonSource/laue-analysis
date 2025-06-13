#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include "Euler.h"


#ifdef DEBUG_ON
#define DEBUG 2
#endif

#define MIN(A,B)	( (A)<(B)?(A):(B) )
#define MAX(A,B)	( (A)>(B)?(A):(B) )
#define MAG3(A,B,C) sqrt((A)*(A)+(B)*(B)+(C)*(C))
#define DOT3(A,B)	(A[0]*B[0] + A[1]*B[1] + A[2]*B[2])
#define MATRIXVECTOR31(M,V,Z)  {   Z[0]=M[0][0]*V[0]+M[0][1]*V[1]+M[0][2]*V[2]; \
									Z[1]=M[1][0]*V[0]+M[1][1]*V[1]+M[1][2]*V[2]; \
									Z[2]=M[2][0]*V[0]+M[2][1]*V[1]+M[2][2]*V[2]; }		/* Z = M*V */
#define MATRIXCOPY33(D,S)  {	D[0][0]=S[0][0]; D[0][1]=S[0][1]; D[0][2]=S[0][2]; \
								D[1][0]=S[1][0]; D[1][1]=S[1][1]; D[1][2]=S[1][2]; \
								D[2][0]=S[2][0]; D[2][1]=S[2][1]; D[2][2]=S[2][2]; }	/* copy, D[][] = S[][] */

#define MATRIXTRANSPOSE33(A)	{ double SWAP;  SWAP=A[0][1]; A[0][1]=A[1][0]; A[1][0]=SWAP; \
												SWAP=A[0][2]; A[0][2]=A[2][0]; A[2][0]=SWAP; \
												SWAP=A[1][2]; A[1][2]=A[2][1]; A[2][1]=SWAP; }  /* transpose the 3x3 matrix */


long foundDotInList(double dot, long Ndots, struct dotPlusIndicies *dotList, long start, double  threshold);
void vecPairs2rotation(double Ghati0[3], double Ghati[3], double Ghatm0[3], double Ghatm[3], double Am0[3][3],double Amat[3][3]);



/* finds the rotation matrix that takes one pair of vectors to another pair
 * this routine assumes that dot(Ghati0,Ghati) equals dot(Ghatm0,Ghatm0), since
 * if the angle between each pair differs, we can never do this
 */
void vecPairs2rotation(
double  Ghati0[3],			/* direction of the i0th measured spot */
double  Ghati[3],			/* direction of the ith measured spot */
double  Ghatm0[3],			/* direction of the m0th G(hkl) */
double  Ghatm[3],			/* direction of the mth G(hkl) */
double  Am0[3][3],			/* matrix that does Am0*Ghatm0 = (001) */
double  Amat[3][3])			/* desired rotation matrix, the result */
{
	double	alphai0;						/* alpha of G^i0 */
	double	betai0;							/* beta to rotated G^i0 */
	double	gammam;							/* gamma to rotate G^m'' into G^i'' */
	double	Ai0[3][3];						/* Ai0 * Ghati0 = (001) */
	double	Ai0inv[3][3];					/* Inverse(Ai0) */
	double  Rz[3][3];						/* rotation matrix about z-axis */
	double  mat[3][3];

	double  Gi[3],Gi0[3];					/* local versions of Ghati[] and Ghati0[] */
	double  Gipp[3];
	double  Ghatmpp[3];
	double  dotm;							/* dot(Ghatm,Ghatm0) */
	double  doti;							/* dot(Ghati,Ghati0) */
	double  perp[3];
/*	double  dth;							// angle to rotate Gi and Gi0 */
	double  cosDelta;						/* acos(doti) - acos(dotm)*/
	double  sine,cosine;					/* sin(dth) and cos(dth) */

	{
		/* Gi[] and Gi0[] are local version to get modified so that the angle between them exactly matches angle between Gmhat & Gmhat0 */
		/* this is done to improve the precision of the results, it should not make a difference in whether it works */
		VECTOR_COPY3(Gi,Ghati);
		VECTOR_COPY3(Gi0,Ghati0);
		dotm = DOT3(Ghatm0,Ghatm);					/* dotm = Ghatm0[0]*Ghatm[0]+Ghatm0[1]*Ghatm[1]+Ghatm0[2]*Ghatm[2]; */
		doti = DOT3(Gi0,Gi);						/* doti = Gi0[0]*Gi[0]+Gi0[1]*Gi[1]+Gi0[2]*Gi[2]; */
		cosDelta = doti*dotm + sqrt((1-doti*doti)*(1-dotm*dotm));   /* = acos(doti) - acos(dotm) */
		sine = sqrt((1-cosDelta)/2);
		cosine = sqrt((1+cosDelta)/2);
/* are the 3 lines above faster than the 3 lines that follow ?
 *		dth = ( acos(doti) - acos(dotm) )/2.;
 *		sine = sin(dth);
 *		cosine = cos(dth);
 */
		perp[0] = Ghati[0] - doti*Ghati0[0];		/* b - (a.b)*a,   perp shoud be perp to a (a is Gi0) */
		perp[1] = Ghati[1] - doti*Ghati0[1];
		perp[2] = Ghati[2] - doti*Ghati0[2];
		normalize3(perp);
		Gi0[0] = cosine*Ghati0[0] + sine*perp[0];
		Gi0[1] = cosine*Ghati0[1] + sine*perp[1];
		Gi0[2] = cosine*Ghati0[2] + sine*perp[2];

		perp[0] = Ghati0[0] - doti*Ghati[0];		/* a - (a.b)*b,   perp shoud be perp to b (b is Gi) */
		perp[1] = Ghati0[1] - doti*Ghati[1];
		perp[2] = Ghati0[2] - doti*Ghati[2];
		normalize3(perp);
		Gi[0] = cosine*Ghati[0] + sine*perp[0];
		Gi[1] = cosine*Ghati[1] + sine*perp[1];
		Gi[2] = cosine*Ghati[2] + sine*perp[2];
		/* now Gi and Gi0 are still close to Ghati and Ghati0, but have the same angle between them as Ghatm and Ghatm0 */
	}

	/* find rotation matrix, Am0*Ghatm0 = (001) */
	/* and  rotation matrix, Ai0*Gi0 = (001) */

	alphai0 = atan2(Gi0[1],Gi0[0]);					/* azimuthal angle of G^i0 */
	betai0 = acos(Gi0[2]);							/* angle from z^ to G^i0 */
	/* EulerMatrix(alphai0,betai0,0,) * Gi0 = (0,0,1)
	 *
	 * [ EulerMatrix(alpham0,betam0,0,) * Ghatm0 ] = [ EulerMatrix(alphai0,betai0,0,)*Gi0 ] = {0,0,1}
	 *  since Rz(any_gamma)*{0,0,1} = {0,0,1}
	 * Rz(gammam)*[ EulerMatrix(alpham0,betam0,0,) * Ghatm0 ] = [ EulerMatrix(alphai0,betai0,0,)*Gi0 ]
	 * for any gammam
	 *
	 * for a vector m and i that should line up, the only allowed change is to rotate with the gammam,  so:   
	 * Rz(gammam)*[ EulerMatrix(alpham0,betam0,0,) * Ghatm ] = [ EulerMatrix(alphai0,betai0,0,)*Gi ]
	 *  and for convienence, Am0 = EulerMatrix(alpham0,betam0,0,),   and Ai0 = EulerMatrix(alphai0,betai0,0,)
	 * Note that I use gammam not gamma (gamma is the final value), since they are different.
	 */
	EulerMatrix(alphai0,betai0,0.,Ai0);
	MATRIXCOPY33(Ai0inv,Ai0)						/* MatrixCopy33(Ai0inv,Ai0);	// Ai0inv[][] = Ai0[][] */
	MATRIXTRANSPOSE33(Ai0inv)						/* MatrixTranspose33(Ai0inv);   // inverse of Ai0 */
	/* find gammam, defined by:
	 * Rz(gammam)*[ EulerMatrix(alpham0,betam0,0,) * Ghatm ] = [ EulerMatrix(alphai0,betai0,0,)*Gi ]
	 * Rz(gammam)*[ Am0 * Ghatm ] = [ Ai0*Gi ], note that the equality will in general not be exact */
	MATRIXVECTOR31(Ai0,Gi,Gipp);					/* MatrixMultiply31(Ai0,Gi,Gipp); */
	/* Rz(gammam)*Ghatmpp = Gipp */
	MATRIXVECTOR31(Am0,Ghatm,Ghatmpp)				/* MatrixMultiply31(Am0,Ghatm,Ghatmpp); */
	gammam = atan2(Ghatmpp[1],Ghatmpp[0]) - atan2(Gipp[1],Gipp[0]);
	MatrixRz(Rz,gammam);
	MatrixMultiply33(Ai0inv,Rz,mat);				/* Then the Amat = Ai0inverse * Rz(gammam) * Am0 */
	MatrixMultiply33(mat,Am0,Amat);
}




long foundDotInList(						/* returns index into dotList of match, <0 means out or range */
double  dot,								/* dot product to search for */
long	Ndots,								/* number of entries in dotList[] */
struct dotPlusIndicies *dotList,			/* pointer to an array of dotPlusIndicies structures */
long	start,								/* start searching with index 'start' */
double  threshold)							/* max difference between dot products to be counted as found*/
{
	long	i;
	double  dotth;								/* dot-threshold */
	start = MAX(0,start);
	if (start>=Ndots) return -4;				/* start must lie in range [0, Ndots-1] */
	if (dot+threshold<dotList[start].dot) return -2;/* dot is too small to even be in list */
	dotth = dot-threshold;
	if (dotth>dotList[Ndots-1].dot) return -3;/* dot is too big to ever be in list */
	for (i=start;i<Ndots && dotth>dotList[i].dot;i++);
	if (fabs(dot-(dotList[i].dot))<=threshold) return i;  /* dot matches dotList[i] */
	return -1;									/* just not close enough */
}




long MakeDotList(					/* returns number of entries in dotList, on error returns negative number */
long	Ni,							/* number of 3-vectors in Ghat */
double	Ghat[][3],					/* Ghat for each of the spots, take all pairs fof Ghat[][3] */
struct dotPlusIndicies **dotList)	/* pointer to an array of dotPlusIndicies structures to create */
{
	long	N;						/* number of entries in dotList */
	long	i,j;
	long	n;						/* current index into dotList */

	if (*dotList) { fprintf(stderr,"dotList is not NULL on entry to MakeDotList()"); return -1; }
	N = Ni*(Ni-1);					/* init to maximum number */
	*dotList = calloc((size_t)N,sizeof(**dotList));
	if (!(*dotList)) { fprintf(stderr,"could not allocate space for dotList in MakeDotList()\n"); return -1; }

	n = 0;
	for (j=0;j<Ni;j++) {
		for (i=0;i<Ni;i++) {
			if (i==j) continue;
			(*dotList)[n].dot = dot3(Ghat[i],Ghat[j]);
			(*dotList)[n].i1 = j;
			(*dotList)[n].i2 = i;
			n++;
		}
	}
	if (n!=N) { fprintf(stderr,"n=%ld  and  N=%ld in MakeDotList()\n",n,N); exit(1); }
	qsort(*dotList,(size_t)N,sizeof(**dotList),compareDouble);  /* next sort by increasing dot product */
#if (DEBUG>2)
fprintf(fout,"in MakeDotList(), made %ld entries\n",N);
#endif
	return N;
}



/* This routine makes a list of all hkl ( and the G vectors) that can possibly match up with one 
 * of the measured spots.  The results are returned in the arrays Gmhat[][3], hklm[][3], and GmLen[].
 * The criteria are that |G| must lie in range given by max and min 2theta and range of wavelengths
 * also, the angle between hkl0 and each hklm must be < cone.
 * Space for Gmhat, hklm, GmLen is allocated here, it must be freed up later by some other routine.
 */
void MakeAllPossibleGvectors(
long	*Nm,					/* number of possible hkl being considered (length of GmLen, Gmhat[Nm][3], & hklm[Nm][3]) */
double  (**Gmhat)[3],			/* all possible Ghats to check, one for each possible hkl (stored in hklm) */
double	**GmLen,				/* corresponding lengths of Gmhat */
int		(**hklm)[3],			/* list of hkl[][3] corresponding to each of the Gmhat[][3] */
int		hkl0[3],				/* preferred hkl at center of pattern */
double  cone,					/* acceptable cone angle from preferred hkl0 */
double  keVmax,					/* maximum energy (keV) to go out to */
struct crystalStructure *xtal)	/* lattice parameters */
{
	/*	Pre-Generate a list of allowable hkl
	 * dmin is given by the maximum 2-theta and the maximum energy using Bragg's Law, lambda = 2d sin(theta)
	 * dmin = lambdaMin/2/sin(thetaMax),  max 2theta is 90+45=135 for our detector
	 * hklmax = ao / dmin, actually do this independently for each h,k,l since lattice may not be cubic
	 */

	double  dmin, dmax;									/* minimum (and max) d-spacing that we can reach */
	double  gmax, gmin;									/* maximum (and min) possible length of a g-vector */
	long	hmax,kmax,lmax;								/* maximum possible values of h, k, and l */
	double  parallel;									/* cos(cone), used to test dot products */
	long	p;
	long	N;
	double  hkl[3];
	double  vec[3];
	double  glen;
	double  ghat0[3];									/* direction of hkl0[3], normalized */
	long	i,h,k,l,m;
	double	*sortDouble7;								/* array of (GmLen,Gmhat[0...3],hklm[0..3])[i] for sorting */
#if (DEBUG)
	clock_t time0;
	time0=clock();
#endif

	if (*Gmhat || *GmLen || *hklm) return;				/* since this will be allocated, it needs to be NULL on entry */
	parallel = cos(cone);								/* dot(hkl,hkl0) must be greater or equal to parallel */
	hkl[0]=hkl0[0];  hkl[1]=hkl0[1];  hkl[2]=hkl0[2];   /* make ghat0[3], direction vector given by hkl0[3] */
	MatrixMultiply31(xtal->recip,hkl,ghat0);
	normalize3(ghat0);
	dmin = (hc/keVmax)/2/sin(THETA_MAX);				/* maximum 2-theta is 135¡ for our detector */
	gmax = 2*M_PI/dmin;
	hmax = ceil(gmax/MAG3(xtal->recip[0][0],xtal->recip[1][0],xtal->recip[2][0])); /* max range to check */
	kmax = ceil(gmax/MAG3(xtal->recip[0][1],xtal->recip[1][1],xtal->recip[2][1]));
	lmax = ceil(gmax/MAG3(xtal->recip[0][2],xtal->recip[1][2],xtal->recip[2][2]));
	dmax = (hc/KEV_MIN)/2/sin(THETA_MIN);				/* minimum 2-theta is 45¡ for our detector */
	gmin = 2*M_PI/dmax;
	N = (2*hmax+1)*(2*kmax+1)*(2*lmax+1);

	*Gmhat = calloc((size_t)N,3*sizeof(double));
	if (!(*Gmhat)) { fprintf(stderr,"could not allocate space (%ld bytes) for Gmhat in MakeAllPossibleGvectors()\n",N*3*sizeof(double)); *Nm=0; return;}
	*GmLen = calloc((size_t)N,sizeof(double));
	if (!(*GmLen)) { fprintf(stderr,"could not allocate space for GmLen in MakeAllPossibleGvectors()\n"); *Nm=0; return;}
	*hklm = calloc((size_t)N,3*sizeof(int));
	if (!(*hklm)) { fprintf(stderr,"could not allocate space for hklm in MakeAllPossibleGvectors()\n"); *Nm=0; return;}
	for (p=0;p<N;p++) (*GmLen)[p] = -1;
	m = 0;
	for (l=-lmax;l<=lmax;l++) {
		hkl[2]=l;
		for (k=-kmax;k<=kmax;k++) {
			hkl[1]=k;
			for (h=-hmax;h<=hmax;h++) {
				if (!allowedHKL(xtal,h,k,l)) continue;	/* check if (hkl) is allowed */
				hkl[0]=h;
				MatrixMultiply31(xtal->recip,hkl,vec);  /* vec is vector pointing in hkl direction */
				glen = normalize3(vec);
				if (glen<gmin || glen>gmax) continue;	/* only consider hkl within distance given by keVmax */
				if (dot3(ghat0,vec)<parallel) continue;	/* angle between hkl and hkl0 is greater than 'cone' */
				VECTOR_COPY3((*Gmhat)[m],vec);			/* save the possible Ghat[m] */
				(*hklm)[m][0]=h; (*hklm)[m][1]=k; (*hklm)[m][2]=l;	/* save the ocrresponding hkl[m]*/
				(*GmLen)[m] = glen;						/* and save the length */
				m++;
			}
		}
	}
	*Nm = m;
	*Gmhat = realloc(*Gmhat,3*(*Nm)*sizeof(double));	/* resize arrays to Nm, what was actually found */
	*GmLen = realloc(*GmLen,(*Nm)*sizeof(double));
	*hklm = realloc(*hklm,3*(*Nm)*sizeof(int));

	sortDouble7 = calloc((size_t)*Nm,7*sizeof(double));			/* sort the Gmhat, GmLen, & hklm by increasing GmLen */
	if (!sortDouble7) {
		CHECK_FREE(*Gmhat);
		CHECK_FREE(*GmLen);
		CHECK_FREE(*hklm);
		return;
	}
	for(i=0;i<(*Nm);i++) {
		sortDouble7[7*i+0] = (*GmLen)[i];				/* save GmLen (what I will be sorting on */
		sortDouble7[7*i+1] = (*Gmhat)[i][0];			/* the 3-vector that I really want to sort */
		sortDouble7[7*i+2] = (*Gmhat)[i][1];
		sortDouble7[7*i+3] = (*Gmhat)[i][2];
		sortDouble7[7*i+4] = (*hklm)[i][0];
		sortDouble7[7*i+5] = (*hklm)[i][1];
		sortDouble7[7*i+6] = (*hklm)[i][2];
	}
#if (DEBUG>1)
{
	double gminx=gmax, gmmaaxx=0;
	for(i=0;i<(*Nm);i++) {
		glen = ((*GmLen)[i]);
		gminx = glen<gminx ? glen : gminx;
		gmmaaxx = glen>gmmaaxx ? glen : gmmaaxx;
	}
	fprintf(fout,"\tjust before first call to qsort(), range of |G| is [%g, %g],   gmin=%g  gmax=%g\n",gminx,gmmaaxx,gmin,gmax);
}
#endif
	qsort(sortDouble7,(size_t)*Nm,7*sizeof(double),compareDouble);
	/* void qsort(
	* void *base,				base void * A pointer to the array to be sorted
	* size_t nmemb,				nmemb size_t The number of elements to sort
	* size_t size,				size size_t The size of an array element
	* int (*compare) (const void *, const void *)) compare void * A pointer to a comparison function
	*/
	for(i=0;i<(*Nm);i++) {
		(*GmLen)[i] = sortDouble7[7*i+0];				/* reassign the sorted values */
		(*Gmhat)[i][0] = sortDouble7[7*i+1];
		(*Gmhat)[i][1] = sortDouble7[7*i+2];
		(*Gmhat)[i][2] = sortDouble7[7*i+3];
		(*hklm)[i][0] = sortDouble7[7*i+4];
		(*hklm)[i][1] = sortDouble7[7*i+5];
		(*hklm)[i][2] = sortDouble7[7*i+6];
	}
	free(sortDouble7);

	/* remove duplicate directions determined by comparing each Gmhat[][3] to each other Gmhat[][3] */
	*Nm = RemoveDuplicateTripletsPlus(*Nm,*Gmhat,*GmLen,sizeof((*GmLen)[0]),*hklm,sizeof((*hklm)[0]),NULL);
	*Gmhat = realloc(*Gmhat,3*(*Nm)*sizeof(double));
	*GmLen = realloc(*GmLen,(*Nm)*sizeof(double));
	*hklm = realloc(*hklm,3*(*Nm)*sizeof(int));

#if (DEBUG)
	fprintf(fout,"MakeAllPossibleGvectors() checking out to (hkl)=(%ld,%ld,%ld), found (Nm=)%ld possible hkl's in %.2f sec\n", \
			hmax,kmax,lmax,*Nm,((double)(clock()-time0))/CLOCKS_PER_SEC);
#endif
}




/* makes the list of all Euler angles for each pair (m0,m) and (i0,i), this is likely to be a very long list
 * It allocates space for Gmhat & GmLen which are freed here, AllEulerAngles must be freed up later
 */
void MakeListOfAllEulerAngles(
struct box_struct abgRange,		/* structure giving the allowed ranges of alpha, beta, and gamma */
double  angleTolerance,			/* difference in angles between G-vector pairs to be considered coincident (radians) */
//long	Ni,						/* number of vectors in GhatSpots */
double	GhatSpots[][3],			/* Ghat for each of the spots on the detector */
double  farDot,					/* dot product between the two data spots that are farthest apart from each other */
long	Ndots,					/* number of elemnts in dotList[] */
struct dotPlusIndicies *dotList,/* list of all dot products from the data spots */
long	Nm,						/* length of Gmhat[] and GmLen[] */
double	Gmhat[][3],				/* list of all possible Gvectors, one for each possible hkl */
//double	*GmLen,					/* length of G vector for each hkl */
struct EulerAngle_pair **AllEulerAngles, /* pointer to an array of EulerAngle_pair structures */
size_t	*NAllEulerAngles)		/* number of slots allocated in AllEulerAngles */
{
	/*	Pre-Generate a list of allowable hkl
	 * dmin is given by the maximum 2-theta and the maximum energy using Bragg's Law, lambda = 2d sin(theta)
	 * dmin = lambdaMin/2/sin(thetaMax),  max 2theta is 90+45=135 for our detector
	 * hklmax = ao / dmin, actually do this independently for each h,k,l since the lattice may not be cubic
	 */

	double  cos_closePair;								/* spots are too close together to rotate about one of them */
	double	alpha, beta, gamma;							/* three Euler angles, loop over alpha, beta, solve for gamma */
	double	doti, dotm;									/* G^m ¥ G^m0, and G^i ¥ G^i0 */

	double	Ghatm0[3];									/* un-rotated direction of axis hkl, (hkl[m0]) */
	double	Ghatm[3];									/* G^ for a particular hkl to match */
	double  Ghatmpp[3];									/* rotated versions of Ghatm */
	double	Am0[3][3];
	double	Amat[3][3];

	long	Neuler;										/* number of entries in AllEulerAngles */
	long	m0,i0;
	long	m, i;
	long	idot;										/* index into dotList */
	double	alpham0;									/* alpha of G^m0 */
	double	betam0;										/* beta to rotated G^m0 */
#if (DEBUG)
	double  seconds;									/* used to print execution time */
	clock_t time0;
	char	str[81];									/* string used to show time */
	time0=clock();
#endif

	/*	precompute the list of (m0,m,i0,i,alpha,beta,gamma) */
	cos_closePair = cos(0.1*M_PI/180);							/* to decide if spots are too close together to rotate about one of them */

	*NAllEulerAngles = 10000;
	*AllEulerAngles = calloc(*NAllEulerAngles,sizeof(**AllEulerAngles));
	Neuler=0;
	for (m0=0;m0<Nm;m0++) {									/* loop over the possibl hkl (hkl[m0] defines the axis) */
#if (DEBUG)
		if (m0==10) {
			time_t  tod;
			char	dateStr[255];
			int		iii;
			seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;
			seconds *= Nm*(Nm-1)/2 / (10/2*(2*Nm-10-1));		/* Nm*(Nm-1)/2 total number, 5*(2Nm-11) is number done so far */
			tod = time(NULL)+seconds;
			strcpy(dateStr,ctime(&tod));
			iii = strlen(dateStr)-1;						/* index to the last character */
			if (dateStr[iii]<32) dateStr[iii]='\0';			/* take care of last char being <LF> */
			if (seconds>10.) fprintf(fout,"this should end at %s   (after %.0f more secconds),     %ld (of %ld)\n",dateStr,seconds,m0,Nm-1);
		}
#endif
		VECTOR_COPY3(Ghatm0,Gmhat[m0]);						/* G^ of hkl[m0] */
		alpham0 = atan2(Ghatm0[1],Ghatm0[0]);				/* azimuthal angle of G^m0 */
		betam0 = acos(Ghatm0[2]);							/* angle from z^ to G^m0 */
		EulerMatrix(alpham0,betam0,0.,Am0);
		/* EulerMatrix(alpham0,0,0,) * Ghatm0 = (1,0,1) */
		/* EulerMatrix(alpham0,betam0,0,) * Ghatm0 = (0,0,1) */
		/*							  Am0 * Ghatm0 = (0,0,1) */

#ifdef __GNUC__
#warning "If I can easily get Euler angles to swap (i0,m0; i,m) to (i0,m; i,m0), then dotList does not have to include (i0,i) and (i,i0)"
#endif
		for (m=m0+1;m<Nm;m++) {								/* avoids (m0,m) and (m,m0) since dotList already has (i0,i) and (i,i0) */
			Ghatm[0]=Gmhat[m][0]; Ghatm[1]=Gmhat[m][1]; Ghatm[2]=Gmhat[m][2];	/* un-rotated direction of hkl[m] */
			dotm = dot3(Ghatm,Ghatm0);						/* between hkl[m] and  hkl[m0] */
			if (dotm < farDot) continue;					/* there are no data points this far apart */
			if (fabs(dotm)>cos_closePair) continue;			/* the spots are too close together to rotate about one of them */
			MatrixMultiply31(Am0,Ghatm,Ghatmpp);

			idot = -1;
			while ((idot=foundDotInList(dotm,Ndots,dotList,idot+1,angleTolerance))>=0) {
				i0 = dotList[idot].i1;
				i = dotList[idot].i2;
				doti = dotList[idot].dot;					/* between ith spot and ioth spot */
				if(fabs(doti-dotm)>angleTolerance) continue;/* no gamma can do this, this lie is not really correct, but is fast, see next line */
				if(fabs(acos(doti)-acos(dotm))>angleTolerance) continue;/* correct, but slower test than in previous line */
#ifdef __GNUC__
#warning "most of the time in this routine comes from the vecPairs2rotation()"
#endif
				vecPairs2rotation(GhatSpots[i0],GhatSpots[i],Ghatm0,Ghatm,Am0,Amat);
				rot2EulerAngles(Amat,&alpha,&beta,&gamma);   /* this computes three Euler angles from A */

				if (alpha<abgRange.xlo || alpha>=abgRange.xhi || beta<abgRange.ylo || beta>=abgRange.yhi || gamma<abgRange.zlo || gamma>=abgRange.zhi) continue;
				/* now that we have the EulerAngles increment correct point in EulerSpace
				 * store the m,m0,i,i0,alpha,beta,gamma in the list */
				if (Neuler >= (long)(*NAllEulerAngles)) {	/* need to extend AllEulerAngles */
					*NAllEulerAngles += 10000;
					*AllEulerAngles = realloc(*AllEulerAngles,sizeof(**AllEulerAngles)*(*NAllEulerAngles));
				}
				/* now store the values in *AllEulerAngles[][] */
				(*AllEulerAngles)[Neuler].m0 = m0;
				(*AllEulerAngles)[Neuler].m = m;
				(*AllEulerAngles)[Neuler].i0 = i0;
				(*AllEulerAngles)[Neuler].i = i;
				(*AllEulerAngles)[Neuler].alpha = alpha;
				(*AllEulerAngles)[Neuler].beta = beta;
				(*AllEulerAngles)[Neuler].gamma = gamma;
				Neuler += 1;
			}			/* end while */
		}				/* end loop over m */
	}					/* end loop over m0 */
	*NAllEulerAngles = Neuler;
	*AllEulerAngles = realloc(*AllEulerAngles,sizeof(**AllEulerAngles)*(*NAllEulerAngles));
#if (DEBUG)
	seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;
	fprintf(fout,"making the list of all Euler angles 'AllEulerAngles[%ld][7]' takes %s  =  (%.2f sec)\n",Neuler,num2sexigesmal(str,seconds,0),seconds);
#endif
	/* don't foget to free up GmLen and Gmhat later */
}




/* for vec[N][3] which is a list of N 3-vectors, remove all duplicate vectors, and corresponding vectors
	 if they exist.  This returns the the number of vectors left in vec[][3] */
long RemoveDuplicateTripletsPlus(
long	N,									/* number of triplets in vec */
double	(*vec)[3],
...)
{
	va_list ap; 
	void	*pntr;							/* pointer to associated array */
	size_t  psize;							/* size of point in associated array */
	double	tolerance;
	double	test;
	long	i,m;
	long	j;
	long	*dups;							/* a list of index to higher orders to remove */
	long	Ndups;
	long	id;
	void	*pntrs[50];
	size_t  sizes[50];

	if (!vec) { fprintf(stderr," the input 'vec[][3]' must exsit in RemoveDuplicateTripletsPlus()\n"); return 0; }
	tolerance = 1e-5;
	dups = calloc((size_t)N,sizeof(long));
	if (!dups) { fprintf(stderr," cannot allocate space for 'dups' in RemoveDuplicateTriplets()\n"); return 0; }
	for (i=0;i<N;i++) dups[i] = -1;					/* init to unused */
	Ndups = 0;
	va_start(ap, vec);
	for (i=0;i<50;i++) {							/* store arguments in local arrays */
		sizes[i] = 0;								/* ensure terminating zeros */
		pntrs[i] = NULL;
		pntr = va_arg(ap,void*);
		if (pntr==NULL) break;
		psize = va_arg(ap,size_t); 
		if (psize <= 0) break;
		sizes[i] = psize;
		pntrs[i] = pntr;
	}
	va_end(ap);

/*		fprintf(fout,"\n");
 *		for (j=0;(sizes[j])>0;j++) fprintf(fout,"pntrs[%ld] = %p,    size = %ld\n",j,pntrs[j],sizes[j]);
 */
	for (m=0;m<N;m+=1) {
		for (i=m+1;i<N;i++) {
			if (i==m) continue;						/* don't check itself */
			test = fabs(vec[m][0]-vec[i][0]) + fabs(vec[m][1]-vec[i][1]) + fabs(vec[m][2]-vec[i][2]);
			if (test<tolerance) {					/* check for duplicates */
				dups[Ndups] = i;
				Ndups++;
				break;
			}
		}
	}
	if (Ndups<1) { free(dups); return N; }			/* no duplicates found */
	/* for (i=0;i<Ndups;i++) fprintf(fout,"dups[%2ld] = %3ld\n",i,dups[i]); */

	qsort(dups,(size_t)Ndups,sizeof(long),compareLongReverse);  /* sort dups[], so largest indicies come first */

	for (i=0;i<Ndups;i++) {							/* delete the duplicate vectors */
		id = dups[i];
		DeletePoints((size_t)(N-id),&(vec[id][0]),3*sizeof(double),1);
		for (j=0;(psize=sizes[j])>0;j++) {			/* args exist, delete these too */
/*			DeletePoints((size_t)(N-id),pntrs[j]+psize*id,psize,1); */
			DeletePoints((size_t)(N-id),((char *)pntrs[j])+psize*id,psize,1);
		}
		N--;
	}
	free(dups);										/* clean things up before leaving */ 
	return N;
}



/* for a 2-d wave (N,3) which is a list of N 3-vectors, remove all duplicate vectors
   returns the changed wave 'list', and the number of vectors in list */
long RemoveDuplicateTriplets(
long	N,									/* number of triplets in list */
double	(**list)[3],
double	**associate)
{
	double	tolerance;
	double	test;
	long	i,m;
	double	mainV[3];
	long	*dups;							/* a list of index to higher orders to remove */
	long	Ndups;
	long	id;

	if (!(*list)) {
		fprintf(stderr," the input list must be (N,3) in RemoveParalleVectors()]\n");
		return 0;
	}

	tolerance = 1e-5;
	dups = calloc((size_t)N,sizeof(long));
	if (!dups) {
		fprintf(stderr," cannot allocate space for 'dups' in RemoveDuplicateTriplets()\n");
		return 0;
	}
	for (i=0;i<N;i++) dups[i] = -1;
	Ndups = 0;

	for (m=0;m<N;m+=1) {
		mainV[0]=(*list)[m][0]; mainV[1]=(*list)[m][1]; mainV[2]=(*list)[m][2];
		for (i=m+1;i<N;i++) {
			if (i==m) continue;						/* don't check itself */
			test = fabs(mainV[0]-(*list)[i][0]) + fabs(mainV[1]-(*list)[i][1]) + fabs(mainV[2]-(*list)[i][2]);
			if (test<tolerance) {					/* check for duplicates */
				dups[Ndups] = i;
				Ndups++;
				break;
			}
		}
	}
	if (Ndups<1) { free(dups); return N; }			/* no duplicates found */
	qsort(dups,(size_t)Ndups,sizeof(long),compareLongReverse);
	for (i=0;i<Ndups;i+=1) {						/* delete the duplicate vectors */
		id = dups[i];
/*		DeletePoints(3*(N-id),&((*list)[id][0]),3*sizeof(double),1); , the extra 3 is wrong? */
		DeletePoints((size_t)(N-id),&((*list)[id][0]),3*sizeof(double),1);
		if (associate) {
			if(*associate) DeletePoints((size_t)(N-id),&((*associate)[id]),sizeof(double),1);
		}
		N--;
	}
	free(dups);
	return N;
}




int compareLongReverse(		/* for reverse sorting a simple array of longs */
const void *a,
const void *b)
{
	long aa,bb;
	aa = *(long *)a;
	bb = *(long *)b;
	if (aa>bb) return -1;
	else if (aa<bb) return 1;
	return 0;				/* for aa==bb */
} 

int compare_double_reverse(		/* for REVERSE sorting a simple array double [n], sorting on the first element of the multiplet */
const void	*a,
const void	*b)
{
	double aa, bb;
	aa = *(double *)a;
	bb = *(double *)b;
	if (aa<bb) return 1;
	else if (aa>bb) return -1;
	return 0;				/* for aa==bb */
} 
/*
 *     User compareDouble for this
 *	int compareGmLenGmhat(		// for sorting a simple array double [n], sorting on the first element of the multiple
 *  const void	*a,
 *  const void	*b)
 *  {
 *  	double aa, bb;
 *  	aa = *(double *)a;
 *  	bb = *(double *)b;
 *  	if (aa<bb) return -1;
 *  	else if (aa>bb) return 1;
 *  	return 0;				// for aa==bb
 *  } 
 */
int compareDouble(		/* for sorting a simple array double [n], sorting on the first element of the multiple */
const void	*a,
const void	*b)
{
	double aa, bb;
	aa = *(double *)a;
	bb = *(double *)b;
	if (aa<bb) return -1;
	else if (aa>bb) return 1;
	return 0;				/* for aa==bb */
} 

/*
 * void qsort(
 * void *base,					base void * A pointer to the array to be sorted
 * size_t nmemb,				nmemb size_t The number of elements to sort
 * size_t size,					size size_t The size of an array element
 * int (*compare) (const void *, const void *))		compare void * A pointer to a comparison function
 *
 * The compare argument is a pointer to a programmer-supplied compare function. 
 * The function takes two pointers to different array elements and compares them 
 * based on the key. If the two elements are equal, compare must return a zero. 
 * The compare function must return a negative number if the first element is
 * less than the second. Likewise, the function must return a positive number if 
 * the first argument is greater than the second. 
 *
 *
 * int comp(const DIRENTRY *rec1, const DIRENTRY *rec2) 
 * {
 * 		return (strcmp((char *)rec1->lname, (char *)rec2->lname));
 * } 
 */


/*
long DimSize(int EulerSpace, int dim);
double DimDelta(int EulerSpace, int dim);
long DimSize(
int EulerSpace,
int dim)
{
	if (dim==0) return 41;
	if (dim==1) return 41;
	if (dim==2) return 164;
	return 0;
}
double DimDelta(
int EulerSpace,
int dim)
{
	if (dim==0) return 9.67381e-05;
	if (dim==1) return 6.21951e-05;
	if (dim==2) return 4.8369e-05;
	return 0;
}
*/
/*
 *  Wave: EulerSpace
 *  Type: FP32   Size: 1103056 bytes
 *  Rows: 41 Units: alpha Start: -0.00295051 Delta: 9.67381e-05
 *  Columns: 41 Units: beta Start: 0.326445 Delta: 6.21951e-05
 *  Layers: 164 Units: gamma Start: -0.661785 Delta: 4.8369e-05
 */
