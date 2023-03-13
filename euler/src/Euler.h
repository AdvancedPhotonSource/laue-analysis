#ifndef _EULER_H_
#define	_EULER_H_


#include "mathUtil.h"
#include "lattice.h"

/* #define	KEV_MAX 6.62, KEV_MIN 4			// maximum energy (kev),   was 20  (11.58 for hklmax=7) */
/* #define	KEV_MAX 7.5, KEV_MIN 4			// maximum energy (kev),   was 20  (11.58 for hklmax=7) */
/* #define	KEV_MAX 8.12, KEV_MIN 4			// maximum energy (kev),   was 20  (11.58 for hklmax=7) */
/* #define	KEV_MAX 11.5, KEV_MIN 4			// maximum energy (kev),   was 20  (11.58 for hklmax=7) */
/* #define	KEV_MAX 13.2					// maximum energy (kev),   was 20  (11.58 for hklmax=7) */
/* #define	KEV_MAX 17.0					// maximum energy (kev),   was 20  (11.58 for hklmax=7) */
#define	KEV_MAX 16.0						/* maximum energy (kev) */
#define	KEV_MIN 4							/* minimum energy (kev),   was 6, this is not yet used */


#define dANGLE_PIXEL (0.025/100./10.)		/* angle between pixels, this is the ultimate limit on angular precision (radian) */


#define THETA_MAX ((135/2)*M_PI/180)		/* maximum theta for the detector (radian).  (max 2-theta = 90+45deg.) */
#define THETA_MIN ((45/2)*M_PI/180)			/* minimum theta for the detector (radian).  (min 2-theta = 90-45deg.) */
#define	DETECTOR_HEIGHT 100.				/* detector height, y coordinate */
/* #define	DETECTOR_HW 50.					// width of detector is ±hw in both x and z */
#define	DETECTOR_HW 102.					/* width of detector is ±hw in both x and z */
/* #define	PI M_PI							// 3.141592653589793238462643 */
#ifndef hc
#define	hc 12.3984187						/* ( h*c (keV-Å), from 2006 CODATA,  http://physics.nist.gov/cuu/Constants/index.html */
#endif
#define	ao 4.05								/* lattice constant of aluminum (Å) */

#define MIN_SPOTS_FOR_1_PATTERN  4			/* minmimum number of spots it takes to identify one Laue pattern */
#define MAX_GRAINS_PER_PATTERN  20			/* maximum number of grains that can be fit in one Laue pattern */
/*#define NUM_SPOTS_FOR_MATCH 4				// number of spots that must match to say we matched a grain */
#define NUM_SPOTS_FOR_MATCH 3				/* number of spots that must match to say we matched a grain */
											/* number of spots indexed <NUM_SPOTS_FOR_MATCH, then stop even trying */

#ifdef __POWERPC__							/* this is special for XCODE because it builds in a lower 'build' directory */
/* #define DEFAULT_FOLDER "../../" */
#define DEFAULT_FOLDER ""
#else
#define DEFAULT_FOLDER ""
#endif

#define ALPHAMIN (-M_PI)					/* range of alpha [-180°,180°) */
#define ALPHAMAX M_PI
#define BETAMIN 0							/* range of beta [0,180°) */
#define BETAMAX M_PI
#define GAMMAMIN 0							/* range of [0,360°) */
#define GAMMAMAX (M_PI+M_PI)
/*	alphaMin = -PI/4  ;  alphaMax = PI/4;   // range of alpha	[-45°,45°) */
/*	betaMin = 0  ;  betaMax = 1;			// range of beta	[0°,57°)   , 57.3°  ~  (001)^(111)	this is for Cubic xtals */
/*	gammaMin = -PI/2  ;  gammaMax = PI/2;	// range of gamma	[-90°,90°) */

extern FILE	*fout;							/* file descriptor, either /dev/null, or stdout */


struct	EulerAngle_pair {					/* now store the values in AllEulerAngles[] */
	long	m0;								/* index to hkl of axis reflection */
	long	m;								/* index to hkl of rotated reflection */
	long	i0;								/* index to G of axis reflection (from detector spot) */
	long	i;								/* index to G of rotated reflection (from detector spot) */
	double  alpha;							/* alpha for Euler Angles */
	double  beta;							/* beta for Euler Angles */
	double  gamma;							/* gamma for Euler Angles */
	};

struct	WaveSpace_struct {		/* struct describing an EulerSpace, 3d with scaling*/
	long	Na;					/* number of points along the alpha dimension */
	long	Nb;					/* number of points along the beta dimension */
	long	Ng;					/* number of points along the gamma dimension */
	double	aoff;				/* alphaOffset */
	double	da;					/* step size along the alpha direction */
	double	boff;				/* betaOffset */
	double	db;					/* step size along the beta direction */
	double	goff;				/* gammaOffset */
	double	dg;					/* step size along the gamma direction */
	long	***space;			/* pointer to a 3-d space */
								/* xxx.space[alpha][beta][gamma] */
	};

struct	box_struct {			/* struct describing a box in 3d */
	double  xlo;				/* minmimum value of G^ in the x direction */
	double  xhi;				/* maximum value of G^ in the x direction */
	double  ylo;				/*   note that min<-1 and max>1 allows all */
	double  yhi;
	double  zlo;
	double  zhi;
	};


struct	dotPlusIndicies {					/* dot product and indicies of source vectors */
	double  dot;							/* Ghat[i1] .dot. Ghat[i2] */
	long	i1;								/* index i1 into Ghat[i1] of one detector spot */
	long	i2;								/* index i2 into Ghat[i2] of other detector spot */
};


struct	patternOfOneGrain {					/* describes the Laue pattern from one grain */
	double  goodness;						/* goodness of pattern, (sum of intensities)*(number of spots) */
	double  alpha;							/* Euler angles for this grain (radian) */
	double  beta;
	double  gamma;
	long	Ni;								/* number of spots in pattern from this gain */
	int		(*hkls)[3];						/* pointer to list of hkl for each of the spots */
	double  (*Ghat)[3];						/* pointer to unit vectors in direction of each hkl after Euler rotation */
	double  *intens;						/* intensity of each spot */
	double  *err;							/* error in radians (measured- predicted) */
	struct  crystalStructure xtal;
	int		*pkIndex;						/* array with index, for found pattern this is spot number of data peak */
};



void MakeAllPossibleGvectors(long *Nm, double (**Gmhat)[3], double **GmLen, int (**hklm)[3], int hkl0[3], \
	double cone, double keVmax,struct crystalStructure *xtal);

/* void MakeListOfAllEulerAngles(struct box_struct abgRange, double angleTolerance, long Ni, double GhatSpots[][3], \
 *	double farDot, long Ndots, struct dotPlusIndicies *dotList, long Nm, double Gmhat[][3], double *GmLen, \
 *	struct EulerAngle_pair **AllEulerAngles, size_t *NAllEulerAngles);
 */
void MakeListOfAllEulerAngles(struct box_struct abgRange, double angleTolerance, double GhatSpots[][3], \
	double farDot, long Ndots, struct dotPlusIndicies *dotList, long Nm, double Gmhat[][3], \
	struct EulerAngle_pair **AllEulerAngles, size_t *NAllEulerAngles);

long RemoveDuplicateTriplets(long N, double (**list)[3], double **associate);
long RemoveDuplicateTripletsPlus(long N, double (*vec)[3], ...);

int compareLongReverse(const void *a, const void *b);
int compareDouble(const void *a, const void *b);
int compare_double_reverse(const void *a, const void *b);

long MakeDotList(long Ni, double Ghat[][3], struct dotPlusIndicies **dotList);
/* long MakePretendDataUnits(long Npatterns, double keVmax, double sigmaAngle, struct patternOfOneGrain **patterns); */

int testOrientFast(char *fname, char *outfile, double keVmaxCalc, \
	double keVmaxTest, double angleTolerance,int hkl0[3], double cone, long maxData);
/*int testOrientFast(char *fname, long Npatterns,double sigmaAngle, double keVmaxPretendData, double keVmaxCalc, \
 *	double keVmaxTest, double angleTolerance,int hkl0[3], double cone, long maxData);
 */

/* int OrientFast(long size, double keVmaxCalc, double keVmaxTest, double angleTolerance, int hkl0[3], double cone, \
 *	struct crystalStructure *xtal, long Ni, double (*GhatSpots)[3], double *intensSpots, long *Nfound, \
 *	struct patternOfOneGrain foundPattern[MAX_GRAINS_PER_PATTERN]);
 */
int OrientFast(long size, double keVmaxCalc, double keVmaxTest, double angleTolerance, int hkl0[3], double cone, \
	struct crystalStructure *xtal, long Ni, double (*GhatSpots)[3], int *indexSpots, long *Nfound, \
	struct patternOfOneGrain foundPattern[MAX_GRAINS_PER_PATTERN]);

int optimizeEulerAngles(double startStep, double epsAbs, long maxIter, struct patternOfOneGrain *pattern);

#endif
