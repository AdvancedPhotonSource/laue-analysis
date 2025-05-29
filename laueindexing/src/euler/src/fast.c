#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _MSC_VER				/* identifies this as a Microsoft compiler */
#define _USE_MATH_DEFINES	/* added RX2011 */
#endif
#include <math.h>
#include <time.h>
#include <limits.h>
#include "Euler.h"
#include "checkFileType.h"

#ifdef DEBUG_ON
#define DEBUG 5				/* a value of 5 gets everything */
#endif
#define MIN_PEAK_IN_HOUGH 2 /* if peak in Hough is less than this, stop zooming */


/* these two values used in the optimization routine */
#define EPS_ABS 1e-6					/* tolerance on the answer (1e-6) */
#define MAX_ITER 100					/* maximum number of iterations, stop after this many regardless, perhaps 100 */


#define MIN(A,B)	( (A)<(B)?(A):(B) )
#define MAX(A,B)	( (A)>(B)?(A):(B) )
/* #define CHECK_FREE(A)   { if(A) free(A); (A)=NULL;}	// moved to mathUtil.h */
#define MAG3(A,B,C) sqrt((A)*(A)+(B)*(B)+(C)*(C))
#define NORM3(A)	sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2])
#define DOT3(A,B)	(A[0]*B[0] + A[1]*B[1] + A[2]*B[2])


long readDataFromJZT(char *fname, struct patternOfOneGrain *dataPattern, long maxData, char **headerBuf);
int strFromTagFile(FILE *f,char *tagIn, char *value, long maxLen);
int strFromTagBuf(char *buffer, char *tagIn, char *value, long maxLen);
long readDataFromWenge(char *fname, struct patternOfOneGrain *dataPattern, long maxData);
long FindClosestMeasuredG(double recip[3][3], int hkls[3], size_t Nmeas, double (*Gmeas)[3]);

int magnifyEulerSpace(struct WaveSpace_struct *EulerSpace, double mag, double center[3]);
/* int EulerSpaceFromSpotsFast(struct WaveSpace_struct *EulerSpace, struct EulerAngle_pair *AllEulerAngles,size_t Neuler, double center[3]); */
int EulerSpaceFromSpotsFast(struct WaveSpace_struct *EulerSpace, struct EulerAngle_pair *AllEulerAngles,size_t Neuler);
long MakeLauePatternUnitVectors(double a, double b, double g, struct box_struct range, struct crystalStructure *xtal, \
	double (**GhatSpots)[3], int (**hklSpots)[3], int hkl0[3], double cone, double keVmax);

void SetGhatRangeFromDataSpots(long Ni, double  (*GhatSpots)[3], struct box_struct *range);
void reflectionProperties(double Ghat[3], int hkl[3], struct crystalStructure *xtal, double *keV, double *intens);

long PeakInHough(struct WaveSpace_struct *HoughSpace, double angles[3], int type);
void vectorCOM(struct WaveSpace_struct *wav3d, double com[3], double hw);
long maxOnCubeFace(long Na, long Nb, long Ng, long ***hough, long i, long j, long k, long hw);
long FindMaxIn3d(long n0, long n1, long n2, long ***a, long *imax, long *jmax, long *kmax);

double dotOfBiggestAngle(long Ni, double hat[][3]);
char *HighestReachableHKL(double keV, double theta, struct crystalStructure *xtal, char str[256]);
void lowestOrderHKLint(int hkl[3]);

#define FREE_PATTERN_OF_GRAIN(A) if ((A).Ni>0) { CHECK_FREE((A).hkls); CHECK_FREE((A).Ghat); CHECK_FREE((A).intens); CHECK_FREE((A).pkIndex); CHECK_FREE((A).err); freeCrystalStructure(&((A).xtal)); (A).Ni=0; }
#define INIT_CLEAN_PATTERN_OF_GRAIN(A) { (A).hkls=NULL; (A).Ghat=NULL; (A).intens=NULL; (A).pkIndex=NULL; (A).err=NULL; InitCleanCrystalStructure(&((A).xtal)); (A).Ni=0; }



/* a good start is testOrientFast(5,12,20) */
/* This routine takes a list of data unit vectors, and produces a list of EulerAngles for the different grains that fit */
int testOrientFast(
char	*fname,				/* file name if using a file for the peaks */
char	*outfile,			/* name of output file */
double  keVmaxCalc,			/* maximum energy (keV) when calculating fit */
double  keVmaxTest,			/* limit in energy for finding test points (Gtest), probably [2^(1/3)]*keVmaxCalc */
double  angleTolerance,		/* difference in angles between G-vector pairs to be considered coincident (radians) */
int		hkl0[3],			/* preferred hkl at center of pattern */
double  cone,				/* acceptable cone angle from preferred hkl0 (radian) */
long	maxData)			/* max number of spots to use, ie use the first maxData spots */
{
	int		err=0;							/* error flag returned, 0=OK */
	long	size=41;						/* size of the Euler Space  (2*size,size,2*size) */
	long	Ni;								/* total number of measured spots */
	struct patternOfOneGrain dataPattern;	/* a patternOfOneGrain structure with the data Laue patterns */
	double  (*GhatSpots)[3]=NULL;			/* direction of each spot on the detector */
	double  *intensSpots=NULL;				/* intensity of each spot on the detector */
	int		*pkIndexSpots=NULL;				/* index of each spot on the detector */
	long	Nfound=0;						/* number of patterns found, and stored in foundPattern[] */
	struct	patternOfOneGrain foundPattern[MAX_GRAINS_PER_PATTERN]; /* holds the fitted patterns */
	long	i,j;
	long	Nindexed;						/* number of spots indexed */
	double  mat[3][3];
	clock_t time0,time1;					/* time for getting execution time */
	double  seconds;						/* used to print execution time */
	char	str[256];						/* string used to show time */
	double  kihat[3]={0,0,1};				/* incident beam direction, used to print keV */
	double	sinTheta;						/* sin(Bragg angle) of reflection, used to print keV */
	double  keV;							/* energy of reflection */
	double  stdDevErr[MAX_GRAINS_PER_PATTERN]; /* std deviation (rad) of foundPattern[j].err[i] */
	long	hkl[3];							/* used to print lowest order allowed reflection */
	FILE	*f;								/* file descriptor */
	char	*headerBuf=NULL;				/* string with tag values to add to output file */
	double  startStep;						/* starting step size (radians), make this a bit larger than the error */
	struct	patternOfOneGrain fitPattern;   /* holds the pattern for fitting */
	double  M_Euler[3][3];					/* an Euler matrix */
	double  recip[3][3];					/* reciprocal space */
	long	ispot;
	double  vec[3];							/* temp vector */

	if (!fname && strlen(fname)<1) { fprintf(stderr,"no input file name given\n"); return 1; }
	if (!outfile || strlen(outfile)<1)  { fprintf(stderr,"no output file name given\n"); return 1; }
#if (DEBUG)
	fprintf(fout,"called:  testOrientFast(fname='%s',out='%s',keVmaxCalc=%g,keVmaxTest=%g,angleTolerance=%gdeg, hkl0=(%d %d %d), cone=%gdeg, maxData=%ld);\n\n", \
		fname,outfile,keVmaxCalc,keVmaxTest,angleTolerance*180/M_PI,hkl0[0],hkl0[1],hkl0[2],cone*180/M_PI,maxData);
	fprintf(fout,"how should 'angleTolerance' be calculated?  Something to think about.\n");
#endif
/*  angleTolerance determines how accuratly the angle between two data spots must match the angle betweeen two calculated hkl */
/*	angleTolerance = 0.025/(Detector height);				// this corresponds to a 25µm pixel separation at 100mm away */
/*	if (strstr(fname,"xyTOkq_Si")) angleTolerance *= 3.;	// 3 pixels wide, for good stuff like Si */
/*	else if (strstr(fname,"xyTOkq_Al")) angleTolerance *= 40.;// 20 pixels wide, for good stuff like Aluminum */


	INIT_CLEAN_PATTERN_OF_GRAIN(dataPattern);
	INIT_CLEAN_PATTERN_OF_GRAIN(fitPattern);
	for (i=0;i<MAX_GRAINS_PER_PATTERN;i++) {				/* initialize all of the structures */
		INIT_CLEAN_PATTERN_OF_GRAIN(foundPattern[i]);
	}
//	InitCleanCrystalStructure(&(dataPattern.xtal));			/* initialize the crystalStructure structs */
//	InitCleanCrystalStructure(&(fitPattern.xtal));
//	for (i=0;i<MAX_GRAINS_PER_PATTERN;i++) {				/* initialize all of the structures */
//		InitCleanCrystalStructure(&(foundPattern[i].xtal));
//	}

	Ni = readDataFromJZT(fname,&dataPattern,maxData,&headerBuf);	/* assume that there is only 1 pattern */
	if (Ni<1) Ni = readDataFromWenge(fname,&dataPattern,maxData);	/* assume that there is only 1 pattern */
	if (Ni<2) {fprintf(stderr,"Unable to read in data\n"); err=1; goto return_path; }
	GhatSpots = calloc((size_t)Ni,3*sizeof(double));
	intensSpots = calloc((size_t)Ni,sizeof(double));
	pkIndexSpots = calloc((size_t)Ni,sizeof(int));
	for (i=0;i<Ni;i++) {							/* transfer all the spots to GhatSpots[][3] */
		VECTOR_COPY3(GhatSpots[i],dataPattern.Ghat[i]);
		intensSpots[i]  = dataPattern.intens[i];
		pkIndexSpots[i]  = dataPattern.pkIndex[i];
	}

#ifdef DEBUG
	fprintf(fout,"read in %ld spots from '%s'\n",Ni,fname);
#if (DEBUG>2)
	fprintf(fout,"  index                 G^		    Intensity\n");
	for (i=0;i<Ni;i++) fprintf(fout,"  [%3ld]   (%8.5f, %8.5f, %8.5f)    %g %d\n",i,GhatSpots[i][0],GhatSpots[i][1],GhatSpots[i][2],intensSpots[i],pkIndexSpots[i]);
#endif
#endif
/*	for (i=0;i<MAX_GRAINS_PER_PATTERN;i++) {
 *		copyCrystalStructure(&(foundPattern[i].xtal),&(dataPattern.xtal));
 *	}
 */
#ifdef DEBUG
	fprintf(fout,"\nabout to call OrientFast(size=%ld, keVmax=%g, keVmaxTest=%g, angleTol=%.5f, hkl0=(%d%d%d), cone=%g, lattice, Ni=%ld, GhatSpots, intensSpots, &Nfound, foundPattern);\n\n", \
		size,keVmaxCalc,keVmaxTest,angleTolerance,hkl0[0],hkl0[1],hkl0[2],cone,Ni);
#endif
	time0 = clock();

	err = OrientFast(size,keVmaxCalc,keVmaxTest,angleTolerance,hkl0,cone,&(dataPattern.xtal),Ni,GhatSpots,pkIndexSpots,&Nfound,foundPattern);
	if (err) goto return_path;
	seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;

	fprintf(fout,"\n");
	for (i=Nindexed=0;i<Nfound;i++) {
		fprintf(fout,"found orientation %2ld  at (%13.8f, %13.8f, %13.8f)    %3ld spots\n",i,foundPattern[i].alpha*180/M_PI,foundPattern[i].beta*180/M_PI,foundPattern[i].gamma*180/M_PI,foundPattern[i].Ni);
		Nindexed += foundPattern[i].Ni;
	}
	fprintf(fout,"%ld spots left (out of %ld),  entire OrientFast() takes %s  =  (%.2f sec)\n",Ni-Nindexed,Ni,num2sexigesmal(str,round(seconds),0),seconds);

	for (j=0;j<Nfound;j++) {
		for (i=0,stdDevErr[j]=0.;i<(foundPattern[j].Ni);i++) 
			stdDevErr[j] += (foundPattern[j].err[i])*(foundPattern[j].err[i]);
		stdDevErr[j] = sqrt(stdDevErr[j]/foundPattern[j].Ni);			/* rms of errors (rad) */
	}

	/*************************************************************************/
	/* optimiize Euler angles for all of the found patterns */
	fprintf(fout,"\n");
	time1 = clock();
	for (j=0;j<Nfound;j++) {							/* loop over the found patterns */
		if (foundPattern[j].Ni<=0) continue;
		if (Nfound>1) fprintf(fout,"\n");
#ifdef DEBUG
/*		fprintf(fout,"before optimizing, stdDevErr[%ld] = %.4g (deg.)\n",j,stdDevErr[j]*180/M_PI); */
		fprintf(fout,"before optimize, stdDevErr[%ld] = %.4g (deg.),  Euler = {%.2f, %.2f, %.2f}\n",j,stdDevErr[j]*180/M_PI, \
			foundPattern[j].alpha*180/M_PI,foundPattern[j].beta*180/M_PI,foundPattern[j].gamma*180/M_PI);
#endif

		/* set fitPattern which gets passed to the optimization routine optimizeEulerAngles */
		copyCrystalStructure(&(fitPattern.xtal),&(foundPattern[j].xtal));
		fitPattern.goodness = foundPattern[j].goodness; /* fill fitPattern from foundPattern[j] */
		fitPattern.alpha	= foundPattern[j].alpha;
		fitPattern.beta		= foundPattern[j].beta;
		fitPattern.gamma	= foundPattern[j].gamma;
		fitPattern.Ni		= foundPattern[j].Ni;
		fitPattern.hkls		= foundPattern[j].hkls;
		fitPattern.intens   = foundPattern[j].intens;
		fitPattern.pkIndex  = foundPattern[j].pkIndex;
		fitPattern.err		= foundPattern[j].err;		/* this gets reset using optimized values in optimizeEulerAngles() */

		/* need to set fitPattern.Ghat to the measured (not calculated) G^ */
		MatrixCopy33(recip,fitPattern.xtal.recip);		/* copy unrotated reciprocal lattice into recip */
		EulerMatrix(fitPattern.alpha,fitPattern.beta,fitPattern.gamma,M_Euler);/* rotation matrix from Euler angles */
		MatrixMultiply33(M_Euler,recip,recip);			/* rotate recip by Euler angles */
		fitPattern.Ghat = calloc((size_t)Ni,3*sizeof(double));
		if (!(fitPattern.Ghat)) { fprintf(stderr,"unable to allocate space for 'fitPattern[%ld].Ghat'\n",j); err=1; goto return_path; }
		for (i=0;i<(fitPattern.Ni);i++) {
			ispot = FindClosestMeasuredG(recip,fitPattern.hkls[i],(size_t)Ni,GhatSpots);
			if (ispot<0) { fprintf(stderr,"Cannot find nearest spot in GhatSpots\n"); err=1; goto return_path; }
			VECTOR_COPY3(fitPattern.Ghat[i],GhatSpots[ispot]);
		}

		startStep = 10*stdDevErr[j];					/* starting step size (radians), make this a bit larger than the error */
		err = optimizeEulerAngles(startStep, EPS_ABS, MAX_ITER, &fitPattern);
		/* err = optimizeEulerAngles(10*stdDevErr[j], 1e-6, 100, &fitPattern);
			int optimizeEulerAngles(
			double  startStep,					// starting step size (radians), make this a bit larger than the error
			struct	patternOfOneGrain *pattern) // provides G^'s and hkl's for one fitted pattern, and the lattice parameters */
		err = (err==-2) ? 0 : err;				/* do not pass an error if maxiterations was reached */
		if (err) fprintf(fout,"optimizeEulerAngles() returned with %d\n",err);

		CHECK_FREE(fitPattern.Ghat);					/* free and re-allocate for each pattern */
		foundPattern[j].alpha = fitPattern.alpha;		/* save optimized Euler angles */
		foundPattern[j].beta  = fitPattern.beta;
		foundPattern[j].gamma = fitPattern.gamma;
		/* note, .err[] was reset in the optimization routine */

		stdDevErr[j] = 0.;								/* recalculate stdDevErr[j] using optimized values */
		for (i=0;i<(foundPattern[j].Ni);i++) stdDevErr[j] += (foundPattern[j].err[i])*(foundPattern[j].err[i]);
		stdDevErr[j] = sqrt(stdDevErr[j]/foundPattern[j].Ni);			/* rms of errors (rad) */
#ifdef DEBUG
/*		fprintf(fout,"after optimize, stdDevErr[%ld] = %.4g (deg.)\n",j,stdDevErr[j]*180/M_PI); */
		fprintf(fout,"after optimize,  stdDevErr[%ld] = %.4g (deg.),  Euler = {%.2f, %.2f, %.2f}\n",j,stdDevErr[j]*180/M_PI, \
			foundPattern[j].alpha*180/M_PI,foundPattern[j].beta*180/M_PI,foundPattern[j].gamma*180/M_PI);
#endif
	}
	seconds = ((double)(clock()-time1))/CLOCKS_PER_SEC;
#ifdef DEBUG
	if (Nfound>1) fprintf(fout,"optimizing takes %s  =  (%.2f sec)\n",num2sexigesmal(str,round(seconds),0),seconds);
	else fprintf(fout,"no patterns found, skipping optimization\n");
#endif

	/* done optimizing */
	/*************************************************************************/

	/* output of the result */
	seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;					/* seconds is now the total execution time */
	fprintf(fout,"\nTotal execution time took %s = (%.2f sec)\n",num2sexigesmal(str,round(seconds),0),seconds);

	if (Nfound<1) { fprintf(fout,"\n\n No matching patterns found\n"); }
	else if (Nfound==1) fprintf(fout,"\n\nFound 1 pattern, it is:\n");
	else fprintf(fout,"\n\nFound %ld patterns, they are:\n",Nfound);

	for (j=0;j<Nfound;j++) {
#ifdef DEBUG
		fprintf(fout,"\n");
		print_crystalStructure(fout,&(foundPattern[j].xtal));
#endif
		fprintf(fout,"\n pattern [%ld] with Euler angles (%13.8f,%13.8f,%13.8f) (deg)\n",j,foundPattern[j].alpha*180/M_PI,foundPattern[j].beta*180/M_PI,foundPattern[j].gamma*180/M_PI);
		EulerMatrix(foundPattern[j].alpha,foundPattern[j].beta,foundPattern[j].gamma,mat);	/* make the rotation matrix */
		fprintf(fout,"   rotation matrix     %8.5f     %8.5f     %8.5f\n",mat[0][0],mat[0][1],mat[0][2]);
		fprintf(fout,"   column vectors      %8.5f     %8.5f     %8.5f\n",mat[1][0],mat[1][1],mat[1][2]);
		fprintf(fout,"                       %8.5f     %8.5f     %8.5f\n",mat[2][0],mat[2][1],mat[2][2]);
		fprintf(fout,"       #                  G^                      (hkl)       intens      E(keV)       err(deg)  PeakIndex\n");
		for (i=0;i<(foundPattern[j].Ni);i++) {
			VECTOR_COPY3(hkl,foundPattern[j].hkls[i]);
			lowestAllowedHKL(hkl,&(foundPattern[j].xtal));
			sinTheta = -dot3(kihat,foundPattern[j].Ghat[i]);			/* sin(Bragg angle) */
			VECTOR_COPY3(vec,hkl);
			MatrixMultiply31(foundPattern[j].xtal.recip,vec,vec);
			keV = hc*NORM3(vec) / (4*M_PI*sinTheta);					/* G = 4*PI*sin(theta)/lambda, and G = 2*PI/d */
			fprintf(fout,"    [%3ld]   (% 8.5f % 8.5f % 8.5f)     (%3ld %3ld %3ld)    %6.4f,    %7.4f,    %9.5f    %d\n",i, \
				foundPattern[j].Ghat[i][0],foundPattern[j].Ghat[i][1],foundPattern[j].Ghat[i][2], \
				hkl[0],hkl[1],hkl[2],foundPattern[j].intens[i],keV,foundPattern[j].err[i]*180./M_PI,foundPattern[j].pkIndex[i]);
		}
		fprintf(fout,"                                             goodness of pattern = %g,   rms err=%.3g (deg)\n",foundPattern[j].goodness,stdDevErr[j]*180/M_PI);
	}

	if ( (f=fopen(outfile,"w")) == NULL) { fprintf(stderr,"Can't open file '%s'\n",outfile); exit(1); }
	fprintf(stdout,"writing output to file '%s'\n",outfile);	/* ALWAYS write this line */
	fprintf(f,"$filetype	IndexFile\n");
	fprintf(f,"// Found %ld patterns, indexed %ld out of %ld spots in %s = (%.2f sec)\n",Nfound,Nindexed,Ni,num2sexigesmal(str,round(seconds),0),seconds);
	fprintf(f,"// ------------------------------------------------------------\n");
	if (fname) fprintf(f,"$peakFile		'%s'		// input data file\n",fname);
	fprintf(f,"$keVmaxCalc		%g 			// max energy (keV) for calculated hkl\n",keVmaxCalc);
	fprintf(f,"$angleTolerance		%g 	// how close to vectors have to be considered to have the correct angle (deg)\n",angleTolerance*180/M_PI);
	fprintf(f,"$keVmaxTest		%.2f		// max energy (keV) matching a spot (for calculating Gtest[][3])\n",keVmaxTest);
	fprintf(f,"$hklPrefer		'{%d,%d,%d}'	// preferred hkl, this should be hkl near center of pattern\n",hkl0[0],hkl0[1],hkl0[2]);
	fprintf(f,"$cone		%g 				// angle from the preferred hkl to look for acceptable hkl when calculating (deg)\n",cone*180/M_PI);
	fprintf(f,"$NpatternsFound		%ld		// number of patterns found\n",Nfound);
	fprintf(f,"$Nindexed		%ld 			// number of spots indexed\n",Nindexed);
	fprintf(f,"$NiData		%ld 				// total number of data spots\n",Ni);
	fprintf(f,"$executionTime		%.2f	// execution time (sec)\n",seconds);
	fprintf(f,"// ------------------------------------------------------------\n");
	fprintf(f,"// these are parameters from header of $peakFile\n");
	fprintf(f,"%s",headerBuf);
	fprintf(f,"// ------------------------------------------------------------\n");

	for (j=0;j<Nfound;j++) {
		fprintf(f,"\n$pattern%ld \n",j);
		fprintf(f,"$EulerAngles%ld {%13.8f,%13.8f,%13.8f}	// Euler angles for this pattern (deg)\n",j,foundPattern[j].alpha*180/M_PI,foundPattern[j].beta*180/M_PI,foundPattern[j].gamma*180/M_PI);
		fprintf(f,"$goodness%ld		%g						// goodness of the this pattern\n",j,foundPattern[j].goodness);
		fprintf(f,"$rms_error%ld		%.3g					// rms error of (measured-predicted) (deg)\n",j,stdDevErr[j]*180/M_PI);
		EulerMatrix(foundPattern[j].alpha,foundPattern[j].beta,foundPattern[j].gamma,mat);	/* make rotation matrix */
		fprintf(f,"$rotation_matrix%ld		{{%.7f,%.7f,%.7f}{%.7f,%.7f,%.7f}{%.7f,%.7f,%.7f}}\n", \
			j, mat[0][0],mat[1][0],mat[2][0],mat[0][1],mat[1][1],mat[2][1],mat[0][2],mat[1][2],mat[2][2]);
		fprintf(f,"//   rotation matrix     %8.5f     %8.5f     %8.5f\n",mat[0][0],mat[0][1],mat[0][2]);
		fprintf(f,"//   column vectors      %8.5f     %8.5f     %8.5f\n",mat[1][0],mat[1][1],mat[1][2]);
		fprintf(f,"//                       %8.5f     %8.5f     %8.5f\n",mat[2][0],mat[2][1],mat[2][2]);
		MatrixMultiply33(mat,foundPattern[j].xtal.recip,mat);			/* rotate recip to pattern[j] orientation */
		mat[0][0] *= 10 ; mat[1][0] *= 10 ; mat[2][0] *= 10;			/* convert from (1/Angstrom) --> (1/nm) */
		mat[0][1] *= 10 ; mat[1][1] *= 10 ; mat[2][1] *= 10;
		mat[0][2] *= 10 ; mat[1][2] *= 10 ; mat[2][2] *= 10;
		fprintf(f,"$recip_lattice%ld		{{%.7f,%.7f,%.7f}{%.7f,%.7f,%.7f}{%.7f,%.7f,%.7f}}\n", \
			j, mat[0][0],mat[1][0],mat[2][0], mat[0][1],mat[1][1],mat[2][1], mat[0][2],mat[1][2],mat[2][2]);
		fprintf(f,"//   reciprocal matrix   %9.5f     %9.5f     %9.5f\n",mat[0][0],mat[0][1],mat[0][2]);
		fprintf(f,"//   column vectors      %9.5f     %9.5f     %9.5f\n",mat[1][0],mat[1][1],mat[1][2]);
		fprintf(f,"//                       %9.5f     %9.5f     %9.5f\n",mat[2][0],mat[2][1],mat[2][2]);
/*		fprintf(f,"//\n//     #                     G^                         (hkl)       intens      E(keV)      err(deg)\n");
		fprintf(f,"$array%ld	%3d %l3d\n",j,10,foundPattern[j].Ni); */
		fprintf(f,"//\n$array%ld	%3d %4ld             G^                         (hkl)       intens      E(keV)      err(deg)   PkIndex\n",j,10,foundPattern[j].Ni);
		for (i=0;i<(foundPattern[j].Ni);i++) {
			VECTOR_COPY3(hkl,foundPattern[j].hkls[i]);
			lowestAllowedHKL(hkl,&(foundPattern[j].xtal));
			sinTheta = -dot3(kihat,foundPattern[j].Ghat[i]);			/* sin(Bragg angle) */
			VECTOR_COPY3(vec,hkl);
			MatrixMultiply31(foundPattern[j].xtal.recip,vec,vec);
			keV = hc*NORM3(vec) / (4*M_PI*sinTheta);					/* G = 4*PI*sin(theta)/lambda, and G = 2*PI/d */
			fprintf(f,"    [%3ld]   (% 10.7f % 10.7f % 10.7f)     (%3ld %3ld %3ld)    %6.4f,    %7.4f,   %9.5f      %d\n",i, \
				foundPattern[j].Ghat[i][0],foundPattern[j].Ghat[i][1],foundPattern[j].Ghat[i][2], \
				hkl[0],hkl[1],hkl[2],foundPattern[j].intens[i],keV,foundPattern[j].err[i]*180./M_PI,foundPattern[j].pkIndex[i]);
		}
	}
	fclose(f);

return_path:
	CHECK_FREE(headerBuf);
	CHECK_FREE(GhatSpots);
	CHECK_FREE(intensSpots);
	CHECK_FREE(pkIndexSpots);
	FREE_PATTERN_OF_GRAIN(dataPattern)
	for (j=0;j<Nfound;j++) {							/* free all of the patterns found in OrientFast() */
		FREE_PATTERN_OF_GRAIN(foundPattern[j])
	}
//	FREE_PATTERN_OF_GRAIN(fitPattern)
	CHECK_FREE(fitPattern.Ghat);

	return err;
}



/* read peak positions from a file, and the associated meta data, returns number of peaks read in */
long readDataFromJZT(
char	*fname,
struct patternOfOneGrain *dataPattern,
long	maxData,			/* max number of spots to use, ie use the first maxData spots */
char	**headerBuf)		/* string with tag values to pass on to the output flie */
{
	long	Ni=0;										/* number of peaks read from file */
	FILE	*f;											/* file descriptor */
	int		i;											/* generic flag or index */
	long	SpaceGroup;									/* SpaceGroup number, ala International Tables */
	double  q1,q2,q3;									/* the 3 components of the Q vector */
	double  intens;										/* intensity at one peak */
	double  val;										/* dummy value */
	char	line[512];									/* line of data read from file */
	double  sumIntens;									/* sum of all the intensity in the pattern */
	size_t	headLen;									/* allocated length of headerBuf[] */
	char	*p;											/* generic pointer into a string */
	double	units2Angstrom=10;							/* conversion factor, converts lattice constants to Angstroms, (the default) */
	size_t	Ntype=0;									/* number of atom types in this file */
	size_t	Nalloc;										/* number of atom types allocated */

	dataPattern->goodness = 0.;
	dataPattern->alpha = 100.;							/* impossible value */
	dataPattern->beta = 100.;
	dataPattern->gamma = 100.;
	dataPattern->Ni = 0;
	dataPattern->hkls = NULL;							/* we do not know the hkls, this is just fitted spots */
	dataPattern->intens = NULL;
	dataPattern->pkIndex = NULL;
	dataPattern->Ghat = NULL;
	dataPattern->err = NULL;

	InitCleanCrystalStructure(&(dataPattern->xtal));
	dataPattern->xtal.a = dataPattern->xtal.b = dataPattern->xtal.c = ao;   /* default is for Aluminum */
	dataPattern->xtal.lengthUnits = 1.e10;				/* ao is in Angstrom */
	dataPattern->xtal.alpha = dataPattern->xtal.beta = dataPattern->xtal.gamma = M_PI_2;  /* all angles 90 deg. */
	dataPattern->xtal.SpaceGroup = FCC;					/* default to FCC */
	if (maxData<1) return 0;
	if (*headerBuf) { fprintf(stderr,"headerBuf[] is not NULL, already has space\n"); return 0; }

	if (( f = fopen(fname, "r")) == NULL) { fprintf(stderr,"Can't open file '%s'\n",fname); exit(1); }
	fgets(line,500,f);
	if (!checkFileTypeLine(line,"PeaksFile")) goto badFile;	/* check that first line is '$PeaksFile' */
//	if (strncmp(line,"$PeaksFile",10)) goto badFile;	/* check that first line is '$PeaksFile' */

	headLen = 500*10;									/* allocate space for the header */
	*headerBuf = calloc(headLen,sizeof(char));			/* work in 10 line bunches */
	if (!(*headerBuf)) { fprintf(stderr,"unable to allocate space for 'headerBuf'\n"); goto badFile; }
	(*headerBuf)[0] = '\0';								/* start this empty */
	/* read the tagged header values into headerBuf[] */
	fgets(line,250,f);
	while(strlen(line) && !strstr(line,"N_Ghat+Intens")) {
		if (line[0]=='$') {								/* only consider taged lines */
			if (headLen-strlen(*headerBuf)<500) {		/* allocate more space */
				headLen = strlen(*headerBuf) + 500*10;
				*headerBuf = realloc(*headerBuf,headLen);
				if (!(*headerBuf)) { fprintf(stderr,"unable to re-allocate more space for 'headerBuf'\n"); goto badFile; }
			}
			strcat(*headerBuf,line);					/* append line to headerBuf[], include the new-line */
		}
		fgets(line,500,f);
	}
	headLen = strlen(*headerBuf)+1;						/* trim to exact size */
	*headerBuf = realloc(*headerBuf,headLen);
	if (!(*headerBuf)) { fprintf(stderr,"unable to re-allocate more space for 'headerBuf'\n"); goto badFile; }
	p = *headerBuf;
	while((p=strchr(p,'\r'))) *p = '\n';				/* convert all carriage returns to new lines */

	dataPattern->xtal.lengthUnits = 1.e10;				/* default is Angstroms */
	if (!strFromTagBuf(*headerBuf,"lengthUnit",line,500)) {	/* search for units of the lattice parameters */
		if (!strcmp(line,"nm")) dataPattern->xtal.lengthUnits = 1.e9;				/* using nm */
		else if (strstr(line,"Ang")==line) dataPattern->xtal.lengthUnits = 1.e10;	/* using Angstrom */
		else if (strstr(line,"micron")==line) dataPattern->xtal.lengthUnits = 1.e6;	/* using micron */
		else { fprintf(stderr,"ERROR, invalid $lengthUnit = '%s',  in readDataFromJZT()\n",line); goto badFile; }
	}
	units2Angstrom = 1.e10 /dataPattern->xtal.lengthUnits;	/* set conversion factor based on xtal.lengthUnit */
//	if (!strFromTagBuf(*headerBuf,"lengthUnit",line,500)) {	/* search for units of the lattice parameters */
//		if (!strcmp(line,"nm")) {
//			units2Angstrom = 10.;						/* found nm, so multiply by 10 to get Angstroms */
//			dataPattern->xtal.lengthUnits = 1;
//		else if (strstr(line,"Ang")==line) {
//			units2Angstrom = 1.;						/* found Angstroms, so multiply by 1 to get Angstroms */
//			dataPattern->xtal.lengthUnits = 0;
//		}
//	}

	i = strFromTagBuf(*headerBuf,"latticeParameters",line,500); /* search for the lattice parameters */
	if (i) {
		fprintf(stderr,"Can't find tag for lattice parameters in file '%s', default to Aluminum\n",fname);
		dataPattern->xtal.lengthUnits = 1.e10;
		}
	else {												/* get lattice parameters from file */
		double a=0,b=0,c=0,alpha=0,beta=0,gamma=0;
		if (sscanf(line,"{%lg, %lg, %lg, %lg, %lg, %lg}",&a,&b,&c,&alpha,&beta,&gamma)!=6) {
			fprintf(stderr,"Unable to read 6 values from $latticeParameters in file '%s'\n",fname);
			goto badFile;
		}
		if (a<=0 || b<=0 || c<=0 || alpha<=0 || beta<=0 || gamma<=0 || alpha>=180 || beta>=180 || gamma>=180) {
			fprintf(stderr,"Invalid lattice parameters in file '%s'\n",fname);
			fprintf(stderr,"Lattice Parameters %g, %g, %g, %g, %g, %g invalid\n",a,b,c,alpha,beta,gamma);
			goto badFile;
		}
		dataPattern->xtal.lengthUnits = 1.e10;			/* switching to Angstrom */
		dataPattern->xtal.a = a*units2Angstrom;			/* everything is OK, set lattice */
		dataPattern->xtal.b = b*units2Angstrom;
		dataPattern->xtal.c = c*units2Angstrom;
		dataPattern->xtal.alpha = alpha*M_PI/180.;
		dataPattern->xtal.beta = beta*M_PI/180.;
		dataPattern->xtal.gamma = gamma*M_PI/180.;

		Nalloc = 100;									/* read the atomTypes */
		dataPattern->xtal.atomType = calloc(Nalloc,sizeof(dataPattern->xtal.atomType[0]));
		if (!(dataPattern->xtal.atomType)) { fprintf(stderr,"unable to allocate space for atomTypes in readDataFromJZT()\n"); goto badFile; }
		for(Ntype=0;;Ntype++) {
			double	x,y,z,occ;
			char	ele[61];
			char	tagName[120];
			sprintf(tagName,"AtomDesctiption%ld",Ntype+1);	/* tag for next atom type */
			if (strFromTagBuf(*headerBuf,tagName,line,500)) break;	/* get next atom type, break if none found */
			if (sscanf(line,"{%s  %lg %lg %lg %lg}",ele,&x,&y,&z,&occ)!=5) break;
			ele[59] = '\0';
			if (Ntype>=Nalloc) {						/* need more room, extend ->xtal.atomType[] */
				Nalloc = Ntype+100;
				dataPattern->xtal.atomType = realloc(dataPattern->xtal.atomType,Nalloc*sizeof(dataPattern->xtal.atomType[0]));
				if (!(dataPattern->xtal.atomType)) { fprintf(stderr,"unable to re-allocate space for atomTypes in readDataFromJZT()\n"); goto badFile; }
			}
			strncpy(dataPattern->xtal.atomType[Ntype].name,ele,60);
			dataPattern->xtal.atomType[Ntype].x = x;
			dataPattern->xtal.atomType[Ntype].y = y;
			dataPattern->xtal.atomType[Ntype].z = z;
			dataPattern->xtal.atomType[Ntype].occ = occ;
			dataPattern->xtal.atomType[Ntype].Zatom = atomicNumber(ele);
		}
		dataPattern->xtal.atomType = realloc(dataPattern->xtal.atomType,Ntype*sizeof(dataPattern->xtal.atomType[0]));
		dataPattern->xtal.Ntype = Ntype;
	}

	i = strFromTagBuf(*headerBuf,"SpaceGroup",line,250);/* search for the lattice SpaceGroup, set to default of FCC above */
	if (i) i = strFromTagBuf(*headerBuf,"latticeStructure",line,250);	/* old for backward compatibility, should use $SpaceGroup */
	if (i) fprintf(stderr,"Can't find tag for SpaceGroup in file '%s', default to FCC\n",fname);
	else {
		SpaceGroup = 0;
		i = sscanf(line,"%ld",&SpaceGroup);				/* get SpaceGroup, and set it if valid */
		if(i==1 && SpaceGroup>=1 && SpaceGroup<=230) dataPattern->xtal.SpaceGroup = SpaceGroup;
		else fprintf(stderr,"in file '%s', $SpaceGroup is '%s', it must be number in range [1,230], defaulting to FCC\n",fname,line);
	}

	fseek(f,0,SEEK_SET); 								/* start searching from start of file */
	i = strFromTagFile(f,"N_Ghat+Intens",line,250);		/* find the start of table, and number of points in it */
	if (i) { fprintf(stderr,"Can't find tag for peak data in file '%s'\n",fname); goto badFile; }
	if (sscanf(line,"%ld",&Ni)!=1) goto badFile;		/* read Ni from the string */
	Ni = MIN(maxData,Ni);
	/* allocate space for arrays */
	dataPattern->intens = calloc((size_t)maxData,sizeof(double));
	if (!(dataPattern->intens)) { fprintf(stderr,"unable to allocate space for 'dataPattern->intens'\n"); goto badFile; }
	dataPattern->pkIndex = calloc((size_t)maxData,sizeof(int));
	if (!(dataPattern->pkIndex)) { fprintf(stderr,"unable to allocate space for 'dataPattern->pkIndex'\n"); goto badFile; }
	dataPattern->Ghat = calloc((size_t)maxData,3*sizeof(double));
	if (!(dataPattern->Ghat)) { fprintf(stderr,"unable to allocate space for 'dataPattern->Ghat'\n"); goto badFile; }

	sumIntens = 0.;
	for (i=0;i<Ni;i++) {								/* read the file line by line, and save the Q and intensity */
		if (!fgets(line,250,f)) goto badFile;			/* cannot read, give up */
		if (sscanf(line,"%lg,   %lg,	%lg,	%lg",&q1,&q2,&q3,&intens)!=4) goto badFile;
		val = sqrt(q1*q1 + q2*q2 + q3*q3);
		if (val==0.) goto badFile;
		dataPattern->Ghat[i][0] = q1/val;				/* do this to ensure the length is really 1 to full precision */
		dataPattern->Ghat[i][1] = q2/val;
		dataPattern->Ghat[i][2] = q3/val;
		dataPattern->intens[i] = intens;
		dataPattern->pkIndex[i] = i;					/* not useful for dataPattern, useful for found Patterns */
		sumIntens += intens;
	}
	fclose(f);
	dataPattern->Ni = Ni;
	dataPattern->goodness = sumIntens>0. ? sumIntens*Ni*Ni : Ni;
	UpdateInternalsOfCrystalStructure(&(dataPattern->xtal));	/* set the reciprocal lattice, atom positions, etc. */
	return Ni;											/* return number of peaks in pattern */
	badFile:
		CHECK_FREE(*headerBuf);							/* free allocated space */
		CHECK_FREE(dataPattern->intens);
		CHECK_FREE(dataPattern->pkIndex);
		CHECK_FREE(dataPattern->Ghat);
		freeCrystalStructure(&(dataPattern->xtal));
		if (f != NULL) fclose(f);						/* ensure an opened file gets closed */
	return 0;
}


/* returns value of associated with a tag from a buffer, each tag value pair is terminated by a new line */
/* and tag is separated from its value by white space.  anything after a // is ignored, and can be used as comments in the file*/
/* tags are limited to 120 characters, and the tag+file://localhost/Users/tischler/dev/Euler/Al.txtvalue is limited to 500 characters */
int strFromTagBuf(	/* look for the tag and return 0 if it finds the tag, 1 if not found */
char	*buffer,	/* char buffer with all of the tagged strings */
char	*tagIn,		/* the tag, this should NOT have the leading '$' */
char	*value,		/* put the rest of the line following tag here, line should be at least maxLen long */
long	maxLen)		/* max length of line to pass back */
{
	char	tag[128];		/* local version of tag with leading $ */
	size_t	tagLen;			/* length of tag */
	char	*p;				/* generic pointer for string stuff */
	long	i;				/* generic index */

	tagLen = strlen(tagIn);
	if (tagLen<1 || tagLen>120) return 1;			/* empty tag is not allowed, not too long either */
	for (p=tagIn;*p>0;p++) if(*p <= ' ') return 1;	/* tag may not contain control characters or spaces */
	strcpy(tag,"$");
	strncat(tag,tagIn,125);							/* now tag[] has the leading '$' */
	tagLen++;

	/* search buffer for $tag followed by character <= 32, also tag must start a line */
	p = buffer;
	while((p=strstr(p,tag))) {						/* find $tag, then check for white space after */
		if (p!=buffer && *(p-1)!='\n') continue;	/* check if $tag at start or preceeded by new line */
		p += tagLen;								/* points to first character after $tag */
		if (*p <= ' ') break;						/* $tag followed by white space, found it */
	}
	if (!p) {value='\0'; return 1; }				/* NULL pointer, $tag was not found */
	/* p now points to first character after $tag */

	i = 0;
	while (*p>0 && *p<=' ' && *p!='\n' && i<500) { p++; i++; }	/* find the start of the value, skip white space */
	if (*p=='\n' || *p==0 || i>=500) { value[0]='\0'; return 0; } /* the tag was there, but no value part found */

	strncpy(value,p,(size_t)(maxLen-1));			/* now value contains the value part */
	value[maxLen-1] = '\0';							/* and it is definitely terminated */
	if ((p=strchr(value,'\n'))) *p = '\0';			/* trim at new line */
	if ((p=strstr(value,"//"))) *p = '\0';			/* strip off any following comment */

	/* trim off any trailing white space */
	p = value+strlen(value)+1;						/* p points to null after last character */
	while(p>value && *(p-1)<=' ') p--;
	*p = '\0';
	return 0;
}

/* returns value of associated with a tag from a file, each tag value pair is terminated by a new line */
/* and tag is separated from its value by white space.  anything after a // is ignored, and can be used as comments in the file*/
/* tags are limited to 120 characters, and the tag+value is limited to 500 characters */
int strFromTagFile(	/* look for the tag and return 0 if it finds the tag, 1 if not found */
FILE	*f,			/* file id */
char	*tagIn,		/* the tag, this should NOT have the leading '$' */
char	*value,		/* put the rest of the line following tag here, line should be at least maxLen long */
long	maxLen)		/* max length of line to pass back */
{
	char	line[512];		/* contains one line from the file */
	char	tag[128];		/* local version of tag with leading $ */
	size_t	tagLen;			/* length of tag */
	char	*p;				/* generic pointer for string stuff */
	long	i;				/* generic index */

	tagLen = strlen(tagIn);
	if (tagLen<1 || tagLen>120) return 1;			/* empty tag is not allowed, not too long either */
	for (p=tagIn;*p>0;p++) if(*p <= ' ') return 1;	/* tag may not contain control characters or spaces */
	strcpy(tag,"$");
	strncat(tag,tagIn,125);							/* now tag[] has the leading '$' */
	tagLen++;

	fseek(f,0,SEEK_SET); 							/* always start searching from start of file */
	while ((p=fgets(line,511,f))) {					/* break on EOF */
		if (strncmp(line,tag,tagLen)) continue;		/* wrong tag, continue checking */
		else if (line[tagLen]>' ') continue;		/* the tag in line is longer than tag[] */
		p = &(line[tagLen]);						/* points to start of value part */
		break;
	}
	if (!p) {value[0]='\0'; return 1; }				/* NULL pointer, this mainly catches the EOF */

	i = tagLen;
	while (*p>0 && *p<=' ' && i<500) { p++; i++; }	/* find the start of the value, skip white space */
	if (*p==0 || i>=500) { value[0]='\0'; return 0; } /* the tag was there, but no value part found */

	strncpy(value,p,(size_t)(maxLen-1));			/* now value contains the value part */
	value[maxLen-1] = '\0';							/* and it is terminated */
	if ((p=strstr(value,"//"))) *p = '\0';			/* strip off any following comment */

	/* trim off any trailing white space */
	p = value+strlen(value)+1;						/* p points to null after last character */
	while(p>value && *(p-1)<=' ') p--;
	*p = '\0';
	return 0;
}



/* reads in data from a Wenge file.  It contains no meta data, just some fixed columns */
long readDataFromWenge(
char	*fname,
struct patternOfOneGrain *dataPattern,
long	maxData)			/* max number of spots to use, ie use the first maxData spots */
{
	long	Ni=0;
	FILE	*f;								/* file descriptor */
	int		i;
	double  q1,q2,q3;
	double  val;							/* dummy value */
	char	line[256];

	dataPattern->goodness = 0.;
	dataPattern->alpha = 100.;				/* impossible value */
	dataPattern->beta = 100.;
	dataPattern->gamma = 100.;
	dataPattern->Ni = 0;
	dataPattern->hkls = NULL;				/* we do not know the hkls, this is just fitted spots */
	dataPattern->intens = NULL;
	dataPattern->pkIndex = NULL;
	dataPattern->Ghat = NULL;
	dataPattern->err = NULL;
	if (maxData<1) return 0;
#if (DEBUG)
	fprintf(fout,"reading data fom a 'Wenge' style file, so a lot of parameters are defaulted\n");
#endif

	dataPattern->intens = calloc((size_t)maxData,sizeof(double));
	if (!(dataPattern->intens)) { fprintf(stderr,"unable to allocate space for 'dataPattern->intens'\n"); return 0; }
	dataPattern->pkIndex = calloc((size_t)maxData,sizeof(int));
	if (!(dataPattern->pkIndex)) { fprintf(stderr,"unable to allocate space for 'dataPattern->pkIndex'\n"); return 0; }
	dataPattern->Ghat = calloc((size_t)maxData,3*sizeof(double));
	if (!(dataPattern->Ghat)) { fprintf(stderr,"unable to allocate space for 'dataPattern->Ghat'\n"); free(dataPattern->intens); free(dataPattern->pkIndex); return 0; }

	if (( f = fopen(fname, "r")) == NULL) { fprintf(stderr,"Can't open file '%s'\n",fname); exit(1); }
	for (i=0;i<4;i++) fgets(line,250,f);	/* just skip first 4 lines */

	Ni = 0;
	do {
		q1 = q2 = q3 = 0.;
		if (!fgets(line,250,f)) break;
		else if ((i=sscanf(line,"%lg %lg %lg %lg %lg %lg %lg %lg",&val,&val,&val,&val,&val,&q1,&q2,&q3))==8) {
			val = sqrt(q1*q1 + q2*q2 + q3*q3);
			if (val==0.) continue;
			dataPattern->Ghat[Ni][0] = q1/val;
			dataPattern->Ghat[Ni][1] = q3/val;
			dataPattern->Ghat[Ni][2] = -q2/val;
			dataPattern->intens[Ni] = 0.;		/* no intensity in file, so set to zero */
			dataPattern->pkIndex[Ni] = i;		/* not useful for dataPattern, useful for found Patterns */
			Ni++;
		}
	} while(i==8 && Ni<maxData);

	fclose(f);
	dataPattern->Ni = Ni;
	dataPattern->goodness = Ni;
	return Ni;
}



long FindClosestMeasuredG(		/* returns index into Gmeas[][3] which is closest to hkl, -1 means failure */
double  recip[3][3],			/* rotated(by Euler angles) reciprocal lattice, recip*hkl = Gdesired */
int		hkls[3],				/* desired hkl */
size_t  Nmeas,					/* number of Gmeas */
double  (*Gmeas)[3])			/* list of measured G vectors */
{
	size_t  i;
	long  index=-1;					/* index into Gmeasured of closest G, the returned value */
	double  G[3];					/* calculated G vector for each hkl */
	double  hkl[3];					/* float version of hkl */
	double  dot;
	double  max;

	VECTOR_COPY3(hkl,hkls);			/* convert integer hkl to float for following MatrixMultiply31 */
	MatrixMultiply31(recip,hkl,G);
	normalize3(G);					/* normalized gvec of hkl[i] rotated by Euler angles */
	for (i=0,max=-1.;i<Nmeas;i++) {
		dot = dot3(G,Gmeas[i]);
		if (dot>max) {				/* the biggest dot product is the closest G vector */
			max = dot;
			index = i;
		}
	}
	return index;
}



/* This routine takes a list of data unit vectors, and produces a list of EulerAngles for the different grains that fit */
/* returns 0 for no error, true means an error happened */
int OrientFast(
long	size,				/* size of the Euler Space  (2*size,size,2*size) */
double  keVmaxCalc,			/* maximum energy (keV) when calculating fit */
double  keVmaxTest,			/* limit in energy for finding test points (Gtest), probably [2^(1/3)]*keVmaxCalc */
double  angleTolerance,		/* difference in angles between G-vector pairs to be considered coincident (radians) */
int		hkl0[3],			/* preferred hkl at center of pattern */
double  cone,				/* acceptable cone angle from preferred hkl0 */
struct crystalStructure *xtal,  /* the structure to find */
long	Ni,					/* number of measured spots */
double  (*GhatSpots)[3],	/* direction of each spot on the detector */
int		*pkIndexSpots,		/* data spot index */
/* double  *intensSpots,		// intensity of each spot on the detector */
long	*Nfound,			/* number of valid patterns found and stored in foundPattern[] */
struct	patternOfOneGrain foundPattern[MAX_GRAINS_PER_PATTERN]) /* hold the fitted patterns */
{
	int		err=0;							/* error flag returned, 0=OK */
	struct box_struct abgRange;				/* structure giving the allowed ranges of alpha, beta, and gamma */
	struct dotPlusIndicies *dotList=NULL;   /* list of all dot products between any two data spots */
	long	NdotList;						/* number of elemnts in dotList[] */
	long	Nm;								/* number of possible G vectors, length of Gmhat[], GmLen[],  and hklm[] */
	double  (*Gmhat)[3]=NULL;
	double  *GmLen=NULL;
	int		(*hklm)[3]=NULL;				/* holds the hkl for corresponding to each of the Gmhat[][3] */
	short int *indexedData=NULL;			/* array of flags to keep track of which spots have been indexed */
			/* 	changed indexedData from char to short int fixed problem with EXC_BAD_ACCESS in foundPattern[0].xtal.equiv.equivXYZM[][][] ??? */
	double	dAnglePixel;					/* min angular resolution of detector (radian), this limits zooming, does not influence paralled  */
	struct EulerAngle_pair *AllEulerAngles=NULL; /* pointer to an array of EulerAngle_pair structures */
	size_t  Neuler;							/* number of slots allocated in AllEulerAngles */
	double  EulerAngles[3];					/* set of three Euler angles */
	double  dEuler;
	int		type=1;							/* type of peak to use in PeakInHough() */
	double  mag;							/* magnification for an Euler space */
	long	itest, id, i;
	size_t	n;
	long	Nremain=Ni;						/* number of measured spots NOT assigned to patterns */
	long	Ncurrent;						/* number of spots assigned to the current pattern */
	double  peakValue;
	struct WaveSpace_struct EulerSpace;		/* pointer to an array of EulerAngle_pair structures */
	double  parallel;						/* cos(2*angleTolerance), if a dot is > parallel, then vectors are assumed in same direction */
	struct box_struct gRange;				/* structure giving the allowed 3d values of (G^ .dot. x^), and also for y^ and z^ */
	double  (*Gtest)[3]=NULL;				/* directions of calculated test spots on the detector */
	int		(*hklTest)[3]=NULL;				/* list of hkl associated with Gtest[][3] */
	char	*foundGtest=NULL;				/* marks which of the Gtest[][3] (and hklTest[][3]) have been identified */
	double	*intensTest=NULL;				/* intensity that goes with Gtest */
	int		*pkIndexTest=NULL;				/* data spot index that goes with Gtest */
	double  *errTest=NULL;					/* error (rad) between predicted spot and measured spot (measured - prdicted) */
	long	Ntest;							/* number of 3-vectors in Gtest */
	double  farDot;							/* dot product between two data spots that are farthest from each other */
	double  goodness;						/* goodness of a pattern, (sum of intensities)*(number of spots)^2 */
	double  keV;
	double  intens;
	double  intensMax;
	double  dot;							/* hold vaue of dot3(Gtest[itest],GhatSpots[id]) */
#if (DEBUG)
	double  seconds;						/* used to print execution time */
	clock_t time0;
	char	str[256];						/* string used to show time */
	time0 = clock();
#endif

	if (!(size>1 && size<200)) { fprintf(stderr,"size=%ld is illegal\n\a",size); err=1; goto return_path; }
	if (Ni < 1) { fprintf(stderr,"There are no detected spots in GhatSpots\n"); err=1; goto return_path; }

	angleTolerance = fabs(angleTolerance);
	cone = fabs(cone);
	cone = (cone==M_PI/2.) ? cone-0.001 : cone; /* for 90 deg, make it a little less */
	cone = (cone>M_PI) ? cone-1e-10 : cone;		/* cannot have cone > 180 deg */
	/* these min and max angles have been chosen for a cubic crystal, AND THEY ARE PROBABLY WRONG */
	/*	alphaMin = -M_PI/4;  	alphaMax = M_PI/4;		// range of alpha	[-45°,45°) */
	/*	betaMin = 0;			betaMax = 1;			// range of beta	[0°,57°)   , 57.3° ~ (001)^(111), for Cubic xtals */
	/*	gammaMin = -M_PI/2;		gammaMax = M_PI/2;		// range of gamma	[-90°,90°) */
	/*	alphaMin *= 1.01; alphaMax *= 1.01; betaMin -= 0.01; betaMax += 0.01; gammaMin *= 1.01; gammaMax *= 1.01; */
	abgRange.xlo = ALPHAMIN-0.01;  abgRange.xhi = ALPHAMAX+0.01;
	abgRange.ylo  = BETAMIN-0.01;  abgRange.yhi  = BETAMAX+0.01;
	abgRange.zlo = GAMMAMIN-0.01;  abgRange.zhi = GAMMAMAX+0.01;

	NdotList = MakeDotList(Ni,GhatSpots,&dotList);		/* make list of dots between any two data spots */
	if (NdotList<1) { err=(NdotList?NdotList:1); goto return_path; }
	farDot = dotList[0].dot;
	/*	farDot = dotOfBiggestAngle(Ni,GhatSpots); */
#if (DEBUG)
	fprintf(fout,"from the data, MakeDotList produced %ld pairs in dotList[]  (from Ni=%ld)\n",NdotList,Ni);
	fprintf(fout,"the largest angle between two data G vectors are is %.2f (deg), with a dot of %g\n",acos(farDot)*180/M_PI,farDot);
	fprintf(fout,"\n---> run MakeAllPossibleGvectors(), to make Gmhat[Nm][3], hklm[Nm][3], and GmLen[Nm]\n");
#endif
	MakeAllPossibleGvectors(&Nm,&Gmhat,&GmLen,&hklm,hkl0,cone,keVmaxCalc,xtal);
	if (Nm<=0) goto return_path;

#if (DEBUG)
	fprintf(fout,"\n---> run MakeListOfAllEulerAngles()\n");
#endif
/*	MakeListOfAllEulerAngles(abgRange,angleTolerance,Ni,GhatSpots,farDot,NdotList,dotList,Nm,Gmhat,GmLen,&AllEulerAngles,&Neuler); */
	MakeListOfAllEulerAngles(abgRange,angleTolerance,GhatSpots,farDot,NdotList,dotList,Nm,Gmhat,&AllEulerAngles,&Neuler);
	
	/* create the EulerSpace, and initialize it, this is the array to hold the Hough transform */
	EulerSpace.aoff = abgRange.xlo; EulerSpace.boff = abgRange.ylo;  EulerSpace.goff = abgRange.zlo;
	EulerSpace.Na = 2*size;		EulerSpace.Nb = size;		EulerSpace.Ng = 2*size;
	EulerSpace.da = (abgRange.xhi-abgRange.xlo)/EulerSpace.Na;
	EulerSpace.db = (abgRange.yhi-abgRange.ylo)/EulerSpace.Nb;
	EulerSpace.dg = (abgRange.zhi-abgRange.zlo)/EulerSpace.Ng;
	EulerSpace.space = calloc((size_t)(EulerSpace.Na),sizeof(long **));
	for (n=0;n<(size_t)EulerSpace.Na;n++) {
		EulerSpace.space[n] = calloc((size_t)(EulerSpace.Nb),sizeof(long *));
	}
	for (n=0;n<(size_t)EulerSpace.Na;n++) {
		for (i=0;i<EulerSpace.Nb;i++) {
			EulerSpace.space[n][i] = calloc((size_t)(EulerSpace.Ng),sizeof(long));
		}
	}
	*Nfound = 0;

	/* dAnglePixel determines how far in to zoom the EulerSpace. */
	dAnglePixel = dANGLE_PIXEL;			/*  = 0.025/100./10.,  angle between pixels, this is the ultimate limit on angular precision (radian) */
	parallel = cos(2.*angleTolerance);  /* used for match a measured spot to a calculated one */
	parallel = MIN(parallel,1.);
#if (DEBUG)
	fprintf(fout,"coincidence angle tolerance between G^ pairs is %.4f (deg),   parallel=%.10f (=%.3g deg)\n",angleTolerance*180/M_PI,parallel,acos(parallel)*180/M_PI);
	fprintf(fout," the highest reachable reflection is the %s\n",HighestReachableHKL(keVmaxCalc,THETA_MAX,xtal,str));
#endif

	SetGhatRangeFromDataSpots(Ni,GhatSpots,&gRange);		/* set 'gRange' the limits on G^ that includes all measured spots */
#if (DEBUG>2)
	fprintf(fout,"gRange = [%g, %g] [%g, %g] [%g, %g]   vol = %g\n",gRange.xlo,gRange.xhi,gRange.ylo,gRange.yhi,gRange.zlo,gRange.zhi, \
		(gRange.xhi-gRange.xlo)*(gRange.yhi-gRange.ylo)*(gRange.zhi*gRange.zlo));
#endif
	indexedData = (short int *) calloc((size_t)Ni,sizeof(*indexedData));
	if (!indexedData) { fprintf(stderr,"could not allocate space for indexedData in OrientFast()\n"); err=1; goto return_path; }
	for (i=0;i<Ni;i++) indexedData[i]=0;					/* start with all spots marked as unindexed */
	/*  0 = this measured spot not assigned to any pattern
	 *  1 = assigned to a valid pattern
	 * -2 = used by current pattern, but we have not yet determined if this pattern is valid
	 * -1 = conditionally assigned in a previous iteration, but the pattern was invalid, so this hkl can be claimed by subsequent patterns
	 */
	do {													/* main loop, each time thru this finds one grain (i.e. one pattern) */
#if (DEBUG)
		seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;
		fprintf(fout,"\n\n--in OrientFast(), time at top of do loop {find pattern %ld} %s  =  (%.2f sec)\n",*Nfound,num2sexigesmal(str,seconds,0),seconds);
#if (DEBUG>1)
		fprintf(fout,"will be testing %ld (=Neuler) pairs of spots\n",Neuler);
#endif
#endif
		foundPattern[*Nfound].Ni=0;
		/* reset to full, un-zoomed range */
		mag = 1;											/* use mag=1 for the first pass, zoom in on later passes */
		EulerSpace.aoff = abgRange.xlo; EulerSpace.boff = abgRange.ylo;  EulerSpace.goff = abgRange.zlo;
		/* EulerSpace.Na = 2*size;		EulerSpace.Nb = size;		EulerSpace.Ng = 2*size; */
		EulerSpace.da = (abgRange.xhi-abgRange.xlo)/EulerSpace.Na;
		EulerSpace.db = (abgRange.yhi-abgRange.ylo)/EulerSpace.Nb;
		EulerSpace.dg = (abgRange.zhi-abgRange.zlo)/EulerSpace.Ng;
		EulerAngles[0] = abgRange.xlo + (EulerSpace.da)*(EulerSpace.Na)/2;  /* center of region (radian) */
		EulerAngles[1] = abgRange.ylo  + (EulerSpace.db)*(EulerSpace.Nb)/2;
		EulerAngles[2] = abgRange.zlo + (EulerSpace.dg)*(EulerSpace.Ng)/2;
		do {												/* refinement loop, each time thru this refines angles of one grain */
			magnifyEulerSpace(&EulerSpace,mag,EulerAngles); /* magnify about the last point */
/*			EulerSpaceFromSpotsFast(&EulerSpace,AllEulerAngles,Neuler,EulerAngles); // fill the EulerSpace array based on angles */
			EulerSpaceFromSpotsFast(&EulerSpace,AllEulerAngles,Neuler); /* fill the EulerSpace array based on angles */
			peakValue = PeakInHough(&EulerSpace,EulerAngles,type);	/* type=1 for peak,  2 for Center of Mass, 3 for COM around peak, 4 COM of peakish part (returns scaled angles) */
			if (peakValue < MIN_PEAK_IN_HOUGH) {			/* no point in looking further, this was 1 */
#if (DEBUG>1)
				fprintf(fout,"   breaking with a peakValue = %g\n",peakValue);
#endif
				/* undo the last mag, and get the center */
				magnifyEulerSpace(&EulerSpace,1./mag,EulerAngles); /* magnify about the last point */
/*				EulerSpaceFromSpotsFast(&EulerSpace,AllEulerAngles,Neuler,EulerAngles); // fill the EulerSpace array based on angles */
				EulerSpaceFromSpotsFast(&EulerSpace,AllEulerAngles,Neuler); /* fill the EulerSpace array based on angles */
				break;
			}
			dEuler= MAX(EulerSpace.da,EulerSpace.db);
			dEuler= MAX(dEuler,EulerSpace.dg);

#if (DEBUG>1)
			fprintf(fout,"with dEuler of %.4f(deg),   found orientation at: (%.3f,  %.3f,  %.3f) with peak=%g\n",dEuler*180/M_PI,EulerAngles[0]*180/M_PI,EulerAngles[1]*180/M_PI,EulerAngles[2]*180/M_PI,peakValue);
#endif
			mag = floor(size/4.);							/* magnify about peak by (size/4), was (size/2) */
		} while (dAnglePixel < dEuler);						/* end of refinement loop */

		peakValue = PeakInHough(&EulerSpace,EulerAngles,4);	/* type=1 for peak,  2 for Center of Mass, 3 for COM around peak, 4 COM of peakish part */
#if (DEBUG)
		seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;
		fprintf(fout,"\nin OrientFast(), found orientation of pattern %ld at: (%.3f, %.3f, %.3f) with peak=%g\n",*Nfound,EulerAngles[0]*180/M_PI,EulerAngles[1]*180/M_PI,EulerAngles[2]*180/M_PI,peakValue);
		fprintf(fout,"\tusing a lattice of a=%g, b=%g, c=%g, and alpha=%g, beta=%g, gammma=%g\n",xtal->a,xtal->b,xtal->c,180/M_PI*xtal->alpha,180/M_PI*xtal->beta,180/M_PI*xtal->gamma);
#endif
		Ntest = MakeLauePatternUnitVectors(EulerAngles[0],EulerAngles[1],EulerAngles[2],gRange,xtal,&Gtest,&hklTest,hkl0,cone,keVmaxTest);
		if (Ntest<1) break;
#if (DEBUG)
		fprintf(fout,"MakeLauePatternUnitVectors() produced %ld (=Ntest) test vectors (out to %g keV) that can hit the detector\n",Ntest,keVmaxTest);
#if (DEBUG>3)
		fprintf(fout,"\nto test this set of angles in OrientFast() using Gtest[%ld][3]:\n",Ntest);
		for (i=0;i<MIN(Ntest,56);i++) fprintf(fout,"Gtest[%ld][] = (%8.5f, %8.5f, %8.5f)  (%3d %3d %3d )\n",i,Gtest[i][0],Gtest[i][1],Gtest[i][2],hklTest[i][0],hklTest[i][1],hklTest[i][2]);
#endif
#endif
		if (foundGtest) foundGtest = realloc(foundGtest,sizeof(char)*Ntest);
		else foundGtest = calloc((size_t)Ntest,sizeof(char)); 
		if (!foundGtest) { fprintf(stderr,"Unable to allocate space for foundGtest in OrientFast()\n"); err=1; goto return_path; }
		intensTest = calloc((size_t)Ntest,sizeof(double)); 
		if (!intensTest) { fprintf(stderr,"Unable to allocate space for intensTest in OrientFast()\n"); err=1; goto return_path; }
		pkIndexTest = calloc((size_t)Ntest,sizeof(int));
		if (!pkIndexTest) { fprintf(stderr,"Unable to allocate space for pkIndexTest in OrientFast()\n"); err=1; goto return_path; }
		errTest = calloc((size_t)Ntest,sizeof(double));
		if (!errTest) { fprintf(stderr,"Unable to allocate space for errTest in OrientFast()\n"); err=1; goto return_path; }
		for (i=0;i<Ntest;i++) { foundGtest[i]=0; errTest[i]=intensTest[i]=0.; pkIndexTest[i]=-1; } /* init to nothing found */

#if (DEBUG>1)
#if (DEBUG>2)
		fprintf(fout,"\nNremain = %ld, Ni=%ld, Ntest=%ld, Neuler=%ld\n",Nremain,Ni,Ntest,Neuler);
#endif
		seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;
		fprintf(fout,"--OrientFast(), sort out which of the %ld test peaks match one of the measured peaks in pattern %ld,   %s  =  (%.2f sec)\n",Ntest,*Nfound,num2sexigesmal(str,seconds,0),seconds);
		fprintf(fout,"start looping with Ni=%ld,  Ntest=%ld,  Neuler=%ld\n",Ni,Ntest,Neuler);
#endif
		/*	find list of peaks that occur in both the just created Gtest[][3] and remain in data spots GhatSpots[][3] */
		/* and mark them with a 1 in indexedData */

		for (i=0,Nremain=0;i<Ni;i++) Nremain += (indexedData[i] ? 0 : 1);/* number data spots remaining to be assigned to patterns, at most Ni */
		Ncurrent = 0;
		for (id=Ni-1;id>=0;id--) {									/* outside loop over all input data spots */
			if (indexedData[id]==1) continue;						/* skip this data spot, already assigned to a valid pattern */
			for (itest=0;itest<Ntest;itest++) {
				if ((dot=DOT3(Gtest[itest],GhatSpots[id])) >= parallel) { /* found a match, delete this data point from future histograms */
					indexedData[id] = -2;							/* conditionally assign this point to the current pattern */
					foundGtest[itest] = 1;							/* mark data point as identified */
					reflectionProperties(Gtest[itest],hklTest[itest],xtal,&keV,&intens);
					intensTest[itest] = intens;
					pkIndexTest[itest] = pkIndexSpots[id];
					errTest[itest] = acos(dot);
					/* remove all points in 'AllEulerAngles' that use this data spot for i or i0 */
#ifdef __GNUC__
					#warning "this 'for()' loop is especially slow, the 'if()'is the slow part, not the '=-1'"
#endif
					for (n=0;n<Neuler;n++) {					/* remove all rows identified with spot id */
						long	itemp,itemp0;						/* this itemp stuff is a pitiful attempt to speed things up */
						itemp = AllEulerAngles[n].i;
						itemp0 = AllEulerAngles[n].i0;
						/* AllEulerAngles[Neuler].xx =  list of (m0,m,i0,i,alpha,beta,gamma) */
/*						if (AllEulerAngles[n].i==id || AllEulerAngles[n].i0==id) {  // remove spots with id either i or i0 */
						if (itemp==id || itemp0==id) {  /* remove spots with id either i or i0 */
							AllEulerAngles[n].i = AllEulerAngles[n].i0 = -1;		/* flags point for deletion */
						}
					}		/* end for(n) */


					Ncurrent++;
					Nremain--;
					break;
				}			/* endif spot and Gtest parallel */
			}				/* end for(itest) */
		}					/* end for(id) */
		id = id < 0 ? 0 : id;									/* if no break, then id may be <0 */

		/* now do the DeletePoints() for AllEulerAngles[] */

#if (DEBUG)
		seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;
		fprintf(fout,"in OrientFast(), marked found spots of pattern %ld,   %s  =  (%.2f sec)\n",*Nfound,num2sexigesmal(str,seconds,0),seconds);
#if (DEBUG>1)
		fprintf(fout,"%ld data spots marked as 'found' in indexedData[]\n",Ncurrent);
#endif
#endif

		for (i=itest=0;i<Ntest;i++) itest += foundGtest[i];		/* itest is now the number of spots that match in this patterns */
		if (!isNAN(EulerAngles[0]+EulerAngles[1]+EulerAngles[2]) && itest>=NUM_SPOTS_FOR_MATCH) { /* match does not count unless enough spots match */
			for(i=(Ntest-1);i>=0;i--) {							/* remove unused points */
				if (foundGtest[i]) continue;
				DeletePoints((size_t)(Ntest-i),&(Gtest[i][0]),	sizeof(Gtest[0]),1);
				DeletePoints((size_t)(Ntest-i),&(hklTest[i][0]),  sizeof(hklTest[0]),1);
				DeletePoints((size_t)(Ntest-i),&(intensTest[i]),  sizeof(intensTest[0]),1);
				DeletePoints((size_t)(Ntest-i),&(pkIndexTest[i]),  sizeof(pkIndexTest[0]),1);
				DeletePoints((size_t)(Ntest-i),&(errTest[i]),		sizeof(errTest[0]),1);
				Ntest--;
			}

			for (i=0;i<itest;i++) lowestOrderHKLint(hklTest[i]);	/* change hklTest[][3] so hkls are all lowest order */
			foundPattern[*Nfound].alpha = EulerAngles[0];
			foundPattern[*Nfound].beta  = EulerAngles[1];
			foundPattern[*Nfound].gamma = EulerAngles[2];
			foundPattern[*Nfound].Ni    = itest;
			foundPattern[*Nfound].hkls  = hklTest;
			foundPattern[*Nfound].Ghat  = Gtest;
			foundPattern[*Nfound].intens= intensTest;
			foundPattern[*Nfound].pkIndex = pkIndexTest;
			foundPattern[*Nfound].err   = errTest;
			intensMax = intensTest[i];
			for (i=0;i<itest;i++) intensMax = MAX(intensMax,intensTest[i]);	/* find maximum intensity */
			for (i=0;i<itest;i++) intensTest[i] /= intensMax;	/* normalize to max of 1 */
			goodness = 0.;
			for (i=0;i<itest;i++) goodness += intensTest[i];	/* set goodness for this pattern */
			goodness = goodness>0. ? goodness*itest*itest : itest;
			foundPattern[*Nfound].goodness = goodness;
			copyCrystalStructure(&(foundPattern[*Nfound].xtal),xtal);
			/* the functions in FillRecipInLattice should be taken care of by copyCrystalStructure() */
			/* FillRecipInLattice(&(foundPattern[*Nfound].lattice)); // set the reciprocal lattice vectors */
//printf("foundPattern[0].xtal.equiv.equivXYZM[3][0][0] = %g\n",foundPattern[0].xtal.equiv.equivXYZM[3][0][0]);	/* worked */
			for (i=0;i<Ni;i++)									/* this pattern is valid, change conditional(-2) --> used(1) */
				indexedData[id] = indexedData[i]==-2 ? 1 : indexedData[i];
//#warning "// This line gives an ERROR with EuAlO3 when indexedData is char, changing indexedData to short int fixes things"
//int Neq = foundPattern[0].xtal.equiv.N;
//fprintf(stdout,"Neq=%d\n",Neq);	/* Neq is 4, so max index is 3 */
//printf("foundPattern[0].xtal.equiv.equivXYZM[3][0][0] = %g\n",foundPattern[0].xtal.equiv.equivXYZM[3][0][0]);	/* failed with EXC_BAD_ACCESS */
			Gtest = NULL;
			hklTest = NULL;
			intensTest = errTest = NULL;
			pkIndexTest = NULL;
			(*Nfound)++;
		}
		else {
			for (i=0;i<Ni;i++)									/* this pattern is NOT valid , change conditional(-2) to available(-1) */
				indexedData[id] = indexedData[i]==-2 ? -1 : indexedData[i];
			CHECK_FREE(Gtest);									/* did not save in foundPattern[], so free it here */
			CHECK_FREE(hklTest);
			CHECK_FREE(intensTest);
			CHECK_FREE(pkIndexTest);
			CHECK_FREE(errTest);
		}
	} while (Ncurrent>0 && Nremain>MIN_SPOTS_FOR_1_PATTERN && *Nfound<MAX_GRAINS_PER_PATTERN && itest>2);
	if (*Nfound>=MAX_GRAINS_PER_PATTERN) fprintf(stderr,"hit %ld, the maximum number of grains in one Laue Pattern!!, check 'MAX_GRAINS_PER_PATTERN'\n",*Nfound);

	/* now reorder the grains in order of decreasing goodness */
	/* here, goodnes is the (sum of intensities) * (number of spots) */
	/* note that goodness needs to be first thing in struct for this to work */
	qsort(foundPattern,(size_t)(*Nfound),sizeof(foundPattern[0]),compare_double_reverse);


return_path:
	CHECK_FREE(Gtest);
	CHECK_FREE(hklTest);
	CHECK_FREE(intensTest);
	CHECK_FREE(pkIndexTest);
	CHECK_FREE(errTest);
	CHECK_FREE(dotList);
	CHECK_FREE(indexedData);
	CHECK_FREE(foundGtest);
	CHECK_FREE(AllEulerAngles);
	CHECK_FREE(Gmhat);
	CHECK_FREE(GmLen);
	CHECK_FREE(hklm);
	if (EulerSpace.space) {
		for (n=0;n<(size_t)EulerSpace.Na;n++) {
			for (i=0;i<EulerSpace.Nb;i++) CHECK_FREE(EulerSpace.space[n][i]);
			CHECK_FREE(EulerSpace.space[n]);
		}
		CHECK_FREE(EulerSpace.space);
	}
	return err;			/* return with no error */
}



/* changes hkl[3] to the lowest order hkl, ignores whether a reflection is allowed, just removes common factors */
void lowestOrderHKLint(
int		hkl[3])
{
	long	hklLong[3];

	hklLong[0] = hkl[0];
	hklLong[1] = hkl[1];
	hklLong[2] = hkl[2];
	lowestOrderHKL(hklLong);
	hkl[0] = hklLong[0];
	hkl[1] = hklLong[1];
	hkl[2] = hklLong[2];
}



/* magnify the EulerSpace centered on center[3], and maginfy by mag */
int magnifyEulerSpace(
struct WaveSpace_struct *EulerSpace,	/* pointer to the existing EulerSpace structure */
double	mag,							/* zoom in by magnification 'mag' about (a0,b0,g0), these only affect the xyz scaling */
double  center[3])						/* angles at center of zoomed region (radian) */
{
	double  a0, b0, g0;								/* local center of zoomed region (radian) */
	double	da,db,dg;								/* new values for the DimDelta of EulerSpace */
	double	alphaMin, betaMin, gammaMin;			/* these values suitable for cubic crystals */
	long	Na,Nb,Ng;								/* local versions of values in EulerSpace */

	a0=center[0]; b0=center[1]; g0=center[2];					/* set local values of the center */
	Na = EulerSpace->Na; Nb = EulerSpace->Nb; Ng = EulerSpace->Ng; /* save local values */
	da = EulerSpace->da; db = EulerSpace->db; dg = EulerSpace->dg;

	alphaMin = EulerSpace->aoff;								/* first set min and max to current values */
	betaMin = EulerSpace->boff;
	gammaMin = EulerSpace->goff;
	if (alphaMin>a0 || a0>(alphaMin+Na*da) || betaMin>b0 || b0>(betaMin+Nb*db) || gammaMin>g0 || g0>(gammaMin+Ng*dg)) {
		fprintf(stderr,"Cannot magnify about a point outside of main region\n");
		return 1;
	}
	da /= mag; db /= mag; dg /= mag;							/* new zoomed step size */
	alphaMin = a0 - da*Na/2;									/* set to zoomed in values */
	betaMin = b0 - db*Nb/2;
	gammaMin = g0 - dg*Ng/2;
#if (DEBUG>2)
	fprintf(fout,"\n");
	fprintf(fout,"centered about (%.2f, %.2f, %.2f)(deg)  or  (%.4f, %.4f, %.4f)rad\n",a0*180/M_PI,b0*180/M_PI,g0*180/M_PI,a0,b0,g0);
#endif
	EulerSpace->aoff = alphaMin;	EulerSpace->da = da;		/* set values of structure to the new zoomed values */
	EulerSpace->boff = betaMin;		EulerSpace->db = db;
	EulerSpace->goff = gammaMin;	EulerSpace->dg = dg;
	return 0;
}


/* fill the EulerSpace array based on angles in 'AllEulerAngles' */
int EulerSpaceFromSpotsFast(
struct WaveSpace_struct *EulerSpace,	/* pointer to the existing EulerSpace structure */
struct EulerAngle_pair *AllEulerAngles, /* pointer to an array of EulerAngle_pair structures */
size_t	Neuler)							/* number of elements in AllEulerAngles */
/* double  center[3])					// angles at center of zoomed region (radian) */
{
	double	da,db,dg;											/* new values for the DimDelta of EulerSpace */
	double	alphaMin,alphaMax, betaMin,betaMax, gammaMin,gammaMax;	/* these values suitable for cubic crystals */
	long	Na,Nb,Ng;											/* local versions of values in EulerSpace */
	long	i,j,k;
	double	alpha,beta,gamma;									/* three Euler angles, loop over alpha, beta, solve for gamma */
	long	ia,ib,ig;
	size_t	n;
#if (DEBUG)
	double  seconds;											/* used to print execution time */
	clock_t time0;
	char	str[256];											/* string used to show time */
	time0=clock();
#endif

	/*	Create the space for the Hough transform */
	Na = EulerSpace->Na; Nb = EulerSpace->Nb; Ng = EulerSpace->Ng; /* save local values */
	da = EulerSpace->da; db = EulerSpace->db; dg = EulerSpace->dg;

	alphaMin = EulerSpace->aoff;								/* first set min and max to current values */
	betaMin = EulerSpace->boff;
	gammaMin = EulerSpace->goff;
	alphaMax = alphaMin + Na*da;
	betaMax = betaMin + Nb*db;
	gammaMax = gammaMin + Ng*dg;

	for (i=0;i<Na;i++) {										/* EulerSpace = 0 */
		for (j=0;j<Nb;j++) {
			for (k=0;k<Ng;k++) {
				EulerSpace->space[i][j][k] = 0;
			}
		}
	}

#if (DEBUG>2)
	fprintf(fout,"$$$Range of EulerSpace[%.2f,%.2f)[%.2f,%.2f)[%.2f,%.2f)(deg)\n",alphaMin*180/M_PI,alphaMax*180/M_PI, betaMin*180/M_PI,betaMax*180/M_PI, gammaMin*180/M_PI,gammaMax*180/M_PI);
	fprintf(fout,"   Range of EulerSpace[%.4f,%.4f)[%.4f,%.4f)[%.4f,%.4f)rad\n",alphaMin,alphaMax, betaMin,betaMax, gammaMin,gammaMax);
	fprintf(fout,"   DimDelta's = (%.4f, %.4f, %.4f)(deg)   or   %.6f, %.6f, %.6f)rad\n",da*180/M_PI,db*180/M_PI,dg*180/M_PI,da,db,dg);
#endif
	/*	over all EulerSpace, accumulate the Hough transform
	 */
	for (n=0;n<Neuler;n++) {									/* AllEulerAngles[Neuler][i] =  list of (m0,m,i0,i,alpha,beta,gamma) */
		if (AllEulerAngles[n].i < 0) continue;					/* means that point has 'been deleted', so skip it */
		alpha = AllEulerAngles[n].alpha;
		beta = AllEulerAngles[n].beta;
		gamma = AllEulerAngles[n].gamma;
		if (alpha<alphaMin || alpha>=alphaMax || beta<betaMin || beta>=betaMax) continue;
		/* now that we have the EulerAngles, increment correct point in EulerSpace */
		ia = round( (alpha-alphaMin) / da );
		ib = round( (beta -betaMin)  / db );
		ig = round( (gamma-gammaMin) / dg );
		if (ia<0 || ia>=Na  ||  ib<0 || ib>=Nb  ||  ig<0 || ig>=Ng) continue;
		((EulerSpace->space)[ia][ib][ig])++;
	}
#if (DEBUG)
	seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;
	if (seconds > 0.5) fprintf(fout,"filling EulerSpace takes %s  =  (%.2f sec)\n",num2sexigesmal(str,seconds,0),seconds);
#endif
	return 0;
}



/* Creates a list of the directions of spots in a Laue pattern expected on a detector.  It creates 2 waves GhatSpots and hklSpots
 *	GhatSpots		a list of 3 vectors containing the direction vector of each of the spots
 *  hklSpots		a list of 3 vecors with the hkl of each corresponding spot
 *  returns the number of spots created.
 */
long MakeLauePatternUnitVectors(
double  a,				/*Euler angles alpha, beta, gamma (radian) */
double  b,
double  g,
struct box_struct gRange,/* structure giving the allowed 3d values of (G^ .dot. x^), and also for y^ and z^ */
struct crystalStructure *xtal,
double  (**GhatSpots)[3],
int		(**hklSpots)[3],
int		hkl0[3],		/* preferred hkl at center of pattern */
double  cone,			/* acceptable cone angle from preferred hkl0 */
double  keVmax)			/* maximum energy (keV) to go out to */
{
	double  gmax, gmin;					/* maximum (and min) length of a G vector */
	long	hmax,kmax,lmax;				/* maximum possible values of h, k, and l */
	double  parallel;					/* cos(cone), used to test dot products */
	double  M_Euler[3][3];				/* an Euler matrix */
	double  recip[3][3];				/* rotated reciprocal space */
	double  glen;						/* length of a G vector */
	double  ghat0[3];					/* direction of hkl0[3], normalized */
	long	h,k,l;						/* miller indices */
	double	hkl[3];						/* vector form of (h,k,l) */
	double  ghat[3];					/* direction of G vector */
	long	n;							/* actual number of triplets in GhatSpots[][3] */
	long	N;							/* allocated number of triplest in GhatSpots[][3] */
	double  kihat[3]={0,0,1};			/* incident beam direction, used to print keV */
	double	keV;						/* energy of a reflection (keV) */

#if (DEBUG>2)
	fprintf(fout,"\nmaking Gtest with angles=(%8.3f, %8.3f, %8.3f) (deg)  in MakeLauePatternUnitVectors()\n",a*180/M_PI,b*180/M_PI,g*180/M_PI);
#endif
	if (*GhatSpots || *hklSpots) {
		fprintf(stderr,"GhatSpots and hklSpots must be NULL (unallocated space) on entry to MakeLauePatternUnitVectors()\n");
		return 0;
	}
	parallel = cos(cone);								/* dot(hkl,hkl0) must be greater or equal to parallel */
	gmin = 4*M_PI*sin(THETA_MIN)*KEV_MIN/hc;			/* minimum 2-theta is 45° for our detector, gmax = 4 PI*sin(theta)/lambda */
	gmax = 4*M_PI*sin(THETA_MAX)*keVmax/hc;				/* maximum 2-theta is 135° for our detector, gmax = 4 PI*sin(theta)/lambda */
	hmax = ceil(gmax/MAG3(xtal->recip[0][0],xtal->recip[1][0],xtal->recip[2][0])); /* max range to check */
	kmax = ceil(gmax/MAG3(xtal->recip[0][1],xtal->recip[1][1],xtal->recip[2][1]));
	lmax = ceil(gmax/MAG3(xtal->recip[0][2],xtal->recip[1][2],xtal->recip[2][2]));

	EulerMatrix(a,b,g,M_Euler);							/* make the rotation matrix M_Euler from Euler angles */
	MatrixCopy33(recip,xtal->recip);					/* copy recip from xtal into local version for rotation */
	MatrixMultiply33(M_Euler,recip,recip);				/* rotate recip here */

#ifdef __GNUC__
	#warning "note that ghat0 is in the rotated frame, not un-rotated as was done previously"
#endif
	hkl[0]=hkl0[0];  hkl[1]=hkl0[1];  hkl[2]=hkl0[2];   /* make ghat0[3], direction vector given by hkl0[3] */
	MatrixMultiply31(recip,hkl,ghat0);
	normalize3(ghat0);

	N = (2*hmax+1)*(2*kmax+1)*(2*lmax+1);				/* number of entries to pre allocate */
	N = (N>2000) ? 2000 : N;
	*GhatSpots = calloc((size_t)N,3*sizeof(double)); 
	if (!(*GhatSpots)) { fprintf(stderr,"Unable to allocate space for GhatSpots in MakeLauePatternUnitVectors()\n"); return 0; }
	*hklSpots = calloc((size_t)N,3*sizeof(int)); 
	if (!(*hklSpots)) { fprintf(stderr,"Unable to allocate space for hklSpots in MakeLauePatternUnitVectors()\n"); return 0; }

	n = 0;
	for (l=-lmax;l<=lmax;l++) {
		hkl[2] = l;
		for (k=-kmax;k<=kmax;k++) {
			hkl[1] = k;
			for (h=-hmax;h<=hmax;h++) {
				if (!allowedHKL(xtal,h,k,l)) continue;
				hkl[0] = h;
				MatrixMultiply31(recip,hkl,ghat);
				glen = normalize3(ghat);
				if (glen<gmin || glen>gmax) continue;	/* g vector is too big (beyond lambda_min) */
				if (dot3(ghat0,ghat)<parallel) continue;/* angle between hkl and hkl0 is greater than 'cone' */
				if (gRange.ylo > ghat[1] || gRange.yhi < ghat[1]) continue;
				if (gRange.xlo > ghat[0] || gRange.xhi < ghat[0]) continue;
				if (gRange.zlo > ghat[2] || gRange.zhi < ghat[2]) continue;
/*				theta = asin(-dot3(kihat,ghat));			// Bragg angle */
/*				keV = hc*glen / (4*M_PI*sin(theta));		// G = 4*PI*sin(theta)/lambda, and G = 2*PI/d */
				keV = hc*glen / (-4*M_PI*dot3(kihat,ghat));	/* G = 4*PI*sin(theta)/lambda, and G = 2*PI/d */
				if (keV>keVmax) continue;
				if (n>=N) {
					N += 500;							/* need more space, reallocate */
					*GhatSpots = realloc(*GhatSpots,N*3*sizeof(double));
					*hklSpots  = realloc(*hklSpots,N*3*sizeof(int));
					if (!(*GhatSpots)  || !(*hklSpots)) { fprintf(stderr,"Unable to re-allocate space for GhatSpots[][3] or hklSpots[][3]\n"); return 0; }
				}
#if (DEBUG>3)
	if (n<55) fprintf(fout,"saving Gtest[%2ld] = (%2ld %2ld %2ld)  =  (%8.5f, %8.5f, %8.5f),  glen=%g,  keV=%g\n",n,h,k,l,ghat[0],ghat[1],ghat[2],glen,keV);
	if (n==56) fprintf(fout," there are more not shown....\n");
#endif
				VECTOR_COPY3((*GhatSpots)[n],ghat);
				(*hklSpots)[n][0] = h;      (*hklSpots)[n][1] = k;      (*hklSpots)[n][2] = l;
				n++;
			}
		}
	}
	N = n = RemoveDuplicateTripletsPlus(n,*GhatSpots,*hklSpots,sizeof((*hklSpots)[0]),0);
	*GhatSpots = realloc(*GhatSpots,N*3*sizeof(double)); /* trim arrays to final size */
	*hklSpots = realloc(*hklSpots,N*3*sizeof(int));
	return N;
}



#define JUST_OUTSIDE 1.00000001			/* used to set limit just outside the ±1 box */
/* returns the range of the G^ vectors that include all of the measured spots.  The range is in units of direction cosines */
/* the detector is assumed here to be definded by the distribution of data spots */
void SetGhatRangeFromDataSpots(
long	Ni,						/* number of measured spots */
double  (*GhatSpots)[3],		/* G^ of each data spot on the detector */
struct box_struct *range)		/* structure giving the allowed 3d values of (G^ .dot. x^), and also for y^ and z^ */
{
	long	i;
	double  xmin=JUST_OUTSIDE, xmax=-JUST_OUTSIDE;  /* init to include nothing */
	double  ymin=JUST_OUTSIDE, ymax=-JUST_OUTSIDE;  /* remember that these are cosines, not angles */
	double  zmin=JUST_OUTSIDE, zmax=-JUST_OUTSIDE;
	double  delta=0.05;								/* fraction to add to each edge, delta=0.05, expands each limit by 5% */
	double  expand;									/* extra space around measured spotsm, some wiggle room */

	for (i=0;i<Ni;i++) {
		xmin = MIN(xmin,GhatSpots[i][0]);
		xmax = MAX(xmax,GhatSpots[i][0]);
		ymin = MIN(ymin,GhatSpots[i][1]);
		ymax = MAX(ymax,GhatSpots[i][1]);
		zmin = MIN(zmin,GhatSpots[i][2]);
		zmax = MAX(zmax,GhatSpots[i][2]);
	}

/*  this code had problems when the max was negative.  Don't multiply, add and subtract instead
 *	double  delta=0.10;								// fraction to add to each edge, delta=0.05, expands each limit by 5%
 *	expand = 1+delta*(xmax-xmin)/2;					// expand both edges by ±(delta/100)% in x-range
 *	range->xlo = MAX(xmin*expand,-JUST_OUTSIDE);	// expand each limit a bit to give a little wiggle room
 *	range->xhi = MIN(xmax*expand,JUST_OUTSIDE);		// and trim so that we are not beyond the box of ±1
 *	expand = 1+delta*(ymax-ymin)/2;					// expand both edges by ±(delta/100)% in y-range
 *	range->ylo = MAX(ymin*expand,-JUST_OUTSIDE);	
 *	range->yhi = MIN(ymax*expand,JUST_OUTSIDE);
 *	expand = 1+delta*(zmax-zmin)/2;					// expand both edges by ±(delta/100)% in z-range
 *	range->zlo = MAX(zmin*expand,-JUST_OUTSIDE);
 *	range->zhi = MIN(zmax*expand,JUST_OUTSIDE);
 */

	expand = delta*(xmax-xmin);						/* expand both edges by ±(delta/100)% in x-range */
	range->xlo = MAX(xmin-expand,-JUST_OUTSIDE);	/* expand each limit a bit to give a little wiggle room */
	range->xhi = MIN(xmax+expand,JUST_OUTSIDE);		/* and trim so that we are not beyond the box of ±1 */
	expand = delta*(ymax-ymin);						/* expand both edges by ±(delta/100)% in y-range */
	range->ylo = MAX(ymin-expand,-JUST_OUTSIDE);	
	range->yhi = MIN(ymax+expand,JUST_OUTSIDE);
	expand = delta*(zmax-zmin);						/* expand both edges by ±(delta/100)% in z-range*/
	range->zlo = MAX(zmin-expand,-JUST_OUTSIDE);
	range->zhi = MIN(zmax+expand,JUST_OUTSIDE);
	return;
}



/* Intensity proportional to F^2 lambda^4 / sin(theta)^2
 *	this is proportional to F^2 * lambda^2 * d^2
 *	for F, use 1.2 * 5 exp(-3s^2), wheere s = 2π/d,  this is approx Al value from Cromer
 *	then scale the whole thing by xxx to put value in nice range.
 */
void reflectionProperties(
double	Ghat[3],				/* direction of G^ for this reflection, unchanged */
int		hklin[3],				/* hkl of reflection, returned as lowest allowed */
struct crystalStructure *xtal,	/* lattice parameters, etc. */
double  *keV,					/* energy of reflection */
double	*intens)				/* predicted intensity */
{
	double  kihat[3]={0,0,1};		/* incident beam direction, used to print keV */
	double  sinTheta;				/* sin(Bragg angle) of reflection, used to print keV */
	double	F;						/* approx structure factor (the atomic form factor is VERY approximate, the atom position part is exact) */
	double  s;						/* 2π/d */
	double  gvec[3];				/* g-vector from hkl */
	double  Fr,Fi;					/* real and imag parts of F */
	long	hkl[3];					/* long version of hkl[3] for passing */

	hkl[0] = hklin[0];  hkl[1] = hklin[1];  hkl[2] = hklin[2];
	lowestAllowedHKL(hkl,xtal);
	gvec[0]=hkl[0];  gvec[1]=hkl[1];  gvec[2]=hkl[2];
	MatrixMultiply31(xtal->recip,gvec,gvec);		/* gvec is now g-vector from hkl */
	s = NORM3(gvec);								/* s = 2*PI/d */
	sinTheta = -dot3(kihat,Ghat);					/* sin(Bragg angle) */
	*keV = hc*s / (4*M_PI*sinTheta);				/* G = 4*PI*sin(theta)/lambda, and G = 2*PI/d */
	F = 1.2 + 5*exp(-3*s*s);						/* apporx. generic atomic form factor */
	Fstruct(xtal,hkl[0],hkl[1],hkl[2],&Fr,&Fi);
	F *= sqrt((Fr*Fr)+(Fi*Fi));						/* |(Fr,Fi)| */
	*intens = F*2*M_PI/s/(*keV);					/* proportional to F^2 * lambda^2 * d^2 */
	*intens = ((*intens) * (*intens));
}



long PeakInHough(
struct WaveSpace_struct *HoughSpace,/* a 3-d Hough transform */
double  angles[3],					/* Euler angles returned in this 3-vector (radian),  {alpha, beta, gamma} */
int		type)						/* type of peak finding routine */
									/* type=1 for peak,  2 for Center of Mass, 3 for COM around peak, 4 COM of peakish part */
{
	long	i,j,k;
	long	peakValue;
	long	***hough;
#if (DEBUG)
	double  seconds;				/* used to print execution time */
	clock_t time0;
	char	str[256];				/* string used to show time */
	time0=clock();
#endif

	if (!HoughSpace) {
		fprintf(stderr,"the 3-d Hough Transformed space does not exist in PeakInHough()\n");
		angles[0] = angles[1] = angles[2] = NAN;
		return 1;
	}
	if (!(type>=1 && type <=4)) {
		fprintf(stderr,"type must be 1, 2, 3, or 4 in PeakInHough()\n");
		return 1;
	}
	hough = HoughSpace->space;							/* convienient shortcut */

	if (type==1) {										/* Maximum */
		peakValue = FindMaxIn3d(HoughSpace->Na,HoughSpace->Nb,HoughSpace->Ng,hough,&i,&j,&k);
		angles[0] = (HoughSpace->aoff + i*HoughSpace->da);
		angles[1] = (HoughSpace->boff + j*HoughSpace->db);
		angles[2] = (HoughSpace->goff + k*HoughSpace->dg);
#if (DEBUG>2)
		fprintf(fout,"max at (%.2f)(%.2f)(%.2f)  or   [%ld][%ld][%ld] = %ld\n",angles[0]*180/M_PI,angles[1]*180/M_PI,angles[2]*180/M_PI,i,j,k,hough[i][j][k]);
#endif
	}

	else if (type==2) {									/* Center of Mass */
		vectorCOM(HoughSpace,angles,-1.);				/* returns angles (radians) */
		i = round( (angles[0] - HoughSpace->aoff) / HoughSpace->da );
		j = round( (angles[1] - HoughSpace->boff) / HoughSpace->db );
		k = round( (angles[2] - HoughSpace->goff) / HoughSpace->dg );
		peakValue = FindMaxIn3d(HoughSpace->Na,HoughSpace->Nb,HoughSpace->Ng,hough,NULL,NULL,NULL);
#if (DEBUG>2)
		fprintf(fout,"Center of Mass in Hough transform is at (%.2f)(%.2f)(%.2f)  or   [%ld][%ld][%ld]\n",angles[0]*180/M_PI,angles[1]*180/M_PI,angles[2]*180/M_PI,i,j,k);
#endif
	}

	else if (type==3) {									/* Center of Mass around the Maximum */
		long	hw=5;									/* half-width of region to get COM */
		hw = floor( pow((double)(HoughSpace->Na * HoughSpace->Nb * HoughSpace->Ng),1./.3)/10. );
		peakValue = FindMaxIn3d(HoughSpace->Na,HoughSpace->Nb,HoughSpace->Ng,hough,&i,&j,&k);
		angles[0]=i; angles[1]=j; angles[2]=k;			/* central indicies passed to vectorCOM() for limited range COM */
		vectorCOM(HoughSpace,angles,(double)hw);
		i = round( (angles[0] - HoughSpace->aoff) / HoughSpace->da );
		j = round( (angles[1] - HoughSpace->boff) / HoughSpace->db );
		k = round( (angles[2] - HoughSpace->goff) / HoughSpace->dg );
#if (DEBUG>2)
		fprintf(fout,"Center of Mas over ±%ld around peak in Hough transform is at (%.2f)(%.2f)(%.2f)  or   [%ld][%ld][%ld]\n",hw,angles[0]*180/M_PI,angles[1]*180/M_PI,angles[2]*180/M_PI,i,j,k);
#endif
	}
	else if (type==4) {									/* check that peak is peakish, then take COM around the max */
		long	hw;										/* half-width of region to get COM */
		long	hwMax;
		long	face;									/* value on the cube face at dist hw from center */
		long	lastface;

		hwMax = MIN(HoughSpace->Na,HoughSpace->Nb);
		hwMax = MIN(HoughSpace->Ng,hwMax)/3.;
		/* hwMax = floor( pow(HoughSpace->Na * HoughSpace->Nb * HoughSpace->Ng,1./.3)/10. ); */
		peakValue = FindMaxIn3d(HoughSpace->Na,HoughSpace->Nb,HoughSpace->Ng,hough,&i,&j,&k);

		/* now start marching out from the peak to a max distance of hwMax looking for non-monotonic behavior */
		/* stop when hw increases to hwMax, or face reaches 1/10 of peakValue */
		face = peakValue;
		hw = 1;											/* this starts the checking at hw==3 */
		do {
			hw++;
			lastface = face;
			face = maxOnCubeFace(HoughSpace->Na,HoughSpace->Nb,HoughSpace->Ng,hough,i,j,k,hw+1);
		} while((hw+2)<hwMax && face<=lastface && face>=(peakValue/10));

		angles[0]=i; angles[1]=j; angles[2]=k;			/* central indicies passed to vectorCOM() for limited range COM */
		vectorCOM(HoughSpace,angles,(double)hw);		/* now get the COM using a distance hw just determined */
		i = round( (angles[0] - HoughSpace->aoff) / HoughSpace->da );
		j = round( (angles[1] - HoughSpace->boff) / HoughSpace->db );
		k = round( (angles[2] - HoughSpace->goff) / HoughSpace->dg );
#if (DEBUG>2)
		fprintf(fout,"Center of Mass arond peakish part of peak in Hough transform is at (%.2f)(%.2f)(%.2f)  or   [%ld][%ld][%ld]\n",angles[0]*180/M_PI,angles[1]*180/M_PI,angles[2]*180/M_PI,i,j,k);
#endif
	}
	else {
		fprintf(stderr,"Illegal type in PeakInHough()\n");
		angles[0] = angles[1] = angles[2] = NAN;
		return 0;
	}

/*	MakeCalculatedData(angles[0],angles[1],angles[2]); */
#if (DEBUG)
	seconds = ((double)(clock()-time0))/CLOCKS_PER_SEC;
	if (seconds > 0.5) fprintf(fout,"filling Peak in Hough Space takes %s  =  (%.2f sec)\n",num2sexigesmal(str,seconds,0),seconds);
#endif
	return peakValue;
}



/* for a 3-d matrix wav, return the center of mass in the vector com */
void vectorCOM(
struct WaveSpace_struct *wav3d, /* 3-d input wave */
double  com[3],					/* set to center of mass */
double  hw)						/* compute COM over ±hw around the value of com on entry, (-1 means do all) */
{
	long	i1=0,j1=0,k1=0;		/* range of ijk to use for computing COM, i in [i1,i2] */
	long	i2,j2,k2;
	double  wt, sumz;
	long	i,j,k;

	i2 = wav3d->Na - 1;
	j2 = wav3d->Nb - 1;
	k2 = wav3d->Ng - 1;

	if (hw>=0) {									/* do only part of the array */
		hw = round(hw);
		i1 = MAX(round(com[0]-hw),0);
		i2 = MIN(round(com[0]+hw),i2);
		j1 = MAX(round(com[1]-hw),0);
		j2 = MIN(round(com[1]+hw),j2);
		k1 = MAX(round(com[2]-hw),0);
		k2 = MIN(round(com[2]+hw),k2);
	}

	com[0] = com[1] = com[2] = 0;
	sumz = 0;
	for (k=k1;k<=k2;k++) {
		for (j=j1;j<=j2;j++) {
			for (i=i1;i<=i2;i++) {
				wt = wav3d->space[i][j][k];
				com[0] += i*wt;
				com[1] += j*wt;
				com[2] += k*wt;
				sumz += wt;
			}
		}
	}
	com[0] = wav3d->aoff + com[0]/sumz*(wav3d->da);	/* convert from point to scaled xyz */
	com[1] = wav3d->boff + com[1]/sumz*(wav3d->db);
	com[2] = wav3d->goff + com[2]/sumz*(wav3d->dg);
}



/* returns the maximum value anywhere on the faces of a cube */
long maxOnCubeFace(
long	Na,				/* dimensions of hough */
long	Nb,
long	Ng,
long	***hough,		/* 3-d space to check */
long	ic,				/* center of cube check about */
long	jc,
long	kc,
long	hw)				/* distance from cube center to edge */
{
	long	i1,i2,j1,j2,k1,k2;		/* indicies ouf bounding cube */
	long	max=0;					/* max value found */
	long	i,j,k;

	i1 = MAX(0,ic-hw);
	i2 = MIN(Na-1,ic+hw);
	j1 = MAX(0,jc-hw);
	j2 = MIN(Nb-1,jc+hw);
	k1 = MAX(0,kc-hw);
	k2 = MIN(Ng-1,kc+hw);

	/* now check the 6 faces */

	/* face 1 & 2, k=k1 or k2, i=[i1,i2], j=[j1,j2] */
	k = k2;
	for (i=i1;i<=i2;i++) {
		for (j=j1;j<=j2;j++) {
			max = MAX(hough[i][j][k1],max);
			max = MAX(hough[i][j][k2],max);
		}
	}

	/* face 3 & 4, j=j1 or j2, i=[i1,i2], k=[k1,k2] */
	k = k2;
	for (i=i1;i<=i2;i++) {
		for (k=k1;k<=j2;k++) {
			max = MAX(hough[i][j1][k],max);
			max = MAX(hough[i][j2][k],max);
		}
	}

	/* face 5 & 6, i=i1 or i2, i=[i1,i2], k=[k1,k2] */
	k = k2;
	for (j=j1;j<=j2;j++) {
		for (k=k1;k<=j2;k++) {
			max = MAX(hough[i1][j][k],max);
			max = MAX(hough[i2][j][k],max);
		}
	}
return max;
}



long FindMaxIn3d(
long	n0,				/* dimension along the 0 direction */
long	n1,
long	n2,
long	***a,			/* the array to search */
long	*imax,			/* location of the max */
long	*jmax,
long	*kmax)
{
	long	i,j,k;
	long	peak;		/* value at the peak */
	long	im,jm,km;   /* local value of imax, jmax, kmax */

	if (!a) { fprintf(stderr,"the 3-d array does not exist in FindMaxIn3d()\n"); return 0; }
	peak = a[0][0][0];
	im = jm = km =0;
	for (i=0;i<n0;i++) {
		for (j=0;j<n1;j++) {
			for (k=0;k<n2;k++) {
				if (peak<a[i][j][k]) {		/* save new peak point */
					peak = a[i][j][k];
					im=i; jm=j; km=k;
				}
			}
		}
	}
	if (imax) *imax = im;
	if (jmax) *jmax = jm;
	if (kmax) *kmax = km;
	return peak;
}



/* return the dot product between the two unit vectors that have the biggest angle between them
 * so the biggest angle (180 deg) will return a -1 */
double dotOfBiggestAngle(
long	Ni,						/* number of vectors in hat[][3] */
double	hat[][3])				/* array of unit vectors */
{
	long	i,j;
	double  minDot=1;			/* smallest dot product, init to largest possible vale */
	double  dot;

	for (j=0;j<(Ni-1);j++) {
		for (i=j+1;i<Ni;i++) {
			dot = dot3(hat[j],hat[i]);
			minDot = MIN(minDot,dot);
		}
	}
	return minDot;
}


/* returns string with highest order reachable reflection at given theta and keV */
char *HighestReachableHKL(
double  keV,
double  theta,
struct crystalStructure *xtal,				/* lattice parameters, etc. */
char	str[256])
{
	long	h,k,l;							/* indicies to loop over */
	long	hkl2;							/* h*h + k*k + l*l */
	long	hklmax;							/* highest possible index to check */
	long	maxhkl2;						/* |hkl|^2 at max G */
	long	max2,hmax,kmax,lmax;			/* values of current max in search */
	double  a_d;							/* a/d */

	a_d = (xtal->a) / ( (hc/keV)/2/sin(theta) );
	maxhkl2 = floor(a_d*a_d);
	hklmax=ceil(a_d);

	max2 = hmax = kmax = lmax = 0;
	for (h=0;h<=hklmax;h++) {
		for (k=h;k<=hklmax;k++) {
			for (l=k;l<=hklmax;l++) {
				if (!allowedHKL(xtal,h,k,l)) continue;
				hkl2 = h*h + k*k + l*l;
				if (hkl2 > max2 && hkl2<=maxhkl2) {			/* save values if this is a new maximum */
					max2 = hkl2;
					hmax = h;
					kmax = k;
					lmax = l;
				}
			}
		}
	}
	sprintf(str,"(%ld %ld %ld)",hmax,kmax,lmax);
	return str;
}




/******************************************************************************************
 * for a lattice structure, set the array recip with the reciprocal lattice based on the 
 * six constants a,b,c, alpha, beta, gamma.
 * calculation comes from:
 * International Tables B, pg 360, sect. 3.3.1 (columns of vector M)
 *  http://journals.iucr.org/iucr-top/comm/cteach/pamphlets/4/node8.html#SECTION00033000000000000000
 ******************************************************************************************/
//	int FillRecipInLattice(
//	struct latticeParameters *lattice)		/* 6 lattice parameters, not changed, just read, recip is set */
//	{
//		double  a1[3], a2[3], a3[3];		/* direct lattice vectors */
//		double  b1[3], b2[3], b3[3];		/* reciprocal lattice vectors */
//		double  phi;						/* similar to Vc,  phi = Vc/(a*b*c) */
//		double  Vc;							/* volume of unit cell */
//		double  a,b,c;						/* local versions of lattice constants */
//		double  sa;							/* sin(alpha) */
//		double  ca,cb,cg;					/* cos(alpha), cos(beta), cos(gamma) */
//		double  factor;						/* 2*PI / [vector triple product a1 .dot. (a2 x a3)] */
//	
//		a = lattice->a;
//		b = lattice->b;
//		c = lattice->c;
//		sa = sin(lattice->alpha);
//		ca = cos(lattice->alpha);
//		cb = cos(lattice->beta);
//		cg = cos(lattice->gamma);
//	
//		phi = sqrt(1.0 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg);
//		Vc = a*b*c * phi;
//	
//		/* set direct lattice vectors a1, a2, a3 */
//		a1[0] = a * phi / sa;				a2[0] = 0;				a3[0] = 0;
//		a1[1] = a * (cg - ca*cb) / sa;		a2[1] = b * sa;			a3[1] = 0;
//		a1[2] = a * cb;						a2[2] = b * ca;			a3[2] = c;
//	
//		/* compute reciprocal lattice vectors (with the 2PI) */
//		factor = (M_PI+M_PI) / Vc;
//		cross(a2,a3,b1); vector3cons(b1,factor);				/* b1 = (a2 x a3) * 2*PI/Vcell */
//		cross(a3,a1,b2); vector3cons(b2,factor);
//		cross(a1,a2,b3); vector3cons(b3,factor);
//	
//		/* fill the reciprocal lattice matrix from vectors */
//		lattice->recip[0][0]=b1[0]  ;  lattice->recip[1][0]=b1[1]  ;  lattice->recip[2][0]=b1[2];   /* astar[] */
//		lattice->recip[0][1]=b2[0]  ;  lattice->recip[1][1]=b2[1]  ;  lattice->recip[2][1]=b2[2];   /* bstar[] */
//		lattice->recip[0][2]=b3[0]  ;  lattice->recip[1][2]=b3[1]  ;  lattice->recip[2][2]=b3[2];   /* cstar[] */
//	
//	#if (DEBUG>4)   /******************************************************************************/
//	fprintf(fout,"lattice parameters are %g, %g, %g,   %g, %g, %g\n",a,b,c,lattice->alpha,lattice->beta,lattice->gamma);
//	fprintf(fout,"a* = (%+8.5f, %+8.5f, %+8.5f)\n",lattice->recip[0][0],lattice->recip[1][0],lattice->recip[2][0]);
//	fprintf(fout,"b* = (%+8.5f, %+8.5f, %+8.5f)\n",lattice->recip[0][1],lattice->recip[1][1],lattice->recip[2][1]);
//	fprintf(fout,"c* = (%+8.5f, %+8.5f, %+8.5f)\n",lattice->recip[0][2],lattice->recip[1][2],lattice->recip[2][2]);
//	#endif			/******************************************************************************/
//		return 0;
//	}


