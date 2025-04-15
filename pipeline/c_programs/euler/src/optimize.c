#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _MSC_VER				/* identifies this as a Microsoft compiler */
#define _USE_MATH_DEFINES	/* added RX2011 */
#endif
#include <math.h>
#include <time.h>
#include <limits.h>
#include <gsl/gsl_multimin.h>
//#include "/Users/tischler/dev/GNU/include/gsl/gsl_multimin.h"
#include "Euler.h"

#ifdef DEBUG_ON
#define DEBUG 1				/* a value of 5 gets everything */
#endif

#define MIN(A,B)	( (A)<(B)?(A):(B) )
#define MAX(A,B)	( (A)>(B)?(A):(B) )

double my_f (const gsl_vector *v, void *params);




/******************************************************************************************
 * return the error between a lattice rotated by Euler angles and the measured spots
 * thus the returned value is a minimum for the optimimal Euler angles
 */
double my_f(
const gsl_vector *v,	/* vector with current value of Euler angles */
void	*pattern)		/* pointer to the parameters, in this case a pointer to the pattern structure */
{
	double  G[3];						/* calculated G vector for each hkl */
	double  recip[3][3];				/* reciprocal space */
	double  M_Euler[3][3];				/* an Euler matrix */
	size_t  N;							/* local copy of number of measured spots to check */
	double a,b,g;						/* local copy of the Euler angles */
	struct patternOfOneGrain *p;		/* local value of pattern */
	double  rms;						/* rms value of error, the returned value */
	size_t  i;							/* loop index */
	double  hkl[3];						/* float value of hkl */
	double  dot;

	/* p = pattern; */
	p = (struct patternOfOneGrain *)pattern;
	N = p->Ni;											/* local value for convenience */
	a = gsl_vector_get(v,0);							/* the Euler angles */
	b = gsl_vector_get(v,1);
	g = gsl_vector_get(v,2);
	MatrixCopy33(recip,p->xtal.recip);					/* copy unrotated reciprocal lattice into recip */

	EulerMatrix(a,b,g,M_Euler);							/* make the rotation matrix M_Euler from Euler angles */
	MatrixMultiply33(M_Euler,recip,recip);				/* rotate recip by Euler angles */
	for (i=0,rms=0.0;i<N;i++) {
		VECTOR_COPY3(hkl,p->hkls[i]);					/* convert integer hkl to float for following MatrixMultiply31 */
if (hkl[0]!=hkl[0]) {
fprintf(stderr,"hkl[0] is NaN\n");
}
		MatrixMultiply31(recip,hkl,G);
if (G[0]!=G[0]) {
fprintf(stderr,"G[0] is NaN\n");
}
		normalize3(G);									/* normalized gvec of hkl[i] rotated by Euler angles */
		dot = dot3(G,p->Ghat[i]);
		dot = MIN(dot,1.0);
		rms += 1. - dot;								/* (1-dot) ~ 0.5*(delta angle)^2*/
	}
	rms = sqrt(rms/(double)N);							/* rms of angle errors */
	return rms;
}





/* When calling optimizeEulerAngles(), be sure to set pattern.xtal:

	e.g.
	pattern.xtal.a = pattern.xtal.b = pattern.xtal.c = 4.05;		// just default to Al for now
	pattern.xtal.alpha = pattern.xtal.beta = pattern.xtal.gamma = M_PI/2.0;

	then call:
		FillRecipInLattice(&(pattern.lattice));
	to set the reciprocal lattice in the lattice structure
*/
/* this routine optimizes approximate Euler angles using the gsl simplex method. */
int optimizeEulerAngles(
double  startStep,					/* starting step size (radians), make this a bit larger than the error */
double  epsAbs,						/* tolerance on the answer (=0.001) */
long	maxIter,					/* maximum number of iterations, stop after this many regardless, perhaps 100 */
struct	patternOfOneGrain *pattern) /* provides G^'s and hkl's for one fitted pattern, and the lattice parameters */
{
	size_t np = 3;					/* number of dimensions of the function, 3 Euler angles */

	/* T tells what kind of minimizer we are using, select the simplex method */
	const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex;
	gsl_multimin_fminimizer *s = NULL;		/* this is the minimizer */
	gsl_vector *ss, *x;						/* pointers to vectors,  ss is stepsize, x is point */
	gsl_multimin_function minex_func;		/* struct defining parts needed for minimizing */
	size_t iter=0, i;
	int status;								/* status of returned gsl function, 0 is OK */
	double size;							/* size of minimizer, sort of maximum error */

	double  G[3];							/* calculated G vector for each hkl */
	double  recip[3][3];					/* reciprocal space */
	double  M_Euler[3][3];					/* an Euler matrix */
	double  hkl[3];							/* float value of hkl */
	double  dot;

	if (!pattern) { fprintf(stderr,"pattern in NULL on entry to optimizeEulerAngles()\n"); return 1; }
	if (pattern->Ni<3) { fprintf(stderr,"less than 3 pairs of Ghat & hkls on entry to optimizeEulerAngles()\n"); return 1; }
	if (!(pattern->Ghat) || !(pattern->hkls)) { fprintf(stderr,"Ghat or hkls are NULL on entry to optimizeEulerAngles()\n"); return 1; }
	if ((pattern->xtal.a)<=0 || (pattern->xtal.b)<=0 || (pattern->xtal.c)<=0) { fprintf(stderr,"invalid lattice in optimizeEulerAngles()\n"); return 1; }

	/* Initial vertex step size vector */
	ss = gsl_vector_alloc(np);				/* step sizes in x and y */

	/* Set all step sizes to startStep, the starting step size */
	gsl_vector_set_all(ss, startStep);		/* set ss[0] = ss[1]  = ss[2] = startStep (radian) */

	/* Starting point */
	x = gsl_vector_alloc(np);				/* allocate space for a vector of length 3 */
	gsl_vector_set(x, 0, pattern->alpha);   /* set to starting guess, input (alpha,beta,gamma) */
	gsl_vector_set(x, 1, pattern->beta);
	gsl_vector_set(x, 2, pattern->gamma);

	/* Initialize the minimizer */
	minex_func.f = &my_f;					/* the function that returns the value */
	minex_func.n = np;						/* number of parameters */
	minex_func.params = (void *)pattern;	/* a generic pointer that points to something with info for my_f() */
	/* Note: my_func.df, and my_func.fdf are not needed since simplex method does not use derivatives */

	s = gsl_multimin_fminimizer_alloc(T, np);	/* allocate for a 2d simplex minimizer named 's' */

	/* Set starting point for the minimizer:
		for the minimizer 's',
		working on a function 'minex_func'
		starting from the initial guess x[2]
		using step sizes of ss[2]  */
	gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

#if (DEBUG)		/******************************************************************************/
	fprintf(fout, "start minimization with Euler angles (%15.10f, %15.10f, %15.10f) (deg.)\n", \
		pattern->alpha*180/M_PI,pattern->beta*180/M_PI,pattern->gamma*180/M_PI);
#endif			/******************************************************************************/

	/* Iterate the minimizer.  This looop iterates the minimization method until it is good enough */
	iter = 0;
	do {
		iter++;
		status = gsl_multimin_fminimizer_iterate(s);	/* do one iteration of the minimizer */
		if (status) break;								/* status=0 is OK, check here for an error */

		size = gsl_multimin_fminimizer_size (s); 		/* get 'size' of minimizer */
		status = gsl_multimin_test_size (size, epsAbs);	/* test size against epsAbs */

#if (DEBUG>1)		/******************************************************************************/
		fprintf(fout,"%5ld  f(", iter);
		for (i=0; i<np; i++) {
			fprintf(fout,"%.8f", gsl_vector_get(s->x, i));
			if (i<np-1) fprintf(fout,", ");
		}
		fprintf(fout,") = %15.8g size = %.5g\n", s->fval, size);
#endif			/******************************************************************************/
	} while (status == GSL_CONTINUE && iter < (size_t)maxIter); /* stop after maxIter regardless of size */

#if (DEBUG)		/******************************************************************************/
	if (status == GSL_SUCCESS) fprintf(fout,"converged to minimum successfully\n");
	else fprintf(fout,"did not converge successfully, status = %d\n",status);
	fprintf(fout, "  optimization changed Euler angles by (%15.10f, %15.10f, %15.10f) (deg.)\n", \
		(gsl_vector_get(s->x,0)-pattern->alpha)*180/M_PI, (gsl_vector_get(s->x,1)-pattern->beta)*180/M_PI, \
		(gsl_vector_get(s->x,2)-pattern->gamma)*180/M_PI);
#endif			/******************************************************************************/

	pattern->alpha = gsl_vector_get(s->x,0);			/* set to final values, resultant (alpha,beta,gamma) returned */
	pattern->beta = gsl_vector_get(s->x,1);
	pattern->gamma = gsl_vector_get(s->x,2);

#if (DEBUG)		/******************************************************************************/
	fprintf(fout, "after minimization, Euler angles are (%15.10f, %15.10f, %15.10f) (deg.)\n", \
		pattern->alpha*180/M_PI,pattern->beta*180/M_PI,pattern->gamma*180/M_PI);
#endif			/******************************************************************************/

	/* recalculate the err[] list in pattern */
	MatrixCopy33(recip,pattern->xtal.recip);		/* copy unrotated reciprocal lattice into recip */
	EulerMatrix(pattern->alpha,pattern->beta,pattern->gamma,M_Euler);/* rotation matrix from Euler angles */
	MatrixMultiply33(M_Euler,recip,recip);			/* rotate recip by Euler angles */
	for (i=0;i<(size_t)(pattern->Ni);i++) {
		VECTOR_COPY3(hkl,pattern->hkls[i]);			/* convert integer hkl to float for following MatrixMultiply31 */
		MatrixMultiply31(recip,hkl,G);
		normalize3(G);								/* normalized gvec of hkl[i] rotated by Euler angles */
		dot = dot3(G,pattern->Ghat[i]);				/* note, Ghat here is the measured, not calculated G */
		(pattern->err)[i] = acos(MIN(dot,1.0));
	}

	gsl_vector_free(x);									/* free vector of x,y positions */
	gsl_vector_free(ss);								/* free vector of step sizes */
	gsl_multimin_fminimizer_free (s);					/* free the minimizer */
	return status;
}







