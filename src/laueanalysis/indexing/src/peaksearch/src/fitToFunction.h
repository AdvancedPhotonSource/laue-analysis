/**********************************************************

	Data Structure and Routines for peak fiting
       

/**********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>

#include <errno.h>

#include "grid.h"
#include "grid_operations.h"
#include "minmax.h"
#include "point.h"

#ifndef _FITTOFUNCTION_H_
#define _FITTOFUNCTION_H_

typedef  struct {
  size_t n;
  double * y;
} ObservedValues ;

void fitToFunctionLorentz(Grid *image, double *fitx, double *fity, 
		   double *background, double *intens,
		   double *widthx, double *widthy, double *tilt,double *chisq);

void fitToFunctionGauss(Grid *image, double *fitx, double *fity, 
		   double *background, double *intens,
		   double *widthx, double *widthy, double *tilt,double *chisq);


#endif
