/*
 *  hardwareSpecific.h
 *  reconstruct
 *
 *  Created by Jon Tischler on 1/19/09.
 *  Copyright 2009 ORNL. All rights reserved.
 *
 */


#define USE_PM500_CORRECTION_MAY06
/* #define USE_DISTORTION_CORRECTION */

#ifdef USE_DISTORTION_CORRECTION
#define PEAKCORRECTION(A) peakcorrection(A)
#else
#define PEAKCORRECTION(A) (A)
#endif


#ifdef USE_PM500_CORRECTION_MAY06
#define X2CORRECTED(A) X2corrected(A)
#define Y2CORRECTED(A) Y2corrected(A)
#define Z2CORRECTED(A) Z2corrected(A)
#else
#define X2CORRECTED(A) (A)
#define Y2CORRECTED(A) (A)
#define Z2CORRECTED(A) (A)
#endif


double X2corrected(double X2);
double Y2corrected(double Y2);
double Z2corrected(double Z2);
point_ccd peakcorrection(point_ccd pixel);

void load_peak_correction_maps(char* filename);
