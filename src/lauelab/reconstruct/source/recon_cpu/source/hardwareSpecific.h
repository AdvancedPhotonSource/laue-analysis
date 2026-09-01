/*
 *  hardwareSpecific.h
 *  reconstruct
 *
 *  Created by Jon Tischler on 1/19/09.
 *  Copyright 2009 ORNL. All rights reserved.
 *
 */


#define USE_POSITIONER_CORRECTION
#define USE_PM500_CORRECTION_MAY06
/* #define USE_DISTORTION_CORRECTION */

extern int	positionerType;		/*	0 means no correction (the default)
							1 means PM500 correction from May 2006
							2 means Alio correction from Oct 2009 */

#define EPOCH 1970		/* using an epoch of Jan 1, 1970, 00:00:00, 1970 is the UNIX epoch */


#ifdef USE_DISTORTION_CORRECTION
#define PEAKCORRECTION(A) peakcorrection(A)
#else
#define PEAKCORRECTION(A) (A)
#endif

int positionerTypeFromFileTime(char *fileTime);

#ifdef USE_POSITIONER_CORRECTION
double X2corrected(double X2);
double Y2corrected(double Y2);
double Z2corrected(double Z2);
double X2correctedPM500(double X2);
double Y2correctedPM500(double Y2);
double Z2correctedPM500(double Z2);
#else
#define X2corrected(A) (A)
#define Y2corrected(A) (A)
#define Z2corrected(A) (A)
#endif



point_ccd peakcorrection(point_ccd pixel);

void load_peak_correction_maps(char* filename);
