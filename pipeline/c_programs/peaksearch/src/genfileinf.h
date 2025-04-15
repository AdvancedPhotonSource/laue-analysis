/**********************************************************

	Data Structure and Routines for a genfileinf

/**********************************************************/
#include <stdlib.h>
#include <stdio.h>

#ifndef _GENFILEINF_H_
#define _GENFILEINF_H_

#ifndef MAX_FILE_LENGTH
#define MAX_FILE_LENGTH 2047
#endif

typedef struct {		/* curvefit attributes */
	int boxsize;
	char CCDFilename[MAX_FILE_LENGTH+1];
	float widthx; 
	float widthy;
	float tilt;

	float xoff;
	float yoff;	

	int peakShape;		/* 0=Lorentzian, 1=Gaussian, that's all so far */

	//for checking the fitted peak
	float minwidth;
	float maxwidth;
	float maxCentToFit;
	float maxRfactor;
} Genfileinf;

Genfileinf * default_genfileinf(void);//for setting default values
void delete_genfileinf(Genfileinf *ginf);
int print_genfileinf(Genfileinf *ginf);

#endif
