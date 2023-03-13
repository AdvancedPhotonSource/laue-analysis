/**********************************************************

	Data Structure and Routines for a genfileinf

/**********************************************************/

#include <string.h>
#include "genfileinf.h"

//default seting, can be read from a file
Genfileinf * default_genfileinf(void){
	Genfileinf *ginf = malloc(sizeof(Genfileinf));
	ginf->tilt = 0.;
	ginf->widthx = 3.;
	ginf->widthy = 3.;
	ginf->boxsize = 10;
	ginf->xoff = 0.;
	ginf->yoff = 0.;
	ginf->minwidth = 0.01;
	ginf->maxwidth = 20.;
	ginf->maxCentToFit = 20.;
	ginf->maxRfactor = 0.9;
	ginf->peakShape = 0;	/* Lorentzian */
	sprintf(ginf->CCDFilename,"./CCD_distorMay03_corr.dat");
	return ginf;
}


void delete_genfileinf(Genfileinf *ginf){
	free(ginf);
}


int print_genfileinf(Genfileinf *ginf)
{
	char peakShape[1024];
	if (ginf==NULL) { fprintf(stderr,"invalid, ginf(==NULL)\n"); return 1; }

	if (ginf->peakShape == 0)		strcpy(peakShape,"'Lorentzian'");
	else if (ginf->peakShape == 1)	strcpy(peakShape,"'Gaussian'");
	else							sprintf(peakShape,"Unknown=%d\n",ginf->peakShape);

	printf("ginf->tilt = %g\n",ginf->tilt);
	printf("ginf->widthx = %g\n",ginf->widthx);
	printf("ginf->widthy = %g\n",ginf->widthy);
	printf("ginf->boxsize = %d\n",ginf->boxsize);
	printf("ginf->xoff = %g\n",ginf->xoff);
	printf("ginf->yoff = %g\n",ginf->yoff);
	printf("ginf->minwidth = %g\n",ginf->minwidth);
	printf("ginf->maxwidth = %g\n",ginf->maxwidth);
	printf("ginf->maxCentToFit = %g\n",ginf->maxCentToFit);
	printf("ginf->maxRfactor = %g\n",ginf->maxRfactor);
	printf("ginf->peakShape = %s\n",peakShape);
	printf("ginf->CCDFilename = '%s'\n",ginf->CCDFilename);
	return 0;
}

