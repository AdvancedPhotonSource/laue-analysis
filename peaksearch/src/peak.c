/**********************************************************

	Data Structure and Routines for a Peak


/**********************************************************/

#include "peak.h"


Peak* peak_new_initialized(
  double x,
  double y,
  double intens,
  double fitX,
  double fitY,
  double fitIntens,
  double fitBackground,
  double fitPeakWidthX,
  double fitPeakWidthY,
  double fitTilt,
  double integr,
  int boxsize,
  double chisq){
	
	Peak* p = malloc(sizeof(Peak));
	if (!p) exit(ENOMEM);		/* Not enough space. */
	p->x = x;
	p->y = y;
	p->intens=intens;
	p->fitX=fitX;
	p->fitY=fitY;
	p->fitIntens=fitIntens;
	p->fitBackground=fitBackground;
	p->fitPeakWidthX=fitPeakWidthX;
	p->fitPeakWidthY=fitPeakWidthY;
	p->fitTilt=fitTilt;
	p->integrIntens=integr;
	p->boxsize=boxsize;
	p->chisq=chisq;
	return p; 
	
}

Peak* peak_new(void){

	Peak* p = malloc(sizeof(Peak));
	if (!p) exit(ENOMEM);		/* Not enough space. */
	p->x = 0.;
	p->y = 0.;
        p->intens=0.;
	
	p->fitX=0.;
	p->fitY=0.;
	p->fitIntens=0.;
	p->fitBackground=0.;
	p->fitPeakWidthX=0.;
	p->fitPeakWidthY=0.;
	p->fitTilt=0.;
	p->integrIntens=0.;
	p->boxsize=0;
	p->chisq=0.;
	return p;
}

Peak* peak_copy(Peak* p){

	Peak* p2 = peak_new_initialized( p ->x,
					 p ->y,
					 p->intens,
	
					 p->fitX,
					 p->fitY,
					 p->fitIntens,
					 p->fitBackground,
					 p->fitPeakWidthX,
					 p->fitPeakWidthY,
					 p->fitTilt,
					 p->integrIntens,
					 p->boxsize,
					 p->chisq);
	return p2;
}

void peak_delete(Peak* p){
	if (p) free(p);
	p = NULL;
}
