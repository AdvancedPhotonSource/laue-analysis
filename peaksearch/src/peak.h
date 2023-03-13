/**********************************************************

	Data Structure and Routines for a fitted peak
       

/**********************************************************/

#include <stdlib.h>
#include <errno.h>

#ifndef _PEAK_H_
#define _PEAK_H_

typedef struct {

  double x;
  double y;
  double intens;

  double fitX;
  double fitY;
  double fitIntens;
  double fitBackground;
  double fitPeakWidthX; //the init widthx widthy used for fitting are stored in Genfileinf 
  double fitPeakWidthY;
  double fitTilt;
  double integrIntens;
  int boxsize;

  /*
    following not included right now
  int fittype; //right now only one type implemented
  double pearsonindice;
  */
  double chisq;
  
	
} Peak;

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
  double integrIntens,
  int boxsize,
  double chisq);

Peak*	peak_new(void);
Peak*	peak_copy(Peak* p);
void	peak_delete(Peak* p);

#endif
