/**********************************************************

	Data Structure and Routines for a calibration params
       

/**********************************************************/
#include <stdlib.h>

#ifndef _CALIBPARAM_H_
#define _CALIBPARAM_H_

typedef struct {
  int xdim;
  int ydim;
  float emin;
  float emax;
  float dpsx;
  float dpsy;
  float dd;
  float xcent;
  float ycent;
  //more ..
  
} Calibparam;

Calibparam * default_calibparam(void);
void delete_calibparam(Calibparam *cp);


#endif
