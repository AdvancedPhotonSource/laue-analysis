/**********************************************************

	Data Structure and Routines for a calibration params
       

/**********************************************************/
#include "calibparam.h"


Calibparam * default_calibparam(void){
  Calibparam *cp=malloc(sizeof(Calibparam));
  cp->xdim=1042;
  cp->ydim=1042;
  cp->emin=5.;
  cp->emax=20.;
  cp->dpsx=27.914;
  cp->dpsy=25.9;
  cp->dd=29.3;
  cp->xcent=493.;
  cp->ycent=512.;
  return cp;
}
void delete_calibparam(Calibparam *cp){
  free(cp);
}

