/**********************************************************

	Data Structure and Routines for a valued point
	It contains an [x, y] location, and an intensity value.

/**********************************************************/

#include <stdlib.h>
#include <errno.h>

#ifndef _POINT_H_
#define _POINT_H_

typedef struct {

	double x;
	double y;
	double value;
	
} Point;



Point*	point_new_initialized(double x, double y, double value);
Point*	point_new(void);
Point*	point_copy(Point* p);
void	point_delete(Point* p);

#endif
