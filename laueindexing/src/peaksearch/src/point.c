/**********************************************************

	Data Structure and Routines for a valued point
	It contains an [x, y] location, and an intensity value.

/**********************************************************/

#include "point.h"


Point* point_new_initialized(double x, double y, double value){
	
	Point* p = malloc(sizeof(Point));
	if (!p) exit(ENOMEM);		/* Not enough space. */
	p->x = x;
	p->y = y;
	p->value = value;
	
	return p;
	
}

Point* point_new(void){

	Point* p = malloc(sizeof(Point));
	if (!p) exit(ENOMEM);		/* Not enough space. */
	p->x = 0;
	p->y = 0;
	p->value = 0;

	return p;
	
}

Point* point_copy(Point* p){

	Point* p2 = point_new_initialized(p->x, p->y, p->value);
	return p2;
}

void point_delete(Point* p){

	free(p);

}
