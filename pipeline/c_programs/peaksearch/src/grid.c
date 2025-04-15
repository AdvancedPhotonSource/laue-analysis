/**********************************************************

	Data Structure and Routines for working with
	a grid of data. This is used (in this case) for storing
	the contents of an SPE image file

/**********************************************************/

#include "grid.h"

#include <assert.h>

/*
 *	Changed Aug 2009 by Jon Tischler, added the mask parts and also made routines ignore pixels that were NAN
 */

/* *********************************************************************************************** */
/* *********************************** All Grids, Grid & GridB *********************************** */

void grid_delete(Grid* g){
	free(g->values);
	free(g);
	return;
}


/* *********************************************************************************************** */
/* ************************************* double Grids, Grid ************************************** */

Grid* grid_new(int width, int height) {
	/*create the struct*/
	Grid* g = malloc(sizeof(Grid));
	if (!g) exit(ENOMEM);				/* Not enough space. */
	g->height = height;
	g->width = width;
	g->values = malloc(sizeof(double) * height * width);
	if (!g->values) exit(ENOMEM);		/* Not enough space. */
	return g;
}


Grid* grid_new_copy(Grid* g) {			/* create a new duplicate of g */
	Grid* g2 = grid_new(g->width, g->height);
	grid_copy(g2, g);
	return g2;	
}


Grid* grid_new_copy_region(Grid* source, int x1, int y1, int x2, int y2){
	/*create a new grid just large enough to hold the indicated region*/
	/*x2/y2 inclusive*/
	int new_height = y2-y1+1;
	int new_width = x2-x1+1;
	Grid* destination = grid_new(new_width, new_height);
	grid_copy_region(destination, source, x1, y1, x2, y2);
	return destination;
}


void grid_copy_region(Grid* destination, Grid* source, int x1, int y1, int x2, int y2){
	int new_height = y2-y1+1;
	int new_width = x2-x1+1;
	int x, y;
	for (y = 0; y < new_height; y++){
		for (x = 0; x < new_width; x++){
			grid_set_value(destination, x, y, grid_get_value(source, x+x1, y+y1));
		}
	}
}


void grid_copy(Grid* destination, Grid* source) {
	int x, y;
	for (y = 0; y < destination->height; y++){
		for (x = 0; x < destination->width; x++){
		
			int location = y*destination->width + x;
			destination->values[location] = source->values[location];
		}
	}
}


void grid_set_value(Grid* g, int x, int y, double value){
/*
	if (x > g->width) exit(-1);
	if (y > g->height) exit(-1);
*/
	int location = y * g->width + x;
	g->values[location] = value;
	return;
}


double grid_get_value(Grid* g, int x, int y){
/*
	if (x > g->width) exit(-1);
	if (y > g->height) exit(-1);
*/
	int location = y*g->width + x;
	return g->values[location];
}


double	grid_get_max(Grid* g){
//	double max_value = g->values[0];
	double max_value = -INFINITY;
	double value;
	int x, y;
	for (y = 0; y < g->height; y++){
		for (x = 0; x < g->width; x++){
			value = grid_get_value(g, x, y);
			if(value > max_value) max_value = value;
		}
	}
	return max_value;
}


double	grid_get_min(Grid* g){
//	double min_value = g->values[0];
	double min_value = INFINITY;
	double value;
	int x, y;
	for (y = 0; y < g->height; y++){
		for (x = 0; x < g->width; x++){
			value = grid_get_value(g, x, y);
			if(value < min_value) min_value = value;
		}
	}
	return min_value;
}


Point grid_get_maxLoc(Grid* g)			// returns position and value of maximum
{
	double	value;
	Point	maxLoc;
	maxLoc.x = maxLoc.y = -1;
//	maxLoc.value = g->values[0];
	maxLoc.value = -INFINITY;

	int x, y;
	for (y = 0; y < g->height; y++){
		for (x = 0; x < g->width; x++){		
			value = grid_get_value(g, x, y);
			if(value > maxLoc.value) {
				maxLoc.x = x;
				maxLoc.y = y;
				maxLoc.value = value;
			}
		}
	}
	return maxLoc;
}


Point grid_get_minLoc(Grid* g)			// returns position and value of minimum
{
	double	value;
	Point	minLoc;
	minLoc.x = minLoc.y = -1;
//	minLoc.value = g->values[0];
	minLoc.value = INFINITY;

	int x, y;
	for (y = 0; y < g->height; y++){
		for (x = 0; x < g->width; x++){		
			value = grid_get_value(g, x, y);
			if(value < minLoc.value) {
				minLoc.x = x;
				minLoc.y = y;
				minLoc.value = value;
			}
		}
	}
	return minLoc;
}


double grid_get_average(Grid *g) {
	double total=0.0;
	double value;
	int x, y;
	int N;
	for (N = y = 0; y < g->height; y++) {
//		for (x = 0; x < g->width; x++) total += grid_get_value(g, x, y);
		for (x = 0; x < g->width; x++) {
			value = grid_get_value(g, x, y);
			if (!(value==value)) continue;
			total += value;
			N++;
		}
	}
//	return total/(g->height*g->width);
	return total/(double)N;
}


double grid_get_total(Grid *g){
	double total=0.;
	double value;
	int x, y;
	for (y = 0; y < g->height; y++) {
//		for (x = 0; x < g->width; x++) total += grid_get_value(g, x, y);
		for (x = 0; x < g->width; x++) {
			value = grid_get_value(g, x, y);
			if (value==value) total += value;
		}
	}
	return total;
}


void grid_subtract(Grid* minuend, Grid* subtrahend, double minimum) {
	/* performs subtraction of subtrahend from minuend, and returns the result in minuend */
	/* requires that the two grids be of the same size */
	int x, y;
	for (y = 0; y < minuend->height; y++){
		for (x = 0; x < minuend->width; x++){
			grid_set_value( minuend, x, y, max(grid_get_value(minuend, x, y) - grid_get_value(subtrahend, x, y), minimum) );
		}
	}
}



/* *********************************************************************************************** */
/* ************************************* 1-byte Grids, GridB ************************************* */

GridB* gridB_new(int width, int height) {		/* also all points are initialized to zero using calloc */
	GridB* g = malloc(sizeof(GridB));	/*create the struct*/
	if (!g) exit(ENOMEM);				/* Not enough space. */
	g->height = height;
	g->width = width;
	g->values = calloc( height*width, sizeof(bool));
	if (!g->values) exit(ENOMEM);		/* Not enough space. */
	return g;
}


GridB* gridB_new_copy(GridB* g){
	GridB* g2 = gridB_new(g->width, g->height);
	gridB_copy(g2, g);
	return g2;
}


GridB* gridB_new_copy_region(GridB* source, int x1, int y1, int x2, int y2){
	/*create a new grid just large enough to hold the indicated region*/
	/*x2/y2 inclusive*/
	int new_height = y2-y1+1;
	int new_width = x2-x1+1;
	GridB* destination = gridB_new(new_width, new_height);
	gridB_copy_region(destination, source, x1, y1, x2, y2);
	return destination;
}


void gridB_copy_region(GridB* destination, GridB* source, int x1, int y1, int x2, int y2){
	int new_height = y2-y1+1;
	int new_width = x2-x1+1;
	int x, y;
	for (y = 0; y < new_height; y++) {
		for (x = 0; x < new_width; x++) {
			gridB_set_value(destination, x, y, gridB_get_value(source, x+x1, y+y1));
		}
	}
}


void gridB_copy(GridB* destination, GridB* source) {
	int x, y;
	int location;
	for (y = 0; y < destination->height; y++){
		for (x = 0; x < destination->width; x++){
			location = y*destination->width + x;
			destination->values[location] = source->values[location];
		}
	}
}


void gridB_set_region(GridB* mask, int x1, int y1, int x2, int y2, bool value) {
	int x, y;
	for (y = y1; y <= y2; y++) {
		for (x = x1; x <= x2; x++) gridB_set_value(mask, x, y, value);
	}
}


void gridB_set_value(GridB* g, int x, int y, bool value){
/*	if (x > g->width) exit(-1);
 *	if (y > g->height) exit(-1);
 */
	int location = y * g->width + x;
	g->values[location] = value;
	return;
}


bool gridB_get_value(GridB* g, int x, int y){
/*	if (x > g->width) exit(-1);
 *	if (y > g->height) exit(-1);
 */
	int location = y*g->width + x;
	return g->values[location];
}


long gridB_get_total(GridB *g){
	long	total=0;
	int x, y;
	for (y = 0; y < g->height; y++){
		for (x = 0; x < g->width; x++) total += gridB_get_value(g, x, y) ? 1 : 0;
	}
	return total;
}




/* *********************************************************************************************** */
/* ********************************* double Grids, Grid & GridB ********************************** */

size_t grid_count_NaNs(Grid* g, GridB* m);
size_t grid_count_NaNs(		/* returns number of NaNs */
Grid*	g,					/* the grid with values (double) */
GridB*	m)					/* the mask, only use points with m==0 */
{
	double	value;
	size_t NaNs=0;
	int		x, y;
	for (y = 0; y < g->height; y++) {
		for (x = 0; x < g->width; x++) {
			if (m) {
				if (gridB_get_value(m, x, y)) continue;		/* skip masked pixels */
			}
			value = grid_get_value(g, x, y);
			NaNs += (value!=value);
		}
	}
	return NaNs;
}


GridStats grid_get_stats(	/* returns position and value of max, min,... , only uses masked points */
Grid*	g,					/* the grid with values (double) */
GridB*	m)					/* the mask, only use points with m==0 */
{
	double	value;
	int		x, y;
	GridStats s;
	s.valMax = -INFINITY;		/* init to very small */
	s.valMin = INFINITY;		/* init to very big */
	s.xMax = s.yMax = -1;		/* init position to bad number */
	s.xMin = s.yMin = -1;
	s.Npts = 0;
	s.total = 0.;
	s.xCOM = s.yCOM = 0.;
	s.NaNs=grid_count_NaNs(g,m);

	for (y = 0; y < g->height; y++) {
		for (x = 0; x < g->width; x++) {
			if (m) {
				if (gridB_get_value(m, x, y)) continue;	/* skip masked pixels */
			}
			value = grid_get_value(g, x, y);
			if (value != value) continue;				/* skip NaNs */
			(s.Npts)++;									/* count valid points */
			s.total += value;	
			s.xCOM += value*x;							/* accumumlate the first moments */
			s.yCOM += value*y;
			if(value > s.valMax) {
				s.xMax = x;
				s.yMax = y;
				s.valMax = value;
			}
			if(value < s.valMin) {
				s.xMin = x;
				s.yMin = y;
				s.valMin = value;
			}
		}
	}

	s.xCOM = (s.total)!=0. ? (s.xCOM)/(s.total) : -1;
	s.yCOM = (s.total)!=0. ? (s.yCOM)/(s.total) : -1;
	s.average = (s.Npts)>0 ? (s.total)/((double)(s.Npts)) : 0.0;
	return s;
}


void grid_set_masked_val(	/* set all pixels outside the mask to value */
Grid*	g,					/* the grid with values (double) */
GridB*	m,					/* the mask, set pixels in g with m!=0 */
double	value)				/* set all masked pixels to value */
{
	int		x, y;
	if (!m || !g) return;	/* if no mask, do nothing */
	for (y = 0; y < g->height; y++) {
		for (x = 0; x < g->width; x++) {
			if (gridB_get_value(m,x,y)) grid_set_value(g,x,y, value);
		}
	}
	return;
}



