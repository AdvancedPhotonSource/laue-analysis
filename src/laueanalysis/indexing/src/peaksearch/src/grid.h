/**********************************************************

	Data Structure and Routines for working with
	a grid of data. This is used (in this case) for storing
	the contents of an SPE image file

/**********************************************************/

#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>

#include "minmax.h"
#include "point.h"

#ifndef _GRID_H_
#define _GRID_H_

typedef struct {			/* a grid of doubles values */
	double* values;
	int height;
	int width;
} Grid;

typedef struct {			/* a grid of 1-byte values (used for masks) */
	bool * values;
	int height;
	int width;
} GridB;

typedef struct {
	int		xMax;
	int		yMax;
	double	valMax;
	int		xMin;
	int		yMin;
	double	valMin;
	long	Npts;
	long	NaNs;
	double	total;
	double	average;
	double	xCOM;
	double	yCOM;
} GridStats;


/* ********* double values ********* */
Grid*	grid_new(int height, int width);
Grid*	grid_new_copy(Grid* g);
Grid*	grid_new_copy_region(Grid* g, int x1, int y1, int x2, int y2);
Grid*	grid_new_bin(Grid* g, int scale_exponent);
void	grid_copy(Grid* destination, Grid* source);
void	grid_copy_region(Grid* destination, Grid* source, int x1, int y1, int x2, int y2);
void	grid_set_value(Grid* g, int x, int y, double value);
double	grid_get_value(Grid* g, int x, int y);
double	grid_get_max(Grid* g);
double	grid_get_min(Grid* g);
Point	grid_get_maxLoc(Grid* g);
Point	grid_get_minLoc(Grid* g);
double	grid_get_average(Grid* g);
double	grid_get_total(Grid * g);
void	grid_subtract(Grid* minuend, Grid* subtrahend, double minimum);
GridStats	grid_get_stats(Grid* g, GridB* m);
void grid_set_masked_val(Grid* g, GridB* m, double value);

/* ********* 1-byte values ********* */
GridB*	gridB_new(int width, int height);
GridB*	gridB_new_copy(GridB* g);
GridB*	gridB_new_copy_region(GridB* source, int x1, int y1, int x2, int y2);
void	gridB_copy_region(GridB* destination, GridB* source, int x1, int y1, int x2, int y2);
void	gridB_copy(GridB* destination, GridB* source);
void	gridB_set_region(GridB* mask, int x1, int y1, int x2, int y2, bool value);
void	gridB_set_value(GridB* g, int x, int y, bool value);
bool	gridB_get_value(GridB* g, int x, int y);
long	gridB_get_total(GridB *g);

/* ********* All values ********* */
void	grid_delete(Grid* g);

#endif
