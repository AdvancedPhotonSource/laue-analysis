#include "grid.h"
#include "point.h"
#ifndef _GRID_OPERATIONS_H_
#define _GRID_OPERATIONS_H_

Grid*	grid_new_bin(Grid* g, int scale_exponent);
Grid*	grid_new_upscale(Grid* g, int scale_exponent);
void	grid_smooth_boxcar(Grid* g, int range);
void	grid_smooth_median(Grid* g, int range);
void	grid_smooth_gauss(Grid* g, int range);
void	shell_sort(double A[], int size);
double median(double A[], int size);
Point*	centroid(Grid* image, int shiftx, int shifty);
Point*	centroid_2(Grid* image, int shiftx, int shifty);
#endif
