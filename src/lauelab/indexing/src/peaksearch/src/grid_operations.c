#include "grid_operations.h"

#include <assert.h>
#include <math.h>

#ifndef MAX
#define MAX(X,Y) ( ((X)<(Y)) ? (Y) : (X) )
#endif
#ifndef MIN
#define MIN(X,Y) ( ((X)>(Y)) ? (Y) : (X) )
#endif

//apply smooth filter (averaging) on image with filter size 2*range+1 by 2*range+1
int grid_smooth_boxcar(Grid* g, int range) {

	/* here, range is the distance from the centrepoint to an outer edge - a radius, were this a circle */
	/* this is to prevent odd or undefined behaviour were an even number to be passed as a width value */
	
	double total_value = 0.0;
	int pixels_counted = 0;
	
	Grid* smoothed_grid = grid_new_copy(g);
	if (!smoothed_grid) return 1;
	
	int x, y;
	for (x = 0; x < g->width; x++){
		for (y = 0; y < g->height; y++){
		
		
			total_value = 0.0;
			pixels_counted = 0;
			
		
			/* loop over the region around the centrepoint, bounded by image size*/
			int x2, y2;
			for (x2 = max(x-range, 0); x2 <= min(x+range, g->width-1); x2++){
				for (y2 = max(y-range, 0); y2 <= min(y+range, g->height-1); y2++){
				
					pixels_counted++;
					total_value += grid_get_value(g, x2, y2);
				
				}
			}
			grid_set_value(smoothed_grid, x, y, total_value / pixels_counted);
			/* end loops around centrepoint */
		
		}
	}
	
	/* copy the resutls back into the first grid and delete the temporary one */
	grid_copy(g, smoothed_grid);
	grid_delete(smoothed_grid);
	return 0;
}

int grid_smooth_median(Grid* g, int range) {
	
	/* here, range is the distance from the centrepoint to an outer edge - a radius, were this a circle */
	/* this is to prevent odd or undefined behaviour were an even number to be passed as a width value */
	
	double median_values[(2*range+1)*(2*range+1)];	/* temporary array used to calculate the median */
	int pixels_counted = 0;
	double median=0.0;
	int x, y, x2, y2;
	
	Grid* smoothed_grid = grid_new_copy(g);
	if (!smoothed_grid) return 1;
	
	//perform median filter on the image, the size of the filter is 2*range+1 by 2*range+1
	for (x = 0; x < g->width; x++){
		for (y = 0; y < g->height; y++){
			pixels_counted = 0;
			/* loop over the region around the centrepoint, bounded by image size*/
			for (x2 = max(x-range, 0); x2 <= min(x+range, g->width-1); x2++) {
				for (y2 = max(y-range, 0); y2 <= min(y+range, g->height-1); y2++) {
					median_values[pixels_counted] = grid_get_value(g, x2, y2);
					pixels_counted++;
				}
			}
			shell_sort(median_values, pixels_counted);
			if (pixels_counted % 2 == 0) median = (median_values[pixels_counted / 2 - 1] + median_values[pixels_counted / 2]) / 2.0;
			else						 median = median_values[pixels_counted / 2];
			grid_set_value(smoothed_grid, x, y, median);
		} /* end loops around centrepoint */
	}
	
	/* copy the resutls back into the first grid and delete the temporary one */
	grid_copy(g, smoothed_grid);
	grid_delete(smoothed_grid);
	return 0;
}


int grid_smooth_gauss(Grid* g, int Nf2) {
	/* use a Gaussian kernel that is (Nf x Nf).  NOTE, Nf must be 5! */
	/* see:	http://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm */
	/* this routine uses zero for edge pixels where the kernel extends beyond the image */
	/* the value for the kernel comes from the Igor "MatrixFilter gauss" routine, shold be easy to calculate this */

	size_t	Nf=(2*Nf2)+1;		/* total size of the filter, this is always odd */
	size_t	Nx,Ny;				/* size of image */
	double	*filter;			/* single line of the filter */
	long	m;					/* index over filter */
	double	sumFilter;			/* used to normalize filter */
	double	arg, val;
	size_t	ip,jp;				/* current pixel being filtered */
	size_t	lo,hi;				/* loop over the kernel */

	if (Nf2!=2) return 0;							/* do nothing if Nf2 not 2,  (so Nf not 5), FIX this limitation */
	if (Nf2<1) return 0;							/* Nf2 must be at least 1 */

	Nx = g->width;
	Ny = g->height;

	if (!(filter=calloc(Nf,sizeof(double)))) return 1;

	for (m=0,sumFilter=0.0; m<Nf; m++) {		/* set filter values */
		arg = (m-Nf2)*0.585;
		val = exp(-arg*arg);
		filter[m] = val;
		sumFilter += val;						/* save sum for normalization */
	}
	for (m=0;m<Nf;m++) filter[m] /= sumFilter;	/* normalize the filter */

	Grid* temp_grid = grid_new_copy(g);			/* temp copy of image for smoothing */
	if (!temp_grid) { free(filter); return 1; }

	for (ip=0;ip<Nx;ip++) {						/* smooth along the X direction */
		lo = ip > (size_t)Nf2 ? ip - (size_t)Nf2 : 0;
		hi = MIN(ip + (size_t)Nf2, Nx - 1);
		for (jp=0;jp<Ny;jp++) {
			for (m=lo,val=0.0; m<=hi; m++) val += grid_get_value(g,m,jp)*filter[m+Nf2-ip];
			grid_set_value(temp_grid,ip,jp,val);
		}
	}

	for (jp=0;jp<Ny;jp++) {						/* smooth along the Y direction */
		lo = jp > (size_t)Nf2 ? jp - (size_t)Nf2 : 0;
		hi = MIN(jp + (size_t)Nf2, Ny - 1);
		for (ip=0;ip<Nx;ip++) {
			for (m=lo,val=0.0; m<=hi; m++) val += grid_get_value(temp_grid,ip,m)*filter[m+Nf2-jp];
			grid_set_value(g,ip,jp,val);
		}
	}
	
	grid_delete(temp_grid);						/* done with temporary grid, delete it */
	free(filter);
	return 0;
}


/* wikipedia saves you time */
void shell_sort(double A[], int size)
{
  int i, j, increment;
  double temp;
  increment = size / 2;
 
  while (increment > 0)
  {
    for (i = increment; i < size; i++) 
    {
      j = i;
      temp = A[i];
      while ((j >= increment) && (A[j-increment] > temp)) 
      {
        A[j] = A[j - increment];
        j = j - increment;
      }
      A[j] = temp;
    }
 
    if (increment == 2)
       increment = 1;
    else 
       increment = (int) (increment / 2.2);
  }
}

double median(double A[], int size){
  shell_sort(A,size);
  
  return A[size/2];
} 


/* for finding the center of a small blob, i.e.,size < 10*10 */
Point* centroid(Grid* image, int shiftx, int shifty){

	/* calculate a weighted average for a centre point, i.e. the Center of Mass */
	
	double xcent = 0.0;
	double ycent = 0.0;
	double total = 0.0;
	double value;
	
	/*iterate over every element in image*/
	int x, y;
	for (x = 0; x < image->width; x++){
		for (y = 0; y < image->height; y++){
			value = grid_get_value(image, x, y);
			if (!(value==value)) continue;			/* skip NaNs */
			xcent += ((double)(x)) * value;
			ycent += ((double)(y)) * value;
			total += value;
		}
	}
	xcent /= total;
	ycent /= total;

	/* if the centre coords are out of bounds, panic, and just pick the middle*/	
	if ( (xcent < 0 || xcent > image->width-1) || (ycent < 0 || ycent > image->height-1) ){
		xcent = image->width / 2.0;
		ycent = image->height / 2.0;
	}
	//	printf("xcent=%f, ycent=%f, maxint=%f\n",xcent+shiftx, ycent+shifty,grid_get_max(image));
	/*create a new point using the data calculated here, and return it - use the maximum value from inside the image as the value*/
	return point_new_initialized(xcent+shiftx, ycent+shifty, grid_get_max(image));
	
}

/*for finding the center of a large blob, i.e. size >=10*10*/
Point* centroid_2(Grid* image, int shiftx, int shifty){

	/*calculate a weighted average for a centre point*/
	
	double xcent = 0.0;
	double ycent = 0.0;
	double value = 0.0;
	double total = 0.0;
	int n=0;
	//printf("11111111\n");
	double maxma=grid_get_max(image);
	/*iterate over every element in image*/
	int x, y;
	for (x = 0; x < image->width; x++){
		for (y = 0; y < image->height; y++){
					
			value = grid_get_value(image, x, y);
                        if(value==maxma){
			 xcent += ((double)(x)) * value;
			 ycent += ((double)(y)) * value;
			 total += value;
			 n++;
			}
		
		}
	}

	xcent /= total;
	ycent /= total;
        //printf("large blobs:totalNumMaxma =%d maxiam=%lf\n",n,value);
	/* if the centre coords are out of bounds, panic, and just pick the middle*/	
	if ( (xcent < 0 || xcent > image->width-1) || (ycent < 0 || ycent > image->height-1) ){
		xcent = image->width / 2.0;
		ycent = image->height / 2.0;
	}

	/*create a new point using the data calculated here, and return it - use the maximum value from inside the image as the value*/
	return point_new_initialized(xcent+shiftx, ycent+shifty, grid_get_max(image));
	
}
