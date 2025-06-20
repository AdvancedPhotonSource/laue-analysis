//
//  cosmicFilter.h
//  reconstructC_Big
//
//  Created by Jon Tischler on 10/14/15.
//  Copyright © 2015 APS/ANL. All rights reserved.
//

#ifndef cosmicFilter_h
#define cosmicFilter_h

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


typedef struct				/* a short vector[7] used for filtering, with median information */
{
	double	v[7];
	size_t	m[7];			/* contains [0-6] with the intensity order, i is weakest if m[i]==0, i is strongest if m[i]==6 */
	size_t	med;			/* index of median, v[med] is median value */
	double	mLeft;			/* median of first 3, v[0,1,2] */
	double	mRight;			/* median of last 3, v[4,5,6], so if v[5] is the strongest, then m[5] should be 6 */


} boxM;

int cosmic_filter(double *vec, size_t N);
#ifdef DEBUG
void test_cosmic(void);
#endif

#endif /* cosmicFilter_h */
