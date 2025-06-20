//
//  cosmicFilter.c
//  reconstructC_Big
//
//  Created by Jon Tischler on 10/14/15.
//  Copyright © 2015 APS/ANL. All rights reserved.
//

#include "cosmicFilter.h"
static void boxM_shiftR(boxM *b, double valRight);
static double filterWithBoxM(boxM b);
static void boxM_init(boxM *b);
static void print_boxM(boxM b);

//#ifndef MAX
//#define MAX(X,Y) ( ((X)<(Y)) ? (Y) : (X) )
//#endif
#ifndef MIN
#define MIN(X,Y) ( ((X)>(Y)) ? (Y) : (X) )
#endif

/* used to id a cosmic event */
static double big=5000;				/* a cosmic is big more than its immediate neighbors */
static double fac=5;				/* a cosmic is more than fac times bigger than its neighbors */

typedef struct						/* used for sorting */
{
	double	v;
	size_t	m;						/* contains [0-6] with the intensity order, i is weakest if m[i]==0, i is strongest if m[i]==6 */
} doubleIndexQS;

int compare_doubleIndexQS(doubleIndexQS *a, doubleIndexQS *b);	/* usded to sort a doubleIndexQS */

//#ifdef DEBUG				/* put these four lines in WireScan.c to test */
//	test_cosmic();
//	exit(1);
//#endif


/* filters vec[] to remove all points that look like cosmic events, vec is changed in place. */
int cosmic_filter(
double	*vec,
size_t	N)
{
	boxM	b;
	size_t	i;

	if (N<7) return 0;				/* too short, leave un-filtered */

	for (i=0;i<7;i++) b.v[i] = vec[i];	/* pre-set box for first 3 */
	boxM_init(&b);					/* set b.m[] & b.med, using only values in b.v[] */

	/* set the first 3 */
	if (vec[0] > b.mLeft+big || vec[0] > fac*(b.mLeft)) vec[0] = b.mLeft;
	if (vec[1] > b.mLeft+big || vec[1] > fac*(b.mLeft)) vec[1] = b.mLeft;
	if (vec[2] > b.mLeft+big || vec[2] > fac*(b.mLeft)) vec[2] = b.mLeft;

	vec[3] = filterWithBoxM(b);
	for (i=4;i<(N-3);i++) {
		boxM_shiftR(&b,vec[i+3]);	/* add vec[i+3] to right end of boxM, remove first value, and update */
		vec[i] = filterWithBoxM(b);
	}

	// and set the last 3
	if (vec[N-3] > b.mRight+big || vec[N-3] > fac*(b.mRight)) vec[N-3] = b.mRight;
	if (vec[N-2] > b.mRight+big || vec[N-2] > fac*(b.mRight)) vec[N-2] = b.mRight;
	if (vec[N-1] > b.mRight+big || vec[N-1] > fac*(b.mRight)) vec[N-1] = b.mLeft;

	return 0;
}



static double filterWithBoxM(
boxM	b)
{
	double	mMax;					/* max of b.mLeft & b.mRight */
	mMax = b.mLeft > b.mRight ? b.mLeft : b.mRight;

	if (b.v[3]> mMax+big || b.v[3]> fac*mMax) return (b.mLeft + b.mRight)/2.0;	/* found a cosmic in b.v[3] */
	return b.v[3];					/* no cosmic */
}



static void boxM_shiftR(			/* add valRight to right end of boxM, remove first value, and update b->m[], b->med, b->mLeft, and b->mRight */
boxM	*b,
double	valRight)
{
	size_t	i, m0;
	int		ilow;					/* number of values below valRight */

	m0 = b->m[0];					/* old value of m[0] */

	memmove(b->v, &(b->v[1]), 6*sizeof(b->v[0]));	/* slide down values of b->v, move [1,6] -> [0,5] */
	memmove(b->m, &(b->m[1]), 6*sizeof(b->m[0]));	/*   and b->m */
	b->v[6] = valRight;								/* now set [6] */

	/* find new intensity order for b->v[6] */
	for (i=0, ilow=0; i<7; i++) ilow += b->v[i] < valRight ? 1 : 0;
	b->m[6] = ilow;

	/* and readjust the first 6 b->m[]'s, and set b->med */
	b->med = ilow;
	for (i=0;i<6;i++) {
		b->m[i] += b->m[i] > m0 ? -1 : 0;
		b->m[i] += b->m[i] >= ilow ? 1 : 0;
		if (b->m[i] == 3) b->med = i;
	}

	if (b->m[0] > b->m[1] && b->m[2] > b->m[1]) {		/* 1 is smallest, must be 0 or 2 */
		if (b->m[0] < b->m[2]) b->mLeft = b->v[0];
		else b->mLeft = b->v[2];
	}
	else if (b->m[1] > b->m[0] && b->m[2] > b->m[0]) {	/* 0 is smallest, must be 1 or 2 */
		if (b->m[1] < b->m[2]) b->mLeft = b->v[1];
		else b->mLeft = b->v[2];
	}
	else if (b->m[1] > b->m[2] && b->m[0] > b->m[2]) {	/* 2 is smallest, must be 0 or 1 */
		if (b->m[1] < b->m[0]) b->mLeft = b->v[1];
		else b->mLeft = b->v[0];
	}

	if (b->m[4] > b->m[5] && b->m[6] > b->m[5]) {		/* 5 is smallest, must be 4 or 6 */
		if (b->m[4] < b->m[6]) b->mRight = b->v[4];
		else b->mRight = b->v[6];
	}
	else if (b->m[5] > b->m[4] && b->m[6] > b->m[4]) {	/* 4 is smallest, must be 5 or 6 */
		if (b->m[5] < b->m[6]) b->mRight = b->v[5];
		else b->mRight = b->v[6];
	}
	else if (b->m[5] > b->m[6] && b->m[4] > b->m[6]) {	/* 6 is smallest, must be 4 or 5 */
		if (b->m[5] < b->m[4]) b->mRight = b->v[5];
		else b->mRight = b->v[4];
	}
}


static void boxM_init(									/* set b->m[] & b->med, using only values in b->v[] */
boxM	*b)
{
	size_t	i;
	doubleIndexQS array[7];

	/* put intensity order into b->m[] */
	for (i=0;i<7;i++) {
		array[i].v = b->v[i];
		array[i].m = i;
	}
	qsort(array, 7, sizeof(doubleIndexQS), (void *)compare_doubleIndexQS);
	for (i=0;i<7;i++) b->m[array[i].m] = i;

	for (i=0;i<7;i++) { if (b->m[i] == 3) break; }		/* find b.med */
	b->med = i;						/* where b->m[i]==3, v[i] is median value */

	if (b->m[0] > b->m[1] && b->m[2] > b->m[1]) {		/* 1 is smallest, must be 0 or 2 */
		if (b->m[0] < b->m[2]) b->mLeft = b->v[0];
		else b->mLeft = b->v[2];
	}
	else if (b->m[1] > b->m[0] && b->m[2] > b->m[0]) {	/* 0 is smallest, must be 1 or 2 */
		if (b->m[1] < b->m[2]) b->mLeft = b->v[1];
		else b->mLeft = b->v[2];
	}
	else if (b->m[1] > b->m[2] && b->m[0] > b->m[2]) {	/* 2 is smallest, must be 0 or 1 */
		if (b->m[1] < b->m[0]) b->mLeft = b->v[1];
		else b->mLeft = b->v[0];
	}

	if (b->m[4] > b->m[5] && b->m[6] > b->m[5]) {		/* 5 is smallest, must be 4 or 6 */
		if (b->m[4] < b->m[6]) b->mRight = b->v[4];
		else b->mRight = b->v[6];
	}
	else if (b->m[5] > b->m[4] && b->m[6] > b->m[4]) {	/* 4 is smallest, must be 5 or 6 */
		if (b->m[5] < b->m[6]) b->mRight = b->v[5];
		else b->mRight = b->v[6];
	}
	else if (b->m[5] > b->m[6] && b->m[4] > b->m[6]) {	/* 6 is smallest, must be 4 or 5 */
		if (b->m[5] < b->m[4]) b->mRight = b->v[5];
		else b->mRight = b->v[4];
	}
 }

int compare_doubleIndexQS(doubleIndexQS *a, doubleIndexQS *b)
{
	return (a->v > b->v) - (a->v < b->v);
}



#ifdef DEBUG
void test_cosmic()
{
	double v0[] = { 100, 102, 101, 103, 207, 209, 205};
	int		i;
	boxM	b;
	for (i=0;i<7;i+=1) {
		b.v[i] = v0[i];	/* set the boxM */
		b.m[i] = 20;
	}
	boxM_init(&b);				/* set b.m[] & b.med, using only values in b.v[] */
	print_boxM(b);

	printf("\n");
	double v1[] = {241,238,231,281,252,241,230,207,245,222,10000,224,203,216,257,200,209,268,223,203};
	v1[3] += 6000;
	v1[19] = round(v1[19]*7);

	for (i=0;i<20;i++) printf("%4d  ",i);
	printf("\n");
	for (i=0;i<20;i++) printf("%5g,",v1[i]);
	printf("\n");

	cosmic_filter(v1,20);
	for (i=0;i<20;i++) printf("%5g,",v1[i]);
	printf("\n");
}

static void print_boxM(boxM b)			/* print contents of boxM structure */
{
	printf("\nmedians are: left=%g,  med=%lu, right=%g\n",b.mLeft,b.med,b.mRight);
	for (int i=0;i<7;i+=1) {
		printf("%lu   %g\n",b.m[i],b.v[i]);
	}
}
#endif



