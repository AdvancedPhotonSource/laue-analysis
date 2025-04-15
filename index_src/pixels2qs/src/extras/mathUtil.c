/*
 *  mathUtil.c
 *  reconstruct
 *
 *  Created by Jon Tischler on 1/16/09.
 *  Copyright 2009 ORNL. All rights reserved.
 *
 */

#include <math.h>
#include "mathUtil.h"

/* some math utility routines */


double distance(point_xyz a, point_xyz b)		/* distance between points a and b */
{
	double dx,dy,dz;
	dx = b.x - a.x;
	dy = b.y - a.y;
	dz = b.z - a.z;
	return sqrt(dx*dx + dy*dy + dz*dz);
}

int compare_double(double *a, double *b)
{
	if (*a == *b) return 0;
	else if (*a < *b) return -1;
	else return 1;
}


/* set mat to be a rotation matrix about axis with angle */
int rotationMatFromAxis(
point_xyz	axis,				/* axis about which to rotate (or possibly Rodriques vector lenght is angle in radian) */
double		angle,				/* angle to rotate (degrees), assumes axis is true Rotation vector if angle invalid (radian) */
double 		mat[3][3])			/* desired rotation matrix */
{
	double	len;
	double	nx,ny,nz;
	double	cosa, sina, c1;		/* cos, sin, 1-cos */

	if (!mat) return 1;			/* probably a big mistake */
	len = NORM(axis);
	angle = (angle==angle) ? angle : len;	/* rotation angle (radian) */

	if (angle==0) {							/* zero angle rotation is just the identity matrix */
		mat[0][0] = mat[1][1] = mat[2][2] = 1;
		mat[0][1] = mat[0][2] = mat[1][0] = mat[1][2] = mat[2][0] = mat[2][1] = 0;
		return 0;
	}

	nx = (axis.x) / len;					/* normalized components of axis */
	ny = (axis.y)/len;
	nz = (axis.z)/len;
	sina = sin(angle);
	cosa = cos(angle);
	c1 = 1-cosa;

	/* from		http://mathworld.wolfram.com/RodriguesRotationFormula.html (I double checked this too.) */
	mat[0][0] = cosa+nx*nx*c1;			mat[0][1] =nx*ny*c1-nz*sina;		mat[0][2] = nx*nz*c1+ny*sina;
	mat[1][0] = nx*ny*c1+nz*sina;		mat[1][1] = cosa+ny*ny*c1;			mat[1][2] =ny*nz*c1-nx*sina;
	mat[2][0] = nx*nz*c1-ny*sina;		mat[2][1] = ny*nz*c1+nx*sina;		mat[2][2] = cosa+nz*nz*c1;
	return 0;
}

double determinant33(
double  a[3][3])
{
	double det;
	det =  a[0][0]*a[1][1]*a[2][2] - a[0][0]*a[2][1]*a[1][2];
	det += a[1][0]*a[2][1]*a[0][2] - a[1][0]*a[0][1]*a[2][2];
	det += a[2][0]*a[0][1]*a[1][2] - a[2][0]*a[1][1]*a[0][2];
	return det;
}



point_xyz MatrixMultiply31(			/* returns a x v */
double		a[3][3],
point_xyz	v)
{
	point_xyz c;
	double v0,v1,v2;
	if (!a) exit(1);								/* failure, arrays must exist */
	v0 = v.x;  v1 = v.y;  v2 = v.z;					/* in case v and c are the same address, ie 'c = a*c' */
	c.x = a[0][0]*v0 + a[0][1]*v1 + a[0][2]*v2;		/* c = a*v */
	c.y = a[1][0]*v0 + a[1][1]*v1 + a[1][2]*v2;
	c.z = a[2][0]*v0 + a[2][1]*v1 + a[2][2]*v2;
	return c;
}



/* this version works whether c in case c equals a or b,   ie a = a*b */
void MatrixMultiply33(				/* c = a x b */
double	a[3][3],
double	b[3][3],
double  c[3][3])
{
	if (!a || !b || !c) exit(1);	/* failure, arrays must exist */
	else if (a != c && b != c) {	/* c is distinct from a and b, fastest.  ie not a  = a*b */
		c[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0];  /* c = a*b */
		c[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0];
		c[2][0] = a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0];

		c[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1];
		c[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1];
		c[2][1] = a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1];

		c[0][2] = a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2];
		c[1][2] = a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2];
		c[2][2] = a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2];
	}
	else {
		double  mat[3][3];			/* temp to hold value */

		mat[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0];	/* mat = a*b */
		mat[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0];
		mat[2][0] = a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0];

		mat[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1];
		mat[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1];
		mat[2][1] = a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1];

		mat[0][2] = a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2];
		mat[1][2] = a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2];
		mat[2][2] = a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2];

		c[0][0] = mat[0][0]; c[0][1] = mat[0][1]; c[0][2] = mat[0][2];		/* c = mat */
		c[1][0] = mat[1][0]; c[1][1] = mat[1][1]; c[1][2] = mat[1][2];
		c[2][0] = mat[2][0]; c[2][1] = mat[2][1]; c[2][2] = mat[2][2];
	}
}



void MatrixTranspose33(				/* transpose the 3x3 matrix a */
double	a[3][3])
{
	double temp;
	temp=a[0][1]; a[0][1]=a[1][0]; a[1][0]=temp;
	temp=a[0][2]; a[0][2]=a[2][0]; a[2][0]=temp;
	temp=a[1][2]; a[1][2]=a[2][1]; a[2][1]=temp;
}

