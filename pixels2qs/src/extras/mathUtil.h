/*
 *  mathUtil.h
 *  reconstruct
 *
 *  Created by Jon Tischler on 1/16/09.
 *  Copyright 2009 ORNL. All rights reserved.
 *
 */

#define NORM(A) sqrt((A.x)*(A.x) + (A.y)*(A.y) + (A.z)*(A.z))
#define VEC3_MULTIPLY(A,B) { (A.x) *= B; (A.y) *= B; (A.z) *=B; }
#define DOT3(A,B) ((A.x)*(B.x) + (A.y)*(B.y) + (A.z)*(B.z))

#include "WireScanDataTypesN.h"

double distance(point_xyz a, point_xyz b);
int compare_double(double *a, double *b);
int rotationMatFromAxis(point_xyz axis, double angle, double mat[3][3]);
double determinant33(double a[3][3]);
point_xyz MatrixMultiply31(double a[3][3], point_xyz v);
void MatrixMultiply33(double a[3][3], double b[3][3], double c[3][3]);
void MatrixTranspose33(double a[3][3]);
