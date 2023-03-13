#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _MSC_VER				/* identifies this as a Microsoft compiler */
#define _USE_MATH_DEFINES	/* added RX2011 */
#endif
#include <math.h>
#include "mathUtil.h"


#define MIN(A,B)	( (A)<(B)?(A):(B) )
#define MAX(A,B)	( (A)>(B)?(A):(B) )
/* #define CHECK_FREE(A)   { if(A) free(A); (A)=NULL;}	// moved to mathUtil.h */
#define MAG3(A,B,C) sqrt((A)*(A)+(B)*(B)+(C)*(C))

/*
 *  long allowFCC(long h, long k, long l);
 *  long allowBCC(long h, long k, long l);
 *  long allowDIA(long h, long k, long l);
 *  long allowHEX(long h, long k, long l);
 */

/* delete numDel bytes starting at ptr, this does not de-allocate the space, just moves points */
void DeletePoints(
size_t	len,			/* total number of existing points from ptr */
void	*ptr,			/* points to the start of first point to delete */
size_t  pntLen,			/* length of one point in bytes */
size_t  numDel)			/* number of points to delete */
{
	size_t  num;		/* total number of bytes to remove */
#ifdef _MSC_VER			/* identifies this as a Microsoft compiler */
	char * ptr00;		/* added RX2011 (char is 1 byte in size, which is what we need) */
#endif

	if (!ptr || numDel<1 || len<=numDel) return;			/* nothing to do */
	num = numDel*pntLen;

#ifdef _MSC_VER			/* identifies this as a Microsoft compiler */
	ptr00 = (char*)ptr;
	memmove(ptr00,ptr00+num,(len-numDel)*pntLen);
	ptr = ptr00;
#else
	memmove(ptr,ptr+num,(len-numDel)*pntLen);
#endif
}
/*
 *  void testDeletePoints();
 *  void testDeletePoints()
 *  {
 *  	double  a[20];
 *  	int		m[20];
 *  	long	i,N=20;
 *
 *  	for (i=0;i<N;i++) a[i] = i;
 *			fprintf(fout,"a = "); for (i=0;i<N;i++) fprintf(fout,"%3.0f  ",a[i]); fprintf(fout,"\n");
 *			fprintf(fout,"DeletePoints(20,a,sizeof(double),2);\n");
 *  	DeletePoints(20,a,sizeof(double),2);
 *			fprintf(fout,"a = "); for (i=0;i<N;i++) fprintf(fout,"%3.0f  ",a[i]); fprintf(fout,"\n");
 *
 *  	fprintf(fout,"\n");
 *  	for (i=0;i<N;i++) m[i] = i;
 *			fprintf(fout,"m = "); for (i=0;i<N;i++) fprintf(fout,"%3d  ",m[i]); fprintf(fout,"\n");
 *  	DeletePoints(20-3,&(m[3]),sizeof(int),4);
 *			fprintf(fout,"DeletePoints(20-3,&(m[3]),sizeof(int),4);\n");
 *			fprintf(fout,"m = "); for (i=0;i<N;i++) fprintf(fout,"%3d  ",m[i]); fprintf(fout,"\n");
 *  }
*/



void EulerMatrix(					/* make the rotation matrix M_Euler from Euler angles (radian) */
double	alpha,
double	beta,
double	gamma,
double	M_Euler[3][3])
{
	double	ca,cb,cg, sa,sb,sg;
	ca = cos(alpha);
	cb = cos(beta);
	sa = sin(alpha);
	sb = sin(beta);

	if (gamma==0) {					/* just a speed up for the frequent gamm=0 */
		M_Euler[0][0] = cb*ca;
		M_Euler[1][0] = -sa;
		M_Euler[2][0] = sb*ca;

		M_Euler[0][1] = cb*sa;
		M_Euler[1][1] = ca;
		M_Euler[2][1] = sb*sa;

		M_Euler[0][2] = -sb;
		M_Euler[1][2] = 0;
		M_Euler[2][2] = cb;
		return;
	}

	cg = cos(gamma);
	sg = sin(gamma);

	M_Euler[0][0] = cg*cb*ca - sg*sa;
	M_Euler[1][0] = -sg*cb*ca - cg*sa;
	M_Euler[2][0] = sb*ca;

	M_Euler[0][1] = cg*cb*sa + sg*ca;
	M_Euler[1][1] = -sg*cb*sa + cg*ca;
	M_Euler[2][1] = sb*sa;

	M_Euler[0][2] = -cg*sb;
	M_Euler[1][2] = sg*sb;
	M_Euler[2][2] = cb;
}



void rot2EulerAngles(
double	A[3][3],					/* the rotation matrix that we know */
double	*alpha,						/* three Euler angles that we want (radian)*/
double	*beta,
double	*gamma)
{
	double	v[3];
	double	RzInv[3][3];
	double	RyInv[3][3];
	double	zhat[3]={0.,0.,1.};
	double	mat[3][3];
	double	Rzalpha[3][3];
	/* Rz(gamma) * zhat = zhat
	 *	and A = Rz(gamma)*Ry(beta)*Rz(alpha), and Rz*zhat = zhat
	 *  A*zhat = Rz(gamma)*Ry(beta)*zhat
	 *	for v=A*zhat, then v[2] gives cos(beta), and v[0],v[[1] gives azimuth
	 *	then, Rz(gamma)*Ry(beta)*Rz(alpha) = A,  so
	 *	Rz(alpha) = RyInv(beta)*RzInv(gamma) * A
	 *	and Rz[0][0] = cos(angle), and Rz[0][1] = sin(angle)
	 */

	MatrixMultiply31(A,zhat,v);			/* Rz(gamma) * zhat = zhat */
	normalize3(v);						/* should not be necessary, but A may not be perfect */
	*beta = acos(v[2]);					/* v[2] is v .dot. (001) */
	*gamma = M_PI - atan2(v[1],v[0]);   /* azimuthal angle */

	MatrixRz(RzInv,-(*gamma));			/* Rz(-gamma) is Inverse(Rz(gamma)) */
	MatrixRy(RyInv,-(*beta));

	/* Rz(alpha) = RyInv(beta)*RzInv(gamma)*A */
	MatrixMultiply33(RzInv,A,mat);
	MatrixMultiply33(RyInv,mat,Rzalpha);			/* Rzalpha is now Rz(alpha) */
	*alpha = atan2(Rzalpha[0][1],Rzalpha[0][0]);	/* atan2( sin(alpha), cos(alpha) ) */
	return;
}



void MatrixRz(						/* rotation matrix about the z axis thru angle (radian) */
double	Rz[3][3],
double	angle)
{
	double cosine, sine;
	if (!Rz) return;
	cosine = cos(angle);
	sine = sin(angle);
	Rz[2][0] = Rz[2][1] = Rz[0][2] = Rz[1][2] = 0.;
	Rz[0][0] = cosine;
	Rz[1][1] = cosine;
	Rz[2][2] = 1;
	Rz[0][1] = sine;
	Rz[1][0] = -sine;
}

void MatrixRy(						/* rotation matrix about the y axis thru angle (radian) */
double	Ry[3][3],
double	angle)
{
	double cosine, sine;
	if (!Ry) return;
	cosine = cos(angle);
	sine = sin(angle);
	Ry[1][0] = Ry[0][1] = Ry[2][1] = Ry[1][2] = 0.;
	Ry[0][0] = cosine;
	Ry[1][1] = 1;
	Ry[2][2] = cosine;
	Ry[2][0] = sine;
	Ry[0][2] = -sine;
}



int gcf(							/* find greatest common factor of n1,n2,n3 */
int		n1,							/* note that gcf(0,0,0) returns 0 */
int		n2,
int		n3)
{
	int		i, n;
	n1 = abs(n1);
	n2 = abs(n2);
	n3 = abs(n3);
	n = (n1>n2) ? n1 : n2;			/* n is greatest of n1,n2,n3 */
	n = (n3>n)  ? n3  : n;
	if (n==0) return 0;				/* gcf(0,0,0) = 0 */
	i = n;
	while((n1%i || n2%i || n3%i) && i>1) i--;
	return i;
}








double normalize3(					/* normalize a, and return the starting magnitude */
double a[3])
{
	double norm;
	norm = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
	a[0] /= norm;
	a[1] /= norm;
	a[2] /= norm;
	return norm;
}



double dot3(						/* dot product of two 3-vectors */
double	a[3],
double	b[3])
{
	return (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]);
}



void cross(							/* vector cross product,   c = a x b */
double	a[3],
double	b[3],
double	c[3])
{
	double	cc[3];					/* used to buffer against c being the same as a or b */
	cc[0] = a[1]*b[2] - a[2]*b[1];
	cc[1] = a[2]*b[0] - a[0]*b[2];
	cc[2] = a[0]*b[1] - a[1]*b[0];
	c[0]=cc[0]  ;  c[1]=cc[1]  ;  c[2]=cc[2];
	return;
}



void vector3cons(					/* multiply each element of a[3] by x */
double  a[3],
double  x)
{
	a[0] *= x;
	a[1] *= x;
	a[2] *= x;
	return;
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



void MatrixMultiply31(				/* c = a x v */
double	a[3][3],
double	v[3],
double	c[3])
{
	double v0,v1,v2;
	if (!a || !v || !c) exit(1);					/* failure, arrays must exist */
	v0 = v[0];  v1 = v[1];  v2 = v[2];				/* in case v and c are the same address, ie 'c = a*c' */
	c[0] = a[0][0]*v0 + a[0][1]*v1 + a[0][2]*v2;	/* c = a*v */
	c[1] = a[1][0]*v0 + a[1][1]*v1 + a[1][2]*v2;
	c[2] = a[2][0]*v0 + a[2][1]*v1 + a[2][2]*v2;
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



/* I could use:
 * memcpy(dest,source*sizeof(double));
 * but it really is no faster
 */
void MatrixCopy33(					/* copy, dest[][] = source[][] */
double  dest[3][3],
double  source[3][3])
{
	dest[0][0]=source[0][0]; dest[0][1]=source[0][1]; dest[0][2]=source[0][2];
	dest[1][0]=source[1][0]; dest[1][1]=source[1][1]; dest[1][2]=source[1][2];
	dest[2][0]=source[2][0]; dest[2][1]=source[2][1]; dest[2][2]=source[2][2];
}



/* find total difference between two vectors */
double diff3(
double  a[3],
double  b[3])
{
	return fabs(a[0]-b[0])+fabs(a[1]-b[1])+fabs(a[2]-b[2]);
}



double matsDelta(					/* returns sum(|a-b|) */
double  a[3][3],
double  b[3][3])
{
	double  sum=0;

	sum += fabs(a[0][0]-b[0][0]);
	sum += fabs(a[1][0]-b[1][0]);
	sum += fabs(a[2][0]-b[2][0]);

	sum += fabs(a[0][1]-b[0][1]);
	sum += fabs(a[1][1]-b[1][1]);
	sum += fabs(a[2][1]-b[2][1]);

	sum += fabs(a[0][2]-b[0][2]);
	sum += fabs(a[1][2]-b[1][2]);
	sum += fabs(a[2][2]-b[2][2]);

	return sum;
}



char *num2sexigesmal(				/* convert seconds into a hh:mm:ss.sss   string */
char	str[60],
double	seconds,
long	places)
{
	long	hours, minutes, sec;
	int		addMinus;
	char	fmt[256];
	char	sfrac[256];

	places = (places>10) ? 10 : places;	/* limit places to range [0,10] */
	places = (places<0) ? 0 : places;
	addMinus = (seconds<0);
	seconds = fabs(seconds);
	sec = seconds;

	sprintf(fmt,"%%.%ldf",places);
	sprintf(sfrac,fmt,fmod(seconds,1.));

	hours = sec/3600;				/* number of hours */
	sec = sec%3600;

	minutes = sec/60;				/* number of minutes */
	sec = sec%60;					/* number of seconds */

	sprintf(str,"%02ld:%02ld:%02ld%s",hours,minutes,sec,&(sfrac[1]));   /* remove the leading zero from sfrac */
	if (addMinus) {					/* str = "-"+str */
		strcpy(sfrac,str);
		str[0] = '-';
		strcpy(&(str[1]),sfrac);
	}
	return str;
}
