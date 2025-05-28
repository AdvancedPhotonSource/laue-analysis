/**********************************************************

	Data Structure and Routines for peak fiting
       

/**********************************************************/

#include "fitToFunction.h"

/*******************************************************************/

static int expb_f_1D (const gsl_vector * x, void *params, gsl_vector * f);
static int expb_df_1D (const gsl_vector * x, void *params, gsl_matrix * J);
static int expb_fdf_1D (const gsl_vector * x, void *params, gsl_vector * f, gsl_matrix * J);
static int fitLorentz_1D(double *a, void *params, double *results);
static int expb_f_2D (const gsl_vector * x, void *params, gsl_vector * f);
static int expb_df_2D (const gsl_vector * x, void *params, gsl_matrix * J);
static int expb_fdf_2D (const gsl_vector * x, void *params, gsl_vector * f, gsl_matrix * J);
static int fitLorentz_2D(double *a, void *params, double *results);
static int Lorentz2DFit(double *a, Grid *image, double *a_fit);

/*******************************************************************/
/*
input: 
 a, the list of paramters of a function model (Lorentzian) to be decide after fitting
 params, the observed data valus to be fitted

output:
 f, the fitting function Yi-y[i], where Yi=a0*EZ + a3, y[i] is observed data
    (The funciton model here is for Lorentzian fit) 
 */
static int expb_f_1D (const gsl_vector * a, void *params, 
	    gsl_vector * f){
  size_t n = ((ObservedValues *)params)->n;
  double *y = ((ObservedValues *)params)->y;

  if(a->size != 4){
    printf("Error, wrong numParameters in expb_f_1D!\n");
    exit(1);
  }

  double a0 = gsl_vector_get (a, 0);
  double a1 = gsl_vector_get (a, 1);
  double a2 = gsl_vector_get (a, 2);
  double a3= gsl_vector_get (a,3);
  double Z, EZ, Yi;


  a2= (a2==0.)?0.0001:a2;

  size_t i;

  for (i = 0; i < n; i++)
    {
      /* Model Yi  */
      double x = i;
      Z =(x-a1)/a2;
      EZ =1/((Z*Z)+1.);

      Yi = a0*EZ + a3; //only consider a->size=4
      
      gsl_vector_set (f, i, (Yi - y[i]));
    }

  return GSL_SUCCESS;

}
/*******************************************************************/
/*******************************************************************
input: 
 a, the list of paramters of a function model (Lorentzian) to be decide after fitting
 params, the observed data valus to be fitted

output:
 J, the Jacobian matrix J is the derivative of the function f with respect to the
    four parameters in a  
 *******************************************************************/
static int expb_df_1D (const gsl_vector * a, void *params, 
	     gsl_matrix * J){

  size_t n = ((ObservedValues *)params)->n;

  if(a->size != 4){
    printf("Error, wrong numParameters in expb_df_1D!\n");
    exit(1);
  }

  double a0 = gsl_vector_get (a, 0);
  double a1 = gsl_vector_get (a, 1);
  double a2 = gsl_vector_get (a, 2);
//  double a3= gsl_vector_get (a,3);
//  double Yi;
  double Z, EZ;


  a2= (a2==0.)?0.0001:a2;

  size_t i;

  for (i = 0; i < n; i++)
    {
      
      double x = i;
      Z =(x-a1)/a2;
      EZ =1/((Z*Z)+1.);

      gsl_matrix_set(J, i, 0,EZ );     //df/da0
      gsl_matrix_set(J, i, 1, a0*2*Z*(EZ*EZ)/a2 );    //df/da1
      gsl_matrix_set(J, i, 2, Z*gsl_matrix_get(J,i,1) );    //df/da2
      gsl_matrix_set(J, i, 3,  1);    //df/da3
    
    }

  return GSL_SUCCESS;
}

static int expb_fdf_1D (const gsl_vector * a, void *params,
	      gsl_vector * f, gsl_matrix * J){
  expb_f_1D (a, params, f);
  expb_df_1D (a, params, J);

  return GSL_SUCCESS;

}
/*******************************************************************/
/*******************************************************************
input: 
 initA, initial paramter list (total 4 elements) 
 params, the observed data to be fitted
output:
 a, the list with fitted parameters 
 *******************************************************************/
static int fitLorentz_1D(double *initA, void *params, double *results){
  
  gsl_vector_view av=gsl_vector_view_array(initA,4);

  gsl_vector *a=&av.vector;
  
  const gsl_multifit_fdfsolver_type *T;
  gsl_multifit_fdfsolver *s;

  int status;
//  size_t i;
  size_t iter = 0;

  const size_t n = ((ObservedValues *)params)->n;
  const size_t p = a->size;

  //  gsl_matrix *covar = gsl_matrix_alloc (p, p);

  
  
  gsl_multifit_function_fdf f;

  
  f.f = &expb_f_1D;
  f.df = &expb_df_1D;
  f.fdf = &expb_fdf_1D;
  f.n = n;
  f.p = p;
  f.params = params;


  T = gsl_multifit_fdfsolver_lmsder;
  s = gsl_multifit_fdfsolver_alloc (T, n, p);
  gsl_multifit_fdfsolver_set (s, &f, a);


  do
    {
      iter++;
      status = gsl_multifit_fdfsolver_iterate (s);

      // printf ("status = %s\n", gsl_strerror (status));

    
      if (status)
        break;

      status = gsl_multifit_test_delta (s->dx, s->x,
                                        1e-6, 1e-6);//1e-3,1e-3
    }while (status == GSL_CONTINUE && iter < 100);//20);
  // printf("1D iter=%d\n",iter);
  //  gsl_multifit_covar (s->J, 0.0, covar);

  

#define FIT(i) gsl_vector_get(s->x, i)
  //#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))
  /*
  printf("a0      = %.5f +/- %.5f\n", FIT(0), ERR(0));
  printf("a1 = %.5f +/- %.5f\n", FIT(1), ERR(1));
  printf("a2     = %.5f +/- %.5f\n", FIT(2), ERR(2));
  printf("a3     = %.5f +/- %.5f\n", FIT(3), ERR(3));
  */
  
  results[0]=FIT(0);
  results[1]=FIT(1);
  results[2]=FIT(2);
  results[3]=FIT(3);
  
  gsl_multifit_fdfsolver_free (s);


  return GSL_SUCCESS;
}

/*******************************************************************/
static int expb_f_2D (const gsl_vector * a, void *image, 
	       gsl_vector * f){
  int nx = ((Grid*)image)->width;
  int ny = ((Grid*)image)->height;

  int n=nx*ny;

  double *y = ((Grid *)image)->values;

  if(a->size < 6 || a->size >7){
    printf("Error, wrong numParameters in expb_f_2D size=%lu !\n",a->size);
    exit(1);
  }
  double xp,yp,u,s,c,Yi;
  
  double a0 = gsl_vector_get (a, 0);
  double a1 = gsl_vector_get (a, 1);
  double a2 = gsl_vector_get (a, 2);
  double a3= gsl_vector_get (a,3);
  double a4= gsl_vector_get (a,4);
  double a5= gsl_vector_get (a,5);
  
  
  if(a->size==7){
    double a6=gsl_vector_get(a,6);
    s = sin(a6);
    c = cos(a6);
  }
  else {
    s=0.;
    c=1.;
  }
  //  printf("\n**xp**:\n");
	int i;
  for(i=0;i<n;i++){
    //   if(i%nx==0) printf("\n");
    xp =(i%ny - a4)* c/a2 - (i/nx - a5) * s /a2;
    yp =(i%ny - a4) *s/a3 +  (i/nx - a5) * c/a3;
    u = 1./((xp*xp) + (yp*yp) + 1.);
    Yi = a0 + a1 * u;
    //    printf("%5.2f ",xp);
    
    gsl_vector_set(f, i, (Yi - y[i]));
    
  }


  return GSL_SUCCESS;

}

/*******************************************************************/

static int expb_df_2D (const gsl_vector * a, void *image, 
		gsl_matrix * J){

  int nx = ((Grid*)image)->width;
  int ny = ((Grid*)image)->height;

  int n=nx*ny;

//  double *y = ((Grid *)image)->values;
  /*
  printf("Image values:\n");
  for(int i=0;i<n;i++){
    printf("%d ",(int)y[i]);
    if(i%nx==0)
      printf("\n");
  }
  */
  if(a->size < 6 || a->size >7){
    printf("Error, wrong numParameters in expb_f_2D!\n");
    exit(1);
  }
  double xp,yp,u,s,c;
  
//  double a0 = gsl_vector_get (a, 0);
  double a1 = gsl_vector_get (a, 1);
  double a2 = gsl_vector_get (a, 2);
  double a3= gsl_vector_get (a,3);
  double a4= gsl_vector_get (a,4);
  double a5= gsl_vector_get (a,5);
  
  
  if(a->size==7){
    double a6=gsl_vector_get(a,6);
    s = sin(a6);
    c = cos(a6);
  }
  else {
    s=0.;
    c=1.;
  }
	int i;
  for(i=0;i<n;i++){
    xp =(i%ny - a4)* c/a2 - (i/nx - a5) * s /a2;
    yp =(i%ny - a4) *s/a3 +  (i/nx - a5) * c/a3;
    u = 1./((xp*xp) + (yp*yp) + 1.);

    gsl_matrix_set(J,i,0,1.);
    gsl_matrix_set(J,i,1,u);
    u = 2.*a1*(u*u);
    gsl_matrix_set(J,i,2,u*(xp*xp)/a2);
    gsl_matrix_set(J,i,3,u*(yp*yp)/a3);
    gsl_matrix_set(J,i,4,u*(c/a2*xp + s/a3*yp));
    gsl_matrix_set(J,i,5,u*(-s/a2*xp + c/a3*yp));
    if(a->size==7)
      gsl_matrix_set(J,i,6,u*xp*yp*(a3/a2-a2/a3));
      
  }


  return GSL_SUCCESS;

}


/*******************************************************************/
static int expb_fdf_2D (const gsl_vector * a, void *params,
		 gsl_vector * f, gsl_matrix * J){
  expb_f_2D (a, params, f);
  expb_df_2D (a, params, J);

  return GSL_SUCCESS;

}


/*******************************************************************/
static int fitLorentz_2D(double *init_a, void *image, double *a_fit){
 
  gsl_vector_view av=gsl_vector_view_array(init_a,7);

  gsl_vector *a=&av.vector;
  
  const gsl_multifit_fdfsolver_type *T;
  gsl_multifit_fdfsolver *s;

  int status;
//  size_t i;
  size_t iter = 0;

  const size_t n = (size_t)(((Grid *)image)->width*((Grid *)image)->height);
  const size_t p = a->size;

  //  gsl_matrix *covar = gsl_matrix_alloc (p, p);

  
  
  gsl_multifit_function_fdf f;
  //printf("array size: %d %d\n",sizeof(init_a),sizeof(double));
  
  f.f = &expb_f_2D;
  f.df = &expb_df_2D;
  f.fdf = &expb_fdf_2D;
  f.n = n;
  f.p = p;
  f.params = image;


  T = gsl_multifit_fdfsolver_lmsder;
  s = gsl_multifit_fdfsolver_alloc (T, n, p);
  gsl_multifit_fdfsolver_set (s, &f, a);


  do
    {
      iter++;
      status = gsl_multifit_fdfsolver_iterate (s);

      // printf ("status = %s\n", gsl_strerror (status));

    
      if (status)
        break;

      status = gsl_multifit_test_delta (s->dx, s->x,
                                        1e-6, 1e-6);
    }while (status == GSL_CONTINUE && iter < 100);//50);
  // printf("2D iter=%d\n",iter);
    //  gsl_multifit_covar (s->J, 0.0, covar);

  

#define FIT(i) gsl_vector_get(s->x, i)
//#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))
  /*
  printf("a0      = %.5f +/- %.5f\n", FIT(0), ERR(0));
  printf("a1 = %.5f +/- %.5f\n", FIT(1), ERR(1));
  printf("a2     = %.5f +/- %.5f\n", FIT(2), ERR(2));
  printf("a3     = %.5f +/- %.5f\n", FIT(3), ERR(3));
*/
	int j;
for(j=0;j<p;j++)
  a_fit[j]=FIT(j);
  
  
  gsl_multifit_fdfsolver_free (s);


  return GSL_SUCCESS;
}


/*******************************************************************/
/*******************************************************************
input:
 image,  ROI of a peak to be fitted 
 a, initial list of paramters to be fit, 7*sizeof(double)
 a_fit,an empty arry for holding the fitted results of a

output:
 a_fit, the parameter list after 2D Lorentz fit
 *******************************************************************/  


static int Lorentz2DFit(double *a, Grid *image, double *a_fit){

  
  int nx=image->width;
  int ny=image->height;
//  int n=nx*ny;
  
  //find maxima location (ix,iy) in image 
  Grid *image_roi=grid_new_copy_region(image,0,0,nx-1,ny-1);
  grid_smooth_boxcar(image_roi,1);
  Point* center=centroid_2(image_roi,0,0);
  int ix=(int)(center->x);
  int iy=(int)(center->y);
  
  double *ax=malloc(4*sizeof(double));
  double *ay=malloc(4*sizeof(double));
  ax[0]=grid_get_value(image,ix,iy)-a[0];
  ax[1]=center->x;
  ax[2]=a[2];
  ax[3]=a[0];
  ay[0]=grid_get_value(image,ix,iy)-a[0];
  ay[1]=center->y;
  ay[2]=a[3];
  ay[3]=a[0];

  double *x=malloc(ny*sizeof(double));
  double *y=malloc(nx*sizeof(double));


	int i;
  for(i=0;i<ny;i++)
    x[i]=grid_get_value(image,i,iy);

  for(i=0;i<nx;i++){
    y[i]=grid_get_value(image,ix,i);
  }


  ObservedValues paramsX = {ny,x};
  ObservedValues paramsY = {nx,y};

  double *ax_fit=malloc(4*sizeof(double));
  double *ay_fit=malloc(4*sizeof(double));
  
  fitLorentz_1D(ax,&paramsX,ax_fit);
  fitLorentz_1D(ay,&paramsY,ay_fit);
   
  


  //adjust parameter list a 
  a[0]=(ax_fit[3] + ay_fit[3])/2.;
  a[1]=sqrt(fabs(ax_fit[0]*ay_fit[0]));
  a[2]=fabs(ax_fit[2]);
  a[3]=fabs(ay_fit[2]);
  a[4]=ax_fit[1];
  a[5]=ay_fit[1];
 
   
  fitLorentz_2D(a, (void *)image, a_fit);
  
  a_fit[6]= mod(a_fit[6], M_PI);

  free(x);
  free(y);
    

  return 0;


}
   
/*******************************************************************/
/*******************************************************************
This is the function to be called in peaksearch.c

input:
 image, the ROI image of a peak to be fitted 

all the other paramters are input & output, after function return they
are all reset with fitted values
changed Aug 2009 by JZT to reject data values of NAN.

 *******************************************************************/
void fitToFunctionLorentz(Grid *image, double *fitx, double *fity, double *background, double *intens,
		  double *widthx, double *widthy, double *tilt,double *chisq){
	double *a=malloc(7*sizeof(double));
	double *a_fit=malloc(7*sizeof(double));
	a[0]=*background;	
	a[1]=*intens;
	a[2]=*widthx;
	a[3]=*widthy;
	a[4]=*fitx;
	a[5]=*fity;
	a[6]=0.;

	Lorentz2DFit(a,image,a_fit); //interface for doing the 2D fit

	int nx = image->width;
	int ny = image->height;
	int n=nx*ny;
	double s=sin(a_fit[6]), c=cos(a_fit[6]);
	double xp, yp,u,F,chi, datai, *data=image->values;
	double sumChi=0.,sumData=0.;
 
	int i;
	for(i=0;i<n;i++){
		datai = data[i];
		if (datai!=datai) continue;		/* skip NaNs in the data */
		xp =(i%ny - a_fit[4]) * c/a_fit[2] - (i/nx - a_fit[5]) * s/a_fit[2];
		yp =(i%ny - a_fit[4]) * s/a_fit[3] + (i/nx - a_fit[5]) * c/a_fit[3];
		u = 1./((xp*xp) + (yp*yp) + 1.);
		F = a_fit[0] + a_fit[1] * u;
		chi = (F-datai) * (F-datai);
		sumChi += chi;
		sumData += (datai*datai);
	}

  *fitx=a_fit[4];
  *fity=a_fit[5];
  *background=a_fit[0];
  *intens=a_fit[1];
  *widthx=(a_fit[2]);
  *widthy=(a_fit[3]);
  *tilt=a_fit[6]*180./M_PI;
  *chisq=sumChi/sumData;
  free(a);
  free(a_fit);
}
     
  
  


  
  
  
  









