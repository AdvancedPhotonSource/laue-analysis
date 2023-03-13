/*
 *  microHDF5.c
 *  hdfTest
 *
 *  Created by Jon Tischler on 1/7/09.
 *  Copyright 2009 ORNL. All rights reserved.
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "microHDF5.h"


/*
#ifndef isnan
#define isnan(A) (!( (A)==(A) ))
#endif
*/
/* #define VERBOSE */

#ifdef __linux__
#define repackPATH "/clhome/aps_tools/hdf5-1.8.2/hdf5/bin/h5repack"
#else
#define repackPATH "/opt/local/bin/h5repack"
#endif

#define STARTX 0			/* default values */
#define ENDX 2047
#define GROUPX 1
#define STARTY 0
#define ENDY 2047
#define GROUPY 1
#define XDIMDET 2048
#define YDIMDET 2048

#define CHECK_FREE(A)   { if(A) free(A); (A)=NULL; }


// #define RECONSTRUCT_BACKWARDS


/* Local Routines */
size_t HDF5ReadDoubleVector(hid_t file_id, const char *dataName, double **vbuf);
int get1HDF5data_float(hid_t file_id, char *dataName, double *value);
int get1HDF5data_int(hid_t file_id, char *dataName, long *value);
int get1HDF5data_string(hid_t file_id, char *dataName, char *value, long N);
herr_t get1HDF5attr_float(hid_t file_id, char *groupName, char *attrName, double *value);
herr_t get1HDF5attr_int(hid_t file_id, char *groupName, char *attrName, long *value);
herr_t get1HDF5attr_string(hid_t file_id, char *groupName, char *attrName, char *value);

int get1HDF5data_tagVal(hid_t file_id, char *groupName, char *dataName, char *tagName, char result1[256]);
int get1HDF5attr_tagVal(hid_t file_id, char *groupName, char *attrName, char *tagName, char result1[256]);
herr_t groupExists(hid_t file_id, char *groupName);



/*******************************************************************************************
*********************************  External HDF5 Routines  *********************************
********************************************************************************************/

/* write a HDF5 file data part ROI.  To get header information, first call HDF5ReadHeader */
/* the image is in vbuf, it is ordered with x moving fastest */
/* the type of data in vbuf comes from the header (itype), and the type of data written is already established in the file */
int HDF5WriteROI(						/* this overwrites an ROI in a dataset.  Note, the data must already exist (and be bigger than the ROI) */
const char *fileName,					/* full path name to file */
const char *dataName,					/* full path name to data, e.g. "entry1/data/data" */
void	*vbuf,							/* pointer to existing data, contains what I will write */
size_t	xlo,							/* writes region [xlo,ylo] to [xhi,yhi] */
size_t	xhi,							/* if xhi or yhi are <0, then it uses xdim or ydim */
size_t	ylo,							/* note, max yhi is ydim-1, and ditto for x */
size_t	yhi,
hid_t	dataType,						/* hdf5 data type (HDF5, not WinView) for the numbers in vbuf, the type in the file was set when dataName was created */
struct HDF5_Header *head)				/* HDF5 header information (header must be valid and have correct values!) */
{
	herr_t	i, err=0;					/* TRUE=ERROR,  FALSE=OK on return */
	hid_t	file_id;
	hid_t	data_id=0;					/* location id of the data in file */
	hid_t	dataspace=0;
	hid_t	memspace=0; 
	hsize_t	dims_out[5];				/* dataset dimensions, 5 is larger than necessary, we should only need 2 */
	int		rank=0;

	hsize_t	dimsm[2]={0,0};				/* memory space dimensions */
	hsize_t	count[2]={0,0};				/* size of the hyperslab in the file */
	hsize_t	offset[2]={0,0};			/* hyperslab offset in the file */
	hsize_t	count_out[2]={0,0};			/* size of the hyperslab in memory */
	hsize_t	offset_out[2]={0,0};		/* hyperslab offset in memory */

	size_t	xdim;						/* x,y dimensions of image in file */
	size_t	ydim;
	size_t	pixels;						/* number of pixels, = xdim*ydim */
	size_t	ilen;						/* number of bytes per pixel */
	size_t	nx;							/* number of points in x,  xhi - xlo + 1 */
	size_t	ny;							/* number of points in y,  yhi - ylo + 1 */

	if (!(vbuf)) return -1;								/* vbuf must exist and be full of data to write */
	if (strlen(fileName)<1 || strlen(dataName)<1) return -1;	/* need valid file and data name */
	if (!head) return -1;								/* header must be valid */
	xdim = head->xdim;
	ydim = head->ydim;
	ilen = head->isize;

	if (!ilen) return 3;								/* no word length in header */

	/* this section used if xhi or yhi <=0 */
	xhi = (xhi<0 || xhi>(xdim-1)) ? xdim-1 : xhi;		/* xhi is now actual to use */
	yhi = (yhi<0 || yhi>(ydim-1)) ? ydim-1 : yhi;
	xlo = (xlo>xhi) ? xhi : xlo;
	ylo = (ylo>yhi) ? yhi : ylo;
	xlo = (xlo<0) ? 0 : xlo;
	ylo = (ylo<0) ? 0 : ylo;
	nx = xhi - xlo + 1;									/* number of pixels in ROI along X and Y */
	ny = yhi - ylo + 1;
	if (xlo<0 || ylo<0 || xlo>xhi || ylo>yhi || xhi>xdim-1 || yhi>ydim-1) return 2;	/* no image to write, invalid range */
	pixels = xdim * ydim;								/* total number of pixels in the image */

	if ((file_id=H5Fopen(fileName,H5F_ACC_RDWR,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5WriteROI(), cannot open the file '%s'\n",fileName); ERROR_PATH(file_id) }
	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5WriteROI(), the data '%s' does not exist\n",dataName); ERROR_PATH(data_id) }
	if ((dataspace=H5Dget_space(data_id))<=0) ERROR_PATH(-1)	/* dataspace identifier */

	/* This check is not necessary,  check for existance of data */
	if ((rank=H5Sget_simple_extent_dims(dataspace,dims_out,NULL))>=0) err = (rank<0 ? -1 : 0);
	if (err) ERROR_PATH(err)
	if (rank != 2) ERROR_PATH(rank)						/* only understand rank==2 data here */

	#ifdef DEBUG
		printf("    image buffer = vbuf = %p,   xdim = %lu, ydim = %lu\n",vbuf,xdim,ydim);
		printf("    [xlo,xhi]=[%ld,%ld] nx=%ld,  [ylo,yh]=[%ld,%ld] ny=%ld,  write %ld pixels\n",xlo,xhi,nx,ylo,yhi,ny,pixels);
	#endif

	/* specify definition of memory space containing the image (i.e. vbuf) */
	dimsm[0] = nx;		dimsm[1] = ny;					/* memory space dimensions */
	if ((memspace=H5Screate_simple(2,dimsm,NULL))<0) ERROR_PATH(memspace)	/* Define the memory dataspace */

	offset[0]=xlo;		offset[1]=ylo;					/* define which part of the data in the file to write */
	count[0]=nx;		count[1]=ny;
	count_out[0]=nx;	count_out[1]=ny;

	if ((i=H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL))<0)
		{ fprintf(stderr,"ERROR -- HDF5WriteROI() in H5Sselect_hyperslab(memspace)=%d\n",err); ERROR_PATH(i) }
	if ((i=H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL))<0)	
		{ fprintf(stderr,"ERROR -- HDF5WriteROI() in H5Sselect_hyperslab(dataspace)=%d\n",err); ERROR_PATH(i) }
	if ((i=H5Dwrite(data_id,dataType,memspace,dataspace,H5P_DEFAULT,vbuf))<0)
		{ fprintf(stderr,"ERROR -- HDF5WriteROI() in H5Dwrite(hyperslab)=%d\n",err); ERROR_PATH(i) }

	error_path:
	if (memspace>0) H5Sclose(memspace);
	if (dataspace>0) H5Sclose(dataspace);
	if (data_id>0) H5Dclose(data_id);
	if (file_id>0) H5Fclose(file_id);
	return err;
}


/* read a HDF5 file data part.  To get header information, first call HDF5ReadHeader */
/* the image is in vbuf, it is ordered with x moving fastest */
/* CAUTION,  It allocates space for image in vbuf only if vbuf is NULL, so pass it with a null value, and remember to free it later yourself */
/* if vbuf is not NULL, then there better be enough room for the image */
int HDF5ReadROI(
const char	*fileName,					/* full path name to file */
const char	*dataName,					/* full path name to data, e.g. "entry1/data/data" */
void	**vbuf,							/* pointer to image, if not NULL, space is allocated here, otherwise assume vbuf is big enough, and it better be too! */
size_t	xlo,							/* reads region [xlo,ylo] to [xhi,yhi] */
size_t	xhi,							/* if xhi or yhi are <0, then it uses xdim or ydim */
size_t	ylo,							/* note, max yhi is ydim-1, and ditto for x */
size_t	yhi,
struct HDF5_Header *head)				/* HDF5 header information (header must be valid!) */
{
	herr_t	i, err=0;
	hid_t	file_id;
	hid_t	data_id=0;					/* location id of the data in file */
	hid_t	dataspace=0;
	hid_t	memspace=0; 
	hsize_t	dims_out[5];				/* dataset dimensions, 5 is larger than necessary, we should only need 2 */
	int		rank=0;

	hsize_t	dimsm[2]={0,0};				/* memory space dimensions */
	hsize_t	count[2]={0,0};				/* size of the hyperslab in the file */
	hsize_t	offset[2]={0,0};			/* hyperslab offset in the file */
	hsize_t	count_out[2]={0,0};			/* size of the hyperslab in memory */
	hsize_t	offset_out[2]={0,0};		/* hyperslab offset in memory */

	size_t	xdim;						/* x,y dimensions of image in file */
	size_t	ydim;
	size_t	pixels;						/* number of pixels, = xdim*ydim */
	size_t	ilen;						/* number of bytes per pixel */
	size_t	nx;							/* number of points in x,  xhi - xlo + 1 */
	size_t	ny;							/* number of points in y,  yhi - ylo + 1 */

	if (strlen(fileName)<1 || strlen(dataName)<1) return -1;	/* need valid file and data name */
	if (!head) return -1;								/* header must be valid */
	xdim = head->xdim;
	ydim = head->ydim;
	ilen = head->isize;

	if (!ilen) return 3;								/* no word length in header */

	/* this section used if xhi or yhi <=0 */
	xhi = (xhi<0 || xhi>(xdim-1)) ? xdim-1 : xhi;		/* xhi is now actual to use */
	yhi = (yhi<0 || yhi>(ydim-1)) ? ydim-1 : yhi;
	xlo = (xlo>xhi) ? xhi : xlo;
	ylo = (ylo>yhi) ? yhi : ylo;
	xlo = (xlo<0) ? 0 : xlo;
	ylo = (ylo<0) ? 0 : ylo;
	nx = xhi - xlo + 1;									/* number of pixels in ROI along X and Y */
	ny = yhi - ylo + 1;
	if (xlo<0 || ylo<0 || xlo>xhi || ylo>yhi || xhi>xdim-1 || yhi>ydim-1) return 2;	/* no image to read, invalid range */
	pixels = xdim * ydim;								/* total number of pixels in the image */


	if ((file_id=H5Fopen(fileName,H5F_ACC_RDONLY,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5ReadROI(), cannot open the file '%s'\n",fileName); ERROR_PATH(file_id) }
	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5ReadROI(), the data '%s' does not exist\n",dataName); ERROR_PATH(data_id) }
	if ((dataspace=H5Dget_space(data_id))<=0) ERROR_PATH(-1)	/* dataspace identifier */

	/* check for existance of data */
	if ((rank=H5Sget_simple_extent_dims(dataspace,dims_out,NULL))>=0) err = (rank<0 ? -1 : 0);
	if (err) ERROR_PATH(err)
	if (rank != 2) ERROR_PATH(rank)						/* only understand rank==2 data here */

	/* allocate space for the image */
	#ifdef DEBUG
		printf("in HDF5ReadROI:\n");
		printf("    about to try to allocate image space of %lu Kbytes\n",pixels*ilen/1024);
	#endif
	if (!(*vbuf)) *vbuf = (char*)calloc(pixels,ilen);	/* allocate space here if vbuf is NULL, otherwise it better be big enough */
	if (!(*vbuf)) return 5;								/* allocation error */
	#ifdef DEBUG
		printf("    image buffer = vbuf = %p,   xdim = %lu, ydim = %lu\n",vbuf,xdim,ydim);
		printf("    [xlo,xhi]=[%ld,%ld] nx=%ld,  [ylo,yh]=[%ld,%ld] ny=%ld,  read %ld pixels\n",xlo,xhi,nx,ylo,yhi,ny,pixels);
	#endif

	dimsm[0] = nx;		dimsm[1] = ny;					/* memory space dimensions */
	if ((memspace=H5Screate_simple(2,dimsm,NULL))<0) ERROR_PATH(memspace)	/* Define the memory dataspace */

	offset[0]=xlo;		offset[1]=ylo;					/* define which part of the data in the file to read */
	count[0]=nx;		count[1]=ny;
	count_out[0]=nx;	count_out[1]=ny;

	if ((i=H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL))<0)	{ fprintf(stderr,"error in H5Sselect_hyperslab(memspace)=%d\n",err); ERROR_PATH(i) }
	if ((i=H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL))<0)			{ fprintf(stderr,"error in H5Sselect_hyperslab(dataspace)=%d\n",err); ERROR_PATH(i) }
	if ((i=H5Dread(data_id,H5T_NATIVE_UINT16,memspace,dataspace,H5P_DEFAULT,*vbuf))<0)		{ fprintf(stderr,"error in H5Dread(hyperslab)=%d\n",err); ERROR_PATH(i) }

	error_path:
	if (memspace>0) H5Sclose(memspace);
	if (dataspace>0) H5Sclose(dataspace);
	if (data_id>0) H5Dclose(data_id);
	if (file_id>0) H5Fclose(file_id);
	return err;
}


/* read an HDF5 file data part.  To get header information, first call HDF5ReadHeader */
/* the image is in vbuf, it is ordered with x moving fastest, This image is "double" */
/* CAUTION,  It allocates space for image in vbuf only if vbuf is NULL, so pass it with a null value, and remember to free it later yourself */
/* if vbuf is not NULL, then there better be enough room for the image */
int HDF5ReadROIdouble(
const char	*fileName,					/* full path name to file */
const char	*dataName,					/* full path name to data, e.g. "entry1/data/data" */
double	**vbuf,							/* pointer to image, if not NULL, space is allocated here, otherwise assume vbuf is big enough, and it better be too! */
size_t	xlo,							/* reads region [xlo,ylo] to [xhi,yhi] */
size_t	xhi,							/* if xhi or yhi are <0, then it uses xdim or ydim */
size_t	ylo,							/* note, max yhi is ydim-1, and ditto for x */
size_t	yhi,
struct HDF5_Header *head)				/* HDF5 header information (header must be valid!) */
{
	herr_t	i, err=0;
	hid_t	file_id;
	hid_t	data_id=0;					/* location id of the data in file */
	hid_t	dataspace=0;
	hid_t	memspace=0; 
	hsize_t	dims_out[5];				/* dataset dimensions, 5 is larger than necessary, we should only need 2 */
	int		rank=0;

	hsize_t	dimsm[2]={0,0};				/* memory space dimensions */
	hsize_t	count[2]={0,0};				/* size of the hyperslab in the file */
	hsize_t	offset[2]={0,0};			/* hyperslab offset in the file */
	hsize_t	count_out[2]={0,0};			/* size of the hyperslab in memory */
	hsize_t	offset_out[2]={0,0};		/* hyperslab offset in memory */

	size_t	xdim;						/* x,y dimensions of image in file */
	size_t	ydim;
	size_t	pixels;						/* number of pixels, = xdim*ydim */
	size_t	ilen;						/* number of bytes per pixel in output array, we are using double here */
	size_t	nx;							/* number of points in x,  xhi - xlo + 1 */
	size_t	ny;							/* number of points in y,  yhi - ylo + 1 */

	if (strlen(fileName)<1 || strlen(dataName)<1) return -1;	/* need valid file and data name */
	if (!head) return -1;								/* header must be valid */
	xdim = head->xdim;
	ydim = head->ydim;
	ilen = sizeof(double);				/*	used to be this:	ilen = head->isize; */

	if (!ilen) return 3;								/* no word length in header */

	/* this section used if xhi or yhi <=0 */
	xhi = (xhi<0 || xhi>(xdim-1)) ? xdim-1 : xhi;		/* xhi is now actual to use */
	yhi = (yhi<0 || yhi>(ydim-1)) ? ydim-1 : yhi;
	xlo = (xlo>xhi) ? xhi : xlo;
	ylo = (ylo>yhi) ? yhi : ylo;
	xlo = (xlo<0) ? 0 : xlo;
	ylo = (ylo<0) ? 0 : ylo;
	nx = xhi - xlo + 1;									/* number of pixels in ROI along X and Y */
	ny = yhi - ylo + 1;
	if (xlo<0 || ylo<0 || xlo>xhi || ylo>yhi || xhi>xdim-1 || yhi>ydim-1) return 2;	/* no image to read, invalid range */
	pixels = xdim * ydim;								/* total number of pixels in the image */

	if ((file_id=H5Fopen(fileName,H5F_ACC_RDONLY,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5ReadROI(), cannot open the file '%s'\n",fileName); ERROR_PATH(file_id) }
	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5ReadROI(), the data '%s' does not exist\n",dataName); ERROR_PATH(data_id) }
	if ((dataspace=H5Dget_space(data_id))<=0) ERROR_PATH(-1)	/* dataspace identifier */

	/* check for existance of data */
	if ((rank=H5Sget_simple_extent_dims(dataspace,dims_out,NULL))>=0) err = (rank<0 ? -1 : 0);
	if (err) ERROR_PATH(err)
	if (rank != 2) ERROR_PATH(rank)						/* only understand rank==2 data here */

	/* allocate space for the image */
	#ifdef DEBUG
		printf("in HDF5ReadROI:\n");
		printf("    about to try to allocate image space of %ld Kbytes\n",pixels*ilen/1024);
	#endif
	if (!(*vbuf)) *vbuf = (double*)calloc(pixels,ilen);	/* allocate space here if vbuf is NULL, otherwise it better be big enough */
	if (!(*vbuf)) return 5;								/* allocation error */
	#ifdef DEBUG
		printf("    image buffer = vbuf = %p,   xdim = %lu, ydim = %lu\n",vbuf,xdim,ydim);
		printf("    [xlo,xhi]=[%ld,%ld] nx=%ld,  [ylo,yh]=[%ld,%ld] ny=%ld,  read %ld pixels\n",xlo,xhi,nx,ylo,yhi,ny,pixels);
	#endif

#ifdef RECONSTRUCT_BACKWARDS
	dimsm[0] = nx;		dimsm[1] = ny;					/* memory space dimensions */
#else
/* HDF stores transpose of what I expect */
	dimsm[1] = nx;		dimsm[0] = ny;					/* memory space dimensions */
#endif
	if ((memspace=H5Screate_simple(2,dimsm,NULL))<0) ERROR_PATH(memspace)	/* Define the memory space */

#ifdef RECONSTRUCT_BACKWARDS
	offset[0]=xlo;		offset[1]=ylo;					/* define which part of the data in the file to read */
	count[0]=nx;		count[1]=ny;
	count_out[0]=nx;	count_out[1]=ny;
#else
	offset[1]=xlo;		offset[0]=ylo;					/* define which part of the data in the file to read */
	count[1]=nx;		count[0]=ny;					/*	size of region to read */
	count_out[1]=nx;	count_out[0]=ny;				/*	size of output region */
#endif

	if ((i=H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL))<0)	{ fprintf(stderr,"error in H5Sselect_hyperslab(memspace)=%d\n",i); ERROR_PATH(i) }
	if ((i=H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL))<0)			{ fprintf(stderr,"error in H5Sselect_hyperslab(dataspace)=%d\n",i); ERROR_PATH(i) }
	if ((i=H5Dread(data_id,H5T_IEEE_F64LE,memspace,dataspace,H5P_DEFAULT,*vbuf))<0)			{ fprintf(stderr,"error in H5Dread(hyperslab)=%d\n",i); ERROR_PATH(i) }

	error_path:
	if (memspace>0) H5Sclose(memspace);
	if (dataspace>0) H5Sclose(dataspace);
	if (data_id>0) H5Dclose(data_id);
	if (file_id>0) H5Fclose(file_id);
	return err;
}



#ifdef MULTI_IMAGE_FILE
/* read an HDF5 file data part.  To get header information, first call HDF5ReadHeader */
/* the image is in vbuf, it is ordered with x moving fastest, This image is "double" */
/* CAUTION,  It allocates space for image in vbuf only if vbuf is NULL, so pass it with a null value, and remember to free it later yourself */
/* if vbuf is not NULL, then there better be enough room for the image */
int HDF5ReadROIdoubleSlice(
const char	*fileName,					/* full path name to file */
const char	*dataName,					/* full path name to data, e.g. "entry1/data/data" */
double	**vbuf,							/* pointer to image, if not NULL, space is allocated here, otherwise assume vbuf is big enough, and it better be too! */
long	xlo,							/* reads region [xlo,ylo] to [xhi,yhi] */
long	xhi,							/* if xhi or yhi are <0, then it uses xdim or ydim */
long	ylo,							/* note, max yhi is ydim-1, and ditto for x */
long	yhi,
struct HDF5_Header *head,				/* HDF5 header information (header must be valid!) */
size_t	slice)							/* particular image in the stack */
{
	herr_t	i, err=0;
	hid_t	file_id;
	hid_t	data_id=0;					/* location id of the data in file */
	hid_t	dataspace=0;
	hid_t	memspace=0; 
	hsize_t	dims_out[5];				/* dataset dimensions, 5 is larger than necessary, we should only need 2 */
	int		rank=0;

	hsize_t	dimsm[3]={0,0,0};			/* memory space dimensions */
	hsize_t	count[3]={0,0,0};			/* size of the hyperslab in the file, file is 3D */
	hsize_t	offset[3]={0,0,0};			/* hyperslab offset in the file */
	hsize_t	count_out[2]={0,0};			/* size of the hyperslab in memory, memory is 2D */
	hsize_t	offset_out[2]={0,0};		/* hyperslab offset in memory */

	size_t	xdim;						/* x,y dimensions of image in file */
	size_t	ydim;
	size_t	pixels;						/* number of pixels to read, = nx*ny */
	size_t	ilen;						/* number of bytes per pixel in output array, we are using double here */
	size_t	nx;							/* number of points in x,  xhi - xlo + 1 */
	size_t	ny;							/* number of points in y,  yhi - ylo + 1 */

	if (strlen(fileName)<1 || strlen(dataName)<1) return -1;	/* need valid file and data name */
	if (!head) return -1;								/* header must be valid */
	if (slice<0 || slice>=head->Nimages) return 2;		/* slice out of range */
	xdim = head->xdim;
	ydim = head->ydim;
	ilen = sizeof(double);				/*	used to be this:	ilen = head->isize; */
	if (!ilen) return 3;				/* no word length in header */

	/* this section used if xhi or yhi <=0 */
	xhi = (xhi<0 || xhi>((long)xdim-1)) ? (long)xdim-1 : xhi;	/* xhi is now actual to use */
	yhi = (yhi<0 || yhi>((long)ydim-1)) ? (long)ydim-1 : yhi;
	xlo = (xlo>xhi) ? xhi : xlo;
	ylo = (ylo>yhi) ? yhi : ylo;
	xlo = (xlo<0) ? 0 : xlo;
	ylo = (ylo<0) ? 0 : ylo;
	nx = xhi - xlo + 1;									/* number of pixels in ROI along X and Y */
	ny = yhi - ylo + 1;
	if (xlo<0 || ylo<0 || xlo>xhi || ylo>yhi || xhi>(long)xdim-1 || yhi>(long)ydim-1) return 2;	/* no image to read, invalid range */
	// pixels = xdim * ydim;							/* total number of pixels in the image */
	pixels = nx * ny;									/* total number of pixels in the output image */

	if ((file_id=H5Fopen(fileName,H5F_ACC_RDONLY,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5ReadROIdoubleSlice(), cannot open the file '%s'\n",fileName); ERROR_PATH(file_id) }
	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5ReadROIdoubleSlice(), the data '%s' does not exist\n",dataName); ERROR_PATH(data_id) }
	if ((dataspace=H5Dget_space(data_id))<=0) ERROR_PATH(-1)	/* dataspace identifier */

	/* check for existance of data */
	if ((rank=H5Sget_simple_extent_dims(dataspace,dims_out,NULL))>=0) err = (rank<0 ? -1 : 0);
	if (err) ERROR_PATH(err)
	if (rank != 3) ERROR_PATH(rank)						/* only understand rank==3 data here */

	/* allocate space for the image */
	#ifdef DEBUG
		printf("in HDF5ReadROIdoubleSlice:\n");
		printf("    about to try to allocate image space of %ld Kbytes\n",pixels*ilen/1024);
	#endif
	if (!(*vbuf)) *vbuf = (double*)calloc(pixels,ilen);	/* allocate space here if vbuf is NULL, otherwise it better be big enough */
	if (!(*vbuf)) return 5;								/* allocation error */
	#ifdef DEBUG
//		printf("    image buffer = vbuf = %p,   nx = %lu, ny = %lu\n",vbuf,nx,ny);
		printf("    [xlo,xhi]=[%ld,%ld] nx=%lu,  [ylo,yh]=[%ld,%ld] ny=%lu,  read %ld pixels\n",xlo,xhi,nx,ylo,yhi,ny,pixels);
	#endif

#ifdef RECONSTRUCT_BACKWARDS
	dimsm[0] = nx;		dimsm[1] = ny;					/* memory space dimensions */
#else
/* HDF stores transpose of what I expect */
	dimsm[0]=nx;		dimsm[1]=ny;					/* memory space dimensions */
#endif
	if ((memspace=H5Screate_simple(2,dimsm,NULL))<0) ERROR_PATH(memspace)	/* Define the memory space */

#ifdef RECONSTRUCT_BACKWARDS
	offset[1]=xlo;		offset[2]=ylo;					/* define which part of the data in the file to read */
	count[1]=nx;		count[2]=ny;					/*	size of region to read */
	count_out[0]=nx;	count_out[1]=ny;				/*	size of output region (memspace) */
#else
	offset[2]=xlo;		offset[1]=ylo;					/* define which part of the data in the file to read */
	count[2]=nx;		count[1]=ny;					/*	size of region to read */
	count_out[1]=nx;	count_out[0]=ny;				/*	size of output region */
#endif
	offset[0]=slice;									/* a single slice into dataspace */
	count[0] = 1LL;

	if ((i=H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL))<0)	{ fprintf(stderr,"error in H5Sselect_hyperslab(memspace)=%d\n",i); ERROR_PATH(i) }
	if ((i=H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL))<0)			{ fprintf(stderr,"error in H5Sselect_hyperslab(dataspace)=%d\n",i); ERROR_PATH(i) }
	if ((i=H5Dread(data_id,H5T_IEEE_F64LE,memspace,dataspace,H5P_DEFAULT,*vbuf))<0)			{ fprintf(stderr,"error in H5Dread(hyperslab)=%d\n",i); ERROR_PATH(i) }

	error_path:
	if (memspace>0) H5Sclose(memspace);
	if (dataspace>0) H5Sclose(dataspace);
	if (data_id>0) H5Dclose(data_id);
	if (file_id>0) H5Fclose(file_id);
	return err;
}
#endif




/* Create the data space for the dataset. */
/*	e.g.	dims[2]={4,6};	 for rank=2 */
int createNewData(
const char *fileName,						/* name of file to use */
const char *dataName,						/* FULL name of data set, e.g. "entry1/data/data" */
int		rank,								/* rank of new data */
int		*dims,								/* inidvidual dimensions (dims must be of length rank) */
int		dataType)							/* HDF5 data type, e.g. H5T_NATIVE_INT32,  	dataType = getHDFtype(itype); */
{
	hid_t	file_id, dataset_id, dataspace_id;  /* identifiers */
	hid_t	attribute_id;
	hid_t	attr_dataspace_id;
	herr_t	status;
	hsize_t	dimsHDF5[rank];
	int		signal=1;
	int		i;
	for (i=0;i<rank;i++) dimsHDF5[i] = dims[i];

	file_id = H5Fopen(fileName,H5F_ACC_RDWR,H5P_DEFAULT);	/* Open an existing file */
	dataspace_id = H5Screate_simple(rank,dimsHDF5,NULL);	/* create the data space */

	/* Create the dataset. */
	dataset_id = H5Dcreate(file_id,dataName,dataType,dataspace_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

	attr_dataspace_id = H5Screate(H5S_SCALAR);
	attribute_id = H5Acreate(dataset_id,"signal",H5T_STD_I32LE,attr_dataspace_id,H5P_DEFAULT,H5P_DEFAULT);	/* Create a dataset attribute. */
	status = H5Awrite(attribute_id,H5T_STD_I32LE,&signal);	/* Write the attribute data. */
	status = H5Aclose(attribute_id);						/* Close the attribute. */
	H5Sclose(attr_dataspace_id);

	status = status | H5Dclose(dataset_id);		/* End access to the dataset and release resources used by it. */
	status = status | H5Sclose(dataspace_id);	/* Terminate access to the data space. */ 
	status = status | H5Fclose(file_id);		/* Close the file. */

	return status;
}


herr_t writeDepthInFile(
const char *fileName,
double	depth)
{
	hid_t	file_id=0;
	hid_t	grp=0;
	hid_t	data_id=0;
	hsize_t	dims[1]={1};
	double	dataBuf[1]={0};
	int		rank;
	herr_t	tempErr, err=0;

	dataBuf[0] = depth;
	if ((file_id=H5Fopen(fileName, H5F_ACC_RDWR, H5P_DEFAULT))<=0) { fprintf(stderr,"after file open, file_id = %d\n",file_id); ERROR_PATH(file_id) }
	if ((grp=H5Gopen(file_id, "entry1",H5P_DEFAULT))<=0) ERROR_PATH(grp)

	if (H5LTget_dataset_ndims(grp,"depth",&rank)<0) {			/* does not exists */
		if (err=H5LTmake_dataset_double(grp,"depth",1,dims,dataBuf)) fprintf(stderr,"error writing dataSet depth, err = %d\n",err);
		if (err=H5LTset_attribute_string(grp, "depth", "units", "micron")) fprintf(stderr,"error writing units attribute to depth, err = %d\n",err);
	}
	else {							/* depth does exist, so just change it */
		#ifdef VERBOSE
		printf("\n\"/entry1/depth\" already exists, changing depth to %lg\n",depth);
		#endif
		if ((data_id=H5Dopen(grp,"depth",H5P_DEFAULT))<=0) ERROR_PATH(data_id)
		err = H5Dwrite(data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,dataBuf);
	}

	error_path:
	if (data_id>0) {
		if (tempErr=H5Dclose(data_id)) { fprintf(stderr,"ERROR -- writeDepthInFile(), data close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (grp>0) {
		if (tempErr=H5Gclose(grp)) { fprintf(stderr,"ERROR -- writeDepthInFile(), group close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (file_id>0) {
		if (tempErr=H5Fclose(file_id)) { fprintf(stderr,"ERROR -- writeDepthInFile(), file close error = %d\n",err); err = err ? err : tempErr; }
	}
	return err;
}


herr_t deleteDataFromFile(
hid_t	file_id,
char	*groupName,
char	*dataName)
{
	hid_t	grp=0;
	int		rank;
	herr_t	tempErr, err=0;

	if ((grp=H5Gopen(file_id,groupName,H5P_DEFAULT))<=0) ERROR_PATH(grp)
	if (H5LTget_dataset_ndims(grp,dataName,&rank)>=0) {
		if (err=H5Ldelete(grp,dataName,H5P_DEFAULT)) { fprintf(stderr,"err on '%s/%s' unlink = %d\n",groupName,dataName,err); ERROR_PATH(err) }
	}

	error_path:
	if (grp>0) {
		if (tempErr=H5Gclose(grp)) { fprintf(stderr,"ERROR -- deleteDataFromFile(), group close error = %d\n",err); err = err ? err : tempErr; }
	}
	return err;
}



int readHDF5header(
const char *fileName,		/* full path name of file to use */
struct HDF5_Header *head)	/* HDF5 header information (header must be valid!) */
{
	hid_t	file_id=0;
	hid_t	data_id=0;
	hid_t	dataspace=0;
	hid_t	dataType=0;
	herr_t	tempErr, err=0;
	char	str[MAX_micro_STRING_LEN+1];
	int		rank=0;
	hsize_t	dims_out[5]={0,0,0,0,0};		/* dataset dimensions, 5 is just larger than necessary */
	size_t	sz;
	hid_t	H5class;
	int		itype=-1;
	double	value;
	long	ivalue;
	size_t	Nimages=1;						/* number of images stored together */
	size_t	Nvec=0;
	int		oldStyleExposure;				/* old way of doing exposure time */
	int		wireFolder;						/* flag, TRUE means 'entry1/wire' exists */

	if (!head) { fprintf(stderr,"readHDF5header called with header=NULL\n"); return 1; }

	Nvec = 0;
	head->itype = -1;						/* pre-set the structure with default values */
	head->isize = 0;
	head->xdim = head->ydim = 0;
	head->xDimDet = head->yDimDet = 0;
	head->startx = head->endx = 0;
	head->starty = head->endy = 0;
	head->groupx = head->groupy = 1;
#ifdef MULTI_IMAGE_FILE
	empty_Dvector(&head->xSample);	empty_Dvector(&head->ySample);	empty_Dvector(&head->zSample);
	empty_Dvector(&head->xWire);	empty_Dvector(&head->yWire);	empty_Dvector(&head->zWire);
	empty_Dvector(&head->AerotechH);
	empty_Dvector(&head->energy);
	empty_Dvector(&head->exposure);
	empty_Dvector(&head->depth);
	empty_Dvector(&head->timingClock);
	empty_Dvector(&head->ringCurrent);
	empty_Dvector(&head->undGap);
	empty_Dvector(&head->undTaper);
#else
	head->xSample = head->ySample = head->zSample = NAN;
	head->xWire = head->yWire = head->zWire = NAN;
	head->AerotechH = NAN;
	head->energy = NAN;
	head->exposure = NAN;
	head->depth = NAN;
	head->timingClock = NAN;
	head->ringCurrent = head->undGap = head->undTaper = NAN;
#endif
	head->wirebaseX = head->wirebaseY = head->wirebaseZ = NAN;		/* wire base (micron) */
	head->monitor_I0 = head->monitor_Istart = head->monitor_Ifinal = -1;
	head->gain = head->sampleDistance = NAN;
	(head->bkgFile)[0] = (head->beamline)[0] = '\0';
	(head->detector_ID)[0] = (head->detector_model)[0] = (head->detector_vendor)[0] = '\0';
	(head->fileName)[0] = (head->fileTime)[0] = '\0';
	(head->monoMode)[0] = '\0';
	#ifdef VO2
		head->VO2_epoch = head->VO2_current = head->VO2_Temperature = head->VO2_Volt = head->VO2_Resistance = NAN;
	#endif

	if ((file_id=H5Fopen(fileName, H5F_ACC_RDONLY, H5P_DEFAULT))<=0) ERROR_PATH(file_id)

	wireFolder = groupExists(file_id,"entry1/wire");
	if (!get1HDF5attr_string(file_id,".","file_name",str)) strncpy(head->fileName,str,MAX_micro_STRING_LEN);
	if (!get1HDF5attr_string(file_id,".","file_time",str)) strncpy(head->fileTime,str,MAX_micro_STRING_LEN);
	{
		long yr,month;
		sscanf(str,"%ld-%ld",&yr,&month);
		oldStyleExposure = (12*yr + month)<24118;		/* 24118 = 12*2009 + 10 */
	}

	head->xDimDet	= get1HDF5data_int(file_id,"/entry1/detector/Nx",&ivalue) ? XDIMDET : ivalue;
	head->yDimDet	= get1HDF5data_int(file_id,"/entry1/detector/Ny",&ivalue) ? YDIMDET : ivalue;
	head->startx	= get1HDF5data_int(file_id,"/entry1/detector/startx",&ivalue) ? STARTX : ivalue;
	head->endx		= get1HDF5data_int(file_id,"/entry1/detector/endx",&ivalue) ? ENDX : ivalue;
	head->groupx	= get1HDF5data_int(file_id,"/entry1/detector/binx",&ivalue) ? GROUPX : ivalue;
	head->starty	= get1HDF5data_int(file_id,"/entry1/detector/starty",&ivalue) ? STARTY : ivalue;
	head->endy		= get1HDF5data_int(file_id,"/entry1/detector/endy",&ivalue) ? ENDY : ivalue;
	head->groupy	= get1HDF5data_int(file_id,"/entry1/detector/biny",&ivalue) ? GROUPY : ivalue;

#ifdef MULTI_IMAGE_FILE
	if (wireFolder) {								/* look in 'entry1/wire' if it exists */
		head->xWire.N = HDF5ReadDoubleVector(file_id, "entry1/wire/wireX", &(head->xWire.v));
		head->yWire.N = HDF5ReadDoubleVector(file_id, "entry1/wire/wireY", &(head->yWire.v));
		head->zWire.N = HDF5ReadDoubleVector(file_id, "entry1/wire/wireZ", &(head->zWire.v));
		head->AerotechH.N = HDF5ReadDoubleVector(file_id, "entry1/wire/AerotechH", &(head->AerotechH.v));
	}
	else {											/* this default is for old files */
		head->xWire.N = HDF5ReadDoubleVector(file_id, "entry1/wireX", &(head->xWire.v));
		head->yWire.N = HDF5ReadDoubleVector(file_id, "entry1/wireY", &(head->yWire.v));
		head->zWire.N = HDF5ReadDoubleVector(file_id, "entry1/wireZ", &(head->zWire.v));
		head->AerotechH.N = HDF5ReadDoubleVector(file_id, "entry1/AerotechH", &(head->AerotechH.v));
	}
	head->xSample.N = HDF5ReadDoubleVector(file_id, "entry1/sample/sampleX", &(head->xSample.v));
	head->ySample.N = HDF5ReadDoubleVector(file_id, "entry1/sample/sampleY", &(head->ySample.v));
	head->zSample.N = HDF5ReadDoubleVector(file_id, "entry1/sample/sampleZ", &(head->zSample.v));
	head->energy.N = HDF5ReadDoubleVector(file_id, "entry1/sample/incident_energy", &(head->energy.v));
	head->depth.N = HDF5ReadDoubleVector(file_id, "entry1/depth", &(head->depth.v));
	head->timingClock.N = HDF5ReadDoubleVector(file_id, "entry1/timingClock", &(head->timingClock.v));
	head->ringCurrent.N = HDF5ReadDoubleVector(file_id, "entry1/microDiffraction/source/current", &(head->ringCurrent.v));
	head->undGap.N = HDF5ReadDoubleVector(file_id, "entry1/microDiffraction/source/gap", &(head->undGap.v));
	head->undTaper.N = HDF5ReadDoubleVector(file_id, "entry1/microDiffraction/source/taper", &(head->undTaper.v));
	head->xWire.N = MAX(head->xWire.N,0);
	head->yWire.N = MAX(head->yWire.N,0);
	head->zWire.N = MAX(head->zWire.N,0);
	head->AerotechH.N = MAX(head->AerotechH.N,0);
	head->xSample.N = MAX(head->xSample.N,0);
	head->ySample.N = MAX(head->ySample.N,0);
	head->zSample.N = MAX(head->zSample.N,0);
	head->energy.N = MAX(head->energy.N,0);
	head->depth.N = MAX(head->depth.N,0);
	head->timingClock.N = MAX(head->timingClock.N,0);
	head->ringCurrent.N = MAX(head->ringCurrent.N,0);
	head->undGap.N = MAX(head->undGap.N,0);
	head->undTaper.N = MAX(head->undTaper.N,0);
	Nvec = MAX(head->xWire.N,head->yWire.N);
	Nvec = MAX(Nvec,head->zWire.N);
	Nvec = MAX(Nvec,head->AerotechH.N);
	Nvec = MAX(Nvec,head->xSample.N);
	Nvec = MAX(Nvec,head->ySample.N);
	Nvec = MAX(Nvec,head->zSample.N);
	Nvec = MAX(Nvec,head->energy.N);
	Nvec = MAX(Nvec,head->depth.N);
	Nvec = MAX(Nvec,head->ringCurrent.N);
	Nvec = MAX(Nvec,head->undGap.N);
	Nvec = MAX(Nvec,head->undTaper.N);
	if (Nvec<1) { fprintf(stderr,"ERROR -- readHDF5header(), all position vectors are empty\n"); ERROR_PATH(-1); }

	head->exposure.N = HDF5ReadDoubleVector(file_id, "entry1/detector/exposure", &(head->exposure.v));
	head->exposure.N = MAX(head->exposure.N,0);
	Nvec = MAX(Nvec,head->exposure.N);
	if (oldStyleExposure && head->exposure.N > 0) {
		value = head->exposure.v[0];
		if (value==value && value==round(value) && value>=0 && value<=7) {		/* value is an integer in [0,7] */
			double expTimes[]={66.5,79.9,99.8,133.2,199.9,400.0,999.8,1999.8};	/* these are times in ms, convert to sec */
			size_t i;
			for (i=0;i<head->exposure.N;i++) head->exposure.v[i] = expTimes[(int)head->exposure.v[i]] * 1e-3;
		}
	}
#else
	if (wireFolder) {								/* look in 'entry1/wire' if it exists */
		head->xWire	= get1HDF5data_float(file_id,"entry1/wire/wireX",&value) ? NAN : value;
		head->yWire	= get1HDF5data_float(file_id,"entry1/wire/wireY",&value) ? NAN : value;
		head->zWire	= get1HDF5data_float(file_id,"entry1/wire/wireZ",&value) ? NAN : value;
		head->AerotechH	= get1HDF5data_float(file_id,"entry1/wire/AerotechH",&value) ? NAN : value;
	}
	else {											/* this default is for old files */
		head->xWire	= get1HDF5data_float(file_id,"entry1/wireX",&value) ? NAN : value;
		head->yWire	= get1HDF5data_float(file_id,"entry1/wireY",&value) ? NAN : value;
		head->zWire	= get1HDF5data_float(file_id,"entry1/wireZ",&value) ? NAN : value;
		head->AerotechH	= get1HDF5data_float(file_id,"entry1/AerotechH",&value) ? NAN : value;
	}
	head->energy	= get1HDF5data_float(file_id,"entry1/sample/incident_energy",&value) ? NAN : value;
	head->timingClock= get1HDF5data_float(file_id,"entry1/timingClock",&value) ? NAN : value;
	head->depth		= get1HDF5data_float(file_id,"entry1/depth",&value) ? NAN : value;
	head->xSample	= get1HDF5data_float(file_id,"entry1/sample/sampleX",&value) ? NAN : value;
	head->ySample	= get1HDF5data_float(file_id,"entry1/sample/sampleY",&value) ? NAN : value;
	head->zSample	= get1HDF5data_float(file_id,"entry1/sample/sampleZ",&value) ? NAN : value;
	head->exposure	= get1HDF5data_float(file_id,"entry1/detector/exposure",&value) ? NAN : value;
	value = head->exposure;
	if (oldStyleExposure && value==value && value==round(value) && value>=0 && value<=7) {	/* value is an integer in [0,7] */
		double expTimes[]={66.5,79.9,99.8,133.2,199.9,400.0,999.8,1999.8};	/* these are times in ms, convert to sec */
		head->exposure = expTimes[(int)head->exposure] * 1e-3;
	}
	head->ringCurrent = get1HDF5data_float(file_id,"entry1/microDiffraction/source/current",&value) ? NAN : value;
	head->undGap = get1HDF5data_float(file_id,"entry1/microDiffraction/source/gap",&value) ? NAN : value;
	head->undTaper = get1HDF5data_float(file_id,"entry1/microDiffraction/source/taper",&value) ? NAN : value;
#endif
	/* wirebaseXYZ is useful when Aerotech at 45 is present */
	if (wireFolder){								/* look in 'entry1/wire' if it exists */
		head->wirebaseX = get1HDF5data_float(file_id,"entry1/wire/wirebaseX",&value) ? NAN : value;
		head->wirebaseY = get1HDF5data_float(file_id,"entry1/wire/wirebaseY",&value) ? NAN : value;
		head->wirebaseZ = get1HDF5data_float(file_id,"entry1/wire/wirebaseZ",&value) ? NAN : value;
	}
	else {																	/* this default is for old files */
		head->wirebaseX = get1HDF5data_float(file_id,"entry1/wirebaseX",&value) ? NAN : value;
		head->wirebaseY = get1HDF5data_float(file_id,"entry1/wirebaseY",&value) ? NAN : value;
		head->wirebaseZ = get1HDF5data_float(file_id,"entry1/wirebaseZ",&value) ? NAN : value;
	}
	head->monitor_I0	= get1HDF5data_int(file_id,"entry1/monitor/I0",&ivalue) ? -1 : ivalue;
	head->monitor_Istart= get1HDF5data_int(file_id,"entry1/monitor/I_start",&ivalue) ? -1 : ivalue;
	head->monitor_Ifinal= get1HDF5data_int(file_id,"entry1/monitor/I_final",&ivalue) ? -1 : ivalue;
	head->sampleDistance = get1HDF5data_float(file_id,"entry1/sample/distance",&value) ? NAN : value*1e-3;	/* convert mm to micron */
	head->scanNum	= get1HDF5data_int(file_id,"/entry1/scanNum",&ivalue) ? -1 : ivalue;
	head->gain		= get1HDF5data_float(file_id,"entry1/detector/gain",&value) ? NAN : value;
	value = head->gain;
	if (value==value && value==round(value) && value>=0 && value<=5) {		/* value is an integer in [0,5] */
		double gain_pF[]={0.25,0.5,1,2,4,8};								/* these are capacitance in pF */
		head->gain = gain_pF[(int)head->gain];
	}
	#ifdef VO2
		head->VO2_current = get1HDF5data_float(file_id,"entry1/sample/VO2_Current",&value) ? NAN : value;	/* VO2 current (A) */
		head->VO2_epoch = get1HDF5data_float(file_id,"entry1/sample/VO2_Epoch",&value) ? NAN : value;		/* VO2 epoch (sec) */
		head->VO2_Temperature = get1HDF5data_float(file_id,"entry1/sample/VO2_Temp",&value) ? NAN : value;	/* VO2 temperature (C) */
		head->VO2_Volt = get1HDF5data_float(file_id,"entry1/sample/VO2_Volt",&value) ? NAN : value;			/* VO2 voltage (V) */
		head->VO2_Resistance = head->VO2_Volt / head->VO2_current;											/* VO2 resistance (V) */
	#endif

	if (!get1HDF5data_string(file_id,"entry1/detector/ID",str,1023))			strncpy(head->detector_ID,str,MAX_micro_STRING_LEN);
	if (!get1HDF5data_string(file_id,"entry1/detector/Model",str,1023))			strncpy(head->detector_model,str,MAX_micro_STRING_LEN);
	if (!get1HDF5data_string(file_id,"entry1/detector/Vendor",str,1023))		strncpy(head->detector_vendor,str,MAX_micro_STRING_LEN);
	if (!get1HDF5data_string(file_id,"facility/facility_beamline",str,1023))	strncpy(head->beamline,str,MAX_micro_STRING_LEN);
	if (!get1HDF5data_string(file_id,"entry1/title",str,1023))					strncpy(head->title,str,MAX_micro_STRING_LEN);
	if (!get1HDF5data_string(file_id,"entry1/user/name",str,1023))				strncpy(head->userName,str,MAX_micro_STRING_LEN);
	if (!get1HDF5data_string(file_id,"entry1/sample/name",str,1023))			strncpy(head->sampleName,str,MAX_micro_STRING_LEN);

	head->beamBad	= get1HDF5data_int(file_id,"/entry1/microDiffraction/BeamBad",&ivalue) ? -1 : ivalue;
	head->CCDshutter= get1HDF5data_int(file_id,"/entry1/microDiffraction/CCDshutter",&ivalue) ? -1 : ivalue;	// 0=in, 1=out
	head->lightOn	= get1HDF5data_int(file_id,"/entry1/microDiffraction/LightOn",&ivalue) ? -1 : ivalue;
	head->hutchTemperature = get1HDF5data_float(file_id,"entry1/microDiffraction/HutchTemperature",&value) ? NAN : value;
	if (!get1HDF5data_string(file_id,"entry1/microDiffraction/MonoMode",str,1023))	strncpy(head->monoMode,str,MAX_micro_STRING_LEN);

	if ((data_id=H5Dopen(file_id,"/entry1/data/data",H5P_DEFAULT))<=0) { fprintf(stderr,"the data \"entry1/data/data\" does not exist\n"); ERROR_PATH(data_id) }
	if ((dataspace=H5Dget_space(data_id))<=0) { fprintf(stderr,"ERROR -- readHDF5header cannot get dataspace\n"); ERROR_PATH(dataspace) }    /* dataspace identifier */
	if ((dataType=H5Dget_type(data_id))<0) { fprintf(stderr,"ERROR -- readHDF5header cannot get dataType\n"); ERROR_PATH(dataType) }

	sz = H5Tget_size(dataType);
	H5class = H5Tget_class(dataType);
	if (H5class == H5T_INTEGER) {
		H5T_sign_t sign;
		sign = H5Tget_sign(dataType);
		if (sign == H5T_SGN_NONE && sz==1) itype = 7;		/* unsigned int8 (1 byte) */
		else if (sign == H5T_SGN_NONE && sz==2) itype = 3;	/* unsigned integer (2 byte) */
		else if (sign == H5T_SGN_2 && sz==4) itype = 1;		/* long integer (4 byte) */
		else if (sign == H5T_SGN_2 && sz==2) itype = 2;		/* integer (2 byte) */
		else if (sign == H5T_SGN_2 && sz==1) itype = 6;		/* integer (1 byte) */
	}
	else if (H5class == H5T_FLOAT && sz==4) itype = 0;		/* float (4 byte) */
	else if (H5class == H5T_FLOAT && sz==8) itype = 5;		/* float (8 byte) */
	else itype = -1;										/* what is this */

	rank = H5Sget_simple_extent_dims(dataspace,dims_out,NULL);
	if (!(rank==2 || rank==3)) ERROR_PATH(rank)
	#ifdef VERBOSE
	printf("		data type = %d,  size of one element = %ld bytes\n",dataType,sz);
	printf("		rank %d, dims_out = [%llu, %llu, %llu, %llu, %llu]\n", rank, dims_out[0],dims_out[1],dims_out[2],dims_out[3],dims_out[4]);
	if (rank==2) printf("		data contains 1 simple image\n");
	else if (rank==3) printf("		data contains %llu, images\n", dims_out[0]);
	#endif

	head->itype	= itype;									/* Old WinView types */
	head->isize	= sz;										/* length of one element (one pixel in bytes) */
#ifdef RECONSTRUCT_BACKWARDS
	if (rank==2) {
		head->xdim	= (size_t)dims_out[0];					/* size of image in file */
		head->ydim	= (size_t)dims_out[1];
	}
	else if (rank==3) {
		Nimages = (size_t)dims_out[0];
		head->xdim	= (size_t)dims_out[1];
		head->ydim	= (size_t)dims_out[2];
	}
#else
	if (rank==2) {
		head->xdim	= (size_t)dims_out[1];					/* size of image in file, HDF stores them backwards! */
		head->ydim	= (size_t)dims_out[0];
	}
	else if (rank==3) {
		Nimages = (size_t)dims_out[0];
		head->xdim	= (size_t)dims_out[2];					/* size of image in file, HDF stores them backwards! */
		head->ydim	= (size_t)dims_out[1];
	}
#endif
	head->Nimages = Nimages;
	long dimtest;
	dimtest = (head->endx - head->startx + 1) / (int)(head->groupx);
	if (dimtest != (long)head->xdim) { fprintf(stderr,"ERROR -- readHDF5header  X size of ROI = %ld, but xdim=%lu\n",dimtest,head->xdim); ERROR_PATH(dataspace) }    /* dataspace identifier */
	dimtest = (head->endy - head->starty + 1) / (int)(head->groupy);
	if (dimtest != (long)head->ydim) { fprintf(stderr,"ERROR -- readHDF5header Y size of ROI = %ld, but ydim=%lu\n",dimtest,head->ydim); ERROR_PATH(dataspace) }    /* dataspace identifier */

	error_path:
	if (dataType>0) {
		if (tempErr=H5Tclose(dataType)) { fprintf(stderr,"ERROR -- readHDF5header(), datatype close error = %d\n",tempErr); err = err ? err : tempErr; }
	}
	if (dataspace>0) {
		if (tempErr=H5Sclose(dataspace)) { fprintf(stderr,"ERROR -- readHDF5header(), dataspace close error = %d\n",tempErr); err = err ? err : tempErr; }
	}
	if (data_id>0) {
		if (tempErr=H5Dclose(data_id)) { fprintf(stderr,"ERROR -- readHDF5header(), data close error = %d\n",tempErr); err = err ? err : tempErr; }
	}
	if (file_id>0) {
		if (tempErr = H5Fclose(file_id)) { fprintf(stderr,"ERROR -- readHDF5header(), file close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (err) return err;

	if (Nvec>0 && rank==3 && Nvec!=(Nimages+1)) { fprintf(stderr,"ERROR -- readHDF5header(), Nimages = %lu,  but biggest vector has %lu values\n",Nimages,Nvec); err=-1; }
	return err;
}



int printHeader(
struct HDF5_Header *h)		/* HDF5 header information (header must be valid!) */
{
	char	str[50];
	if (!h) return 1;
#ifdef MULTI_IMAGE_FILE
	int		Nx,Ny,Nz;
#endif

	printf("data file header:\n");
	printf("	itype = %d,  %s\n",h->itype,getFileTypeString(h->itype,str));
	printf("	length of one pixel %d bytes\n",h->isize);								/* number of bytes in each pixel */
	printf("	full un-binned detector is %lu x %lu pixels\n",h->xDimDet,h->yDimDet);	/* x-y dimension of detector (pixels) */
	printf("	image in file is %ld x %ld binned pixels",h->xdim,h->ydim);			/* x,y dimensions of image (after any internal binning) */
	if (h->Nimages > 1) printf(",	file contains %lu images",h->Nimages);			/* number of images in file */
	printf("\n");
	printf("	binned region X=[%ld, %ld], group=%ld\n",h->startx,h->endx,h->groupx);	/* ROI (unbinned pixels) */
	printf("	binned region Y=[%ld, %ld], group=%ld\n",h->starty,h->endy,h->groupy);
#ifdef MULTI_IMAGE_FILE
	Nx = h->exposure.N;
	if (Nx > 0) {																	/* exposure time (seconds) */
		if (Nx > 1) printf("	exposure times = [%g, %g](%d) sec",h->exposure.v[0],h->exposure.v[Nx-1],Nx);
		else printf("	exposure time = %g sec",h->exposure.v[0]);
	}
	if (!isnan(h->gain)) {
		if (Nx > 0) printf(",   ");
		printf("	gain = %gpF",h->gain);											/* and gain */
	}
	if (Nx > 0 || !isnan(h->gain)) printf("\n");
#else
	if (!isnan(h->exposure)) printf("	exposure time = %g sec",h->exposure);		/* exposure time (seconds) */
	if (!isnan(h->gain)) printf("	gain = %gpF",h->gain);							/* and gain */
	if (!isnan(h->exposure) || !isnan(h->gain)) printf("\n");
#endif
	if (!isnan(h->sampleDistance)) printf("	sample distance = %g micron\n",h->sampleDistance);	/* sample distance from Keyence (micron) */

	#ifdef VO2
		if (!isnan(h->VO2_epoch)) printf("	VO2 epoch = %g seconds\n",h->VO2_current);			/* VO2 epoch (sec) */
		if (!isnan(h->VO2_current)) printf("	VO2 current = %g Amp\n",h->VO2_current);		/* VO2 current (A) */
		if (!isnan(h->VO2_Temperature)) printf("	VO2 temperature = %g C\n",h->VO2_Temperature);	/* VO2 temperature (C) */
		if (!isnan(h->VO2_Volt)) printf("	VO2 voltage = %g V\n",h->VO2_Volt);					/* VO2 voltage (V) */
		if (!isnan(h->VO2_Resistance)) printf("	VO2 resistance = %g Ohm\n",h->VO2_Resistance);	/* VO2 resistance (Ohm) */
	#endif

	if (strlen(h->bkgFile)) printf("	background file = '%s'\n",h->bkgFile);		/* name of possible background file */
	if (strlen(h->detector_ID)+strlen(h->detector_model)) printf("	detector is model='%s',  ID = '%s',  Vendor = '%s'\n",h->detector_model,h->detector_ID,h->detector_vendor);
	if (strlen(h->fileName)) printf("	original file = '%s'\n",h->fileName);
	if (strlen(h->fileTime)) printf("	image taken on '%s'\n",h->fileTime);
	if ((h->scanNum)>=0) printf("	scan number = %d\n",h->scanNum);

	printf("\t--------\n");
	if (strlen(h->title)) printf("	title = '%s'\n",h->title);
	if (strlen(h->sampleName)+strlen(h->userName)) printf("	for sample '%s',  user = '%s'\n",h->sampleName,h->userName);
#ifdef MULTI_IMAGE_FILE
	Nx = h->xSample.N;		Ny = h->ySample.N;		Nz = h->zSample.N;
	if (Nx>0 || Ny>0 || Nz>0) {
		printf("	sample positioner\n"); /* sample position from PM500 */
		if (Nx>0) printf(",  X=[%g, %g](%d)",h->xSample.v[0],h->xSample.v[Nx-1],Nx);
		if (Ny>0) printf(",  Y=[%g, %g](%d)",h->ySample.v[0],h->ySample.v[Ny-1],Ny);
		if (Nz>0) printf(",  Z=[%g, %g](%d)",h->zSample.v[0],h->zSample.v[Nz-1],Nz);
		printf(" (micron)\n");
	}
	Nx = h->xWire.N;		Ny = h->yWire.N;		Nz = h->zWire.N;
	if (Nx>0 || Ny>0 || Nz>0) {														/* wire position from Alio Stage */
		printf("	wire raw positioner");
		if (Nx>0) printf(",  X=[%g, %g](%d)",h->xWire.v[0],h->xWire.v[Nx-1],Nx);
		if (Ny>0) printf(",  Y=[%g, %g](%d)",h->yWire.v[0],h->yWire.v[Ny-1],Ny);
		if (Nz>0) printf(",  Z=[%g, %g](%d)",h->zWire.v[0],h->zWire.v[Nz-1],Nz);
		printf(" (micron)\n");
	}

	Nx = h->AerotechH.N;															/* AerotechH (micron) */
	if (Nx>0) printf("	AerotechH = [%lg, %lg](%d) (micron)\n",h->AerotechH.v[0],h->AerotechH.v[Nx-1],Nx);

	Nx = h->energy.N;																/* monochromator energy (keV) */
	if (Nx>0) printf("	energy = [%lg, %lg](%d) (keV)\n",h->energy.v[0],h->energy.v[Nx-1],Nx);

	Nx = h->depth.N;																/* depth of reconstructed image (micron) */
	if (Nx>0) printf("	depth = [%lg, %lg](%d) (keV)\n",h->depth.v[0],h->depth.v[Nx-1],Nx);

	Nx = h->timingClock.N;															/* timing clock from Struck (second) */
	if (Nx>0) printf("	timingClock = [%lg, %lg](%d) (sec)\n",h->timingClock.v[0],h->timingClock.v[Nx-1],Nx);

	Nx = h->ringCurrent.N;
	if (Nx>0) printf("	ring current = [%lg, %lg](%d) (mA)\n",h->ringCurrent.v[0],h->ringCurrent.v[Nx-1],Nx);

	Nx = h->undGap.N;																/* undulator gap (mm) */
	if (Nx>0) printf("	undulator gap = [%lg, %lg](%d) (mm)\n",h->undGap.v[0],h->undGap.v[Nx-1],Nx);

	Nx = h->undTaper.N;																/* undulator taper (mm) */
	if (Nx>0) printf("	undulator taper = [%lg, %lg](%d) (mm)\n",h->undTaper.v[0],h->undTaper.v[Nx-1],Nx);

#else
	if (!isnan(h->xSample + h->ySample + h->zSample))
		printf("	sample PM500 = {%g, %g, %g} (micron)\n",h->xSample,h->ySample,h->zSample); /* sample position from PM500 */
	if (!isnan(h->xWire + h->yWire + h->zWire))
		printf("	wire raw positioner = {%g, %g, %g} (micron)\n",h->xWire,h->yWire,h->zWire);	/* wire position from wire stage */
	if (!isnan(h->AerotechH)) printf("	AerotechH = %lg (micron)\n",h->AerotechH);				/* AerotechH (micron) */
	if (!isnan(h->energy)) printf("	energy = %lg (keV)\n",h->energy);							/* monochromator energy (keV) */
	if (!isnan(h->depth)) printf("	depth = %lg (micron)\n",h->depth);							/* depth of reconstructed image (micron) */
	if (!isnan(h->timingClock)) printf("	timingClock = %lg (sec)\n",h->timingClock);			/* timing clock from Struck (second) */
	if (!isnan(h->ringCurrent)) printf("	current = %lg (mA)\n",h->ringCurrent);				/* storage ring current (mA) */
	if (!isnan(h->undGap)) printf("	gap = %lg (mm)\n",h->undGap);								/* undulator gap (mm) */
	if (!isnan(h->undTaper)) printf("	taper = %lg (mm)\n",h->undTaper);						/* undulator taper (mm) */
#endif
	if (!isnan(h->wirebaseX + h->wirebaseY + h->wirebaseZ))
		printf("	wire base = {%g, %g, %g} (micron)\n",h->wirebaseX,h->wirebaseY,h->wirebaseZ);	/* wire position from Alio Stage */
	if (h->monitor_I0 >= 0 || h->monitor_Istart >= 0 || h->monitor_Ifinal >= 0) {
		printf("	monitor");								/* monitor intensities */
		if (h->monitor_I0 >= 0) printf(", I0 = %d (counts)\n",h->monitor_I0);					/* monitor I0 */
		if (h->monitor_Istart >= 0) printf(", I_start = %d (counts)\n",h->monitor_Istart);		/* monitor I_start */
		if (h->monitor_Ifinal >= 0) printf(", I_final = %d (counts)\n",h->monitor_Ifinal);		/* monitor I_final */
		printf("\n");
	}
	if (!isnan(h->hutchTemperature)) printf("	hutchTemperature = %lg (C)\n",h->hutchTemperature);	/* Hutch Temperature (C) */
	if (h->beamBad == 1) printf("	Beam is BAD !!!!!!!!!!!!!!!!!!!\n");
		else if (h->beamBad == 0) printf("	Beam is OK\n");
	if (h->lightOn == 1) printf("	microscope illuminator is ON\n");
		else if (h->lightOn == 0) printf("	microscope illuminator is OFF\n");
	if (h->CCDshutter == 0) printf("	CCD shutter was IN  !!!!!!!!!!!!!!!!!!!\n");
		else if (h->CCDshutter == 1) printf("	CCD shutter was OUT\n");
	if (strlen(h->monoMode)) printf("	monochromator mode = '%s'\n",h->monoMode);
	if (strlen(h->beamline)) printf("	measured at beam line '%s'\n",h->beamline);

	return 0;
}


/* a short cut routine to read just one value from a file, data always returned as double */
double readHDF5oneValue(
const char *fileName,				/* full path name of file to use */
const char *dataName)				/* full data name, e.g. "entry1/wireX" */
{
	hid_t	file_id=0;
	herr_t	tempErr, err=0;
	double	value=NAN;				/* default to NaN */

	hid_t	data_id=0;
	hid_t	dataType=0;
	hid_t	H5class;
	hsize_t	total;
	size_t	size;
	int		un_signed=0;
	hid_t	scalarSpace;			/* set scalarSpace to H5S_ALL to read the whole vector or array, here we only want just the first value */

	if (strlen(fileName)<1 || strlen(dataName)<1) ERROR_PATH(-1);
	if ((file_id=H5Fopen(fileName, H5F_ACC_RDONLY, H5P_DEFAULT))<=0) ERROR_PATH(file_id)
	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<0) ERROR_PATH(data_id)	/* probably not there, failure, but not a real error */
	if ((dataType=H5Dget_type(data_id))<0) { fprintf(stderr,"ERROR -- readHDF5oneValue cannot get dataType\n"); ERROR_PATH(dataType) }

	size = H5Tget_size(dataType);						/* size of one element */
	total = H5Dget_storage_size(data_id);				/* get total size of data in this data set */
	if (total<size) { fprintf(stderr,"ERROR, readHDF5oneValue(), %llu != %lu\n",total,size); ERROR_PATH(-1) }
	scalarSpace = H5Screate(H5S_SCALAR);
	//	scalarSpace = H5S_ALL;

	#ifdef VERBOSE
	InfoAboutDataType(dataType);
	printf("		total data size = %ld bytes\n",total);
	#endif

	H5class = H5Tget_class(dataType);
	if (H5class==H5T_INTEGER) un_signed = (H5Tget_sign(dataType)==H5T_SGN_NONE);

	if (H5class==H5T_FLOAT && size==4) {					/* single precision float */
		float fnum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&fnum);
		value = fnum;
	}
	else if (H5class==H5T_FLOAT && size==8) {				/* double precision float */
		double dnum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&dnum);
		value = dnum;
	}
	else if (H5class==H5T_INTEGER && un_signed && size==1) {/* 1 byte unsigned int */
		unsigned char inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		value = inum;
	}
	else if (H5class==H5T_INTEGER && size==1) {				/* 1 byte signed int */
		char inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		value = inum;
	}
	else if (H5class==H5T_INTEGER && un_signed && size==2) {/* 2 byte unsigned int */
		unsigned short int inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		value = inum;
	}
	else if (H5class==H5T_INTEGER && size==2) {				/* 2 byte signed int */
		short int inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		value = inum;
	}
	else if (H5class==H5T_INTEGER && un_signed && size==4) {/* 4 byte unsigned int */
		unsigned long inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		value = inum;
	}
	else if (H5class==H5T_INTEGER && size==2) {				/* 4 byte signed int */
		long inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		value = inum;
	}
	else { fprintf(stderr,"ERROR -- readHDF5oneValue(), data not a valid float or int type\n"); ERROR_PATH(-1) }
	if (err) { fprintf(stderr,"ERROR, readHDF5oneValue(), data read error = %d\n",err); ERROR_PATH(err) }
  
	error_path:
	if (dataType>0) {
		if (tempErr=H5Tclose(dataType)) { fprintf(stderr,"ERROR -- readHDF5oneValue(), dataType close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (data_id>0) {
		if (tempErr=H5Dclose(data_id)) { fprintf(stderr,"ERROR -- readHDF5oneValue(), data close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (file_id>0) {
		if (tempErr = H5Fclose(file_id)) { fprintf(stderr,"ERROR -- readHDF5oneValue(), file close error = %d\n",err); err = err ? err : tempErr; }
	}
	return value;
}

double readHDF5oneHeaderValue(			/* return value of item in header based on string in name */
struct HDF5_Header *head,
char *name)						/* name of value to return */
{
	if (!strcmp(name,"itype"))			return (double)head->itype;
	else if (!strcmp(name,"isize"))		return (double)head->isize;
	else if (!strcmp(name,"xDimDet"))	return (double)head->xDimDet;
	else if (!strcmp(name,"yDimDet"))	return (double)head->yDimDet;
	else if (!strcmp(name,"xdim"))		return (double)head->xdim;
	else if (!strcmp(name,"ydim"))		return (double)head->ydim;
	else if (!strcmp(name,"startx"))	return (double)head->startx;
	else if (!strcmp(name,"endx"))		return (double)head->endx;
	else if (!strcmp(name,"groupx"))	return (double)head->groupx;
	else if (!strcmp(name,"starty"))	return (double)head->starty;
	else if (!strcmp(name,"endy"))		return (double)head->endy;
	else if (!strcmp(name,"groupy"))	return (double)head->groupy;
	else if (!strcmp(name,"gain"))		return head->gain;
#ifdef MULTI_IMAGE_FILE
	else if (!strcmp(name,"xSample"))	return head->xSample.N ? head->xSample.v[0] : NAN;
	else if (!strcmp(name,"ySample"))	return head->ySample.N ? head->ySample.v[0] : NAN;
	else if (!strcmp(name,"zSample"))	return head->zSample.N ? head->zSample.v[0] : NAN;
	else if (!strcmp(name,"xWire"))		return head->xWire.N ? head->xWire.v[0] : NAN;
	else if (!strcmp(name,"yWire"))		return head->yWire.N ? head->yWire.v[0] : NAN;
	else if (!strcmp(name,"zWire"))		return head->zWire.N ? head->zWire.v[0] : NAN;
	else if (!strcmp(name,"AerotechH"))	return head->AerotechH.N ? head->AerotechH.v[0] : NAN;
	else if (!strcmp(name,"energy"))	return head->energy.N ? head->energy.v[0] : NAN;
	else if (!strcmp(name,"depth"))		return head->depth.N ? head->depth.v[0] : NAN;
	else if (!strcmp(name,"timingClock")) return head->timingClock.N ? head->timingClock.v[0] : NAN;
	else if (!strcmp(name,"exposure"))	return head->exposure.N ? head->exposure.v[0] : NAN;
	else if (!strcmp(name,"mA"))		return head->ringCurrent.N ? head->ringCurrent.v[0] : NAN;
	else if (!strcmp(name,"gap"))		return head->undGap.N ? head->undGap.v[0] : NAN;
	else if (!strcmp(name,"taper"))		return head->undTaper.N ? head->undTaper.v[0] : NAN;
#else
	else if (!strcmp(name,"xSample"))	return head->xSample;
	else if (!strcmp(name,"ySample"))	return head->ySample;
	else if (!strcmp(name,"zSample"))	return head->zSample;
	else if (!strcmp(name,"xWire"))		return head->xWire;
	else if (!strcmp(name,"yWire"))		return head->yWire;
	else if (!strcmp(name,"zWire"))		return head->zWire;
	else if (!strcmp(name,"AerotechH"))	return head->AerotechH;	
	else if (!strcmp(name,"energy"))	return head->energy;
	else if (!strcmp(name,"depth"))		return head->depth;
	else if (!strcmp(name,"timingClock"))	return head->timingClock;
	else if (!strcmp(name,"exposure"))	return head->exposure;
	else if (!strcmp(name,"mA"))		return head->ringCurrent;
	else if (!strcmp(name,"gap"))		return head->undGap;
	else if (!strcmp(name,"taper"))		return head->undTaper;
#endif
	else if (!strcmp(name,"wirebaseX")) return head->wirebaseX;
	else if (!strcmp(name,"wirebaseY")) return head->wirebaseY;
	else if (!strcmp(name,"wirebaseZ")) return head->wirebaseZ;
	else if (!strcmp(name,"I0")) return (double) (head->monitor_I0);
	else if (!strcmp(name,"I_start")) return (double) (head->monitor_Istart);
	else if (!strcmp(name,"I_final")) return (double) (head->monitor_Ifinal);
	else if (!strcmp(name,"sampleDistance")) return head->sampleDistance;
	#ifdef VO2
		else if (!strcmp(name,"VO2_epoch")) return head->VO2_epoch;
		else if (!strcmp(name,"VO2_current")) return head->VO2_current;
		else if (!strcmp(name,"VO2_Temperature")) return head->VO2_Temperature;
		else if (!strcmp(name,"VO2_Volt")) return head->VO2_Volt;
		else if (!strcmp(name,"VO2_Resistance")) return head->VO2_Resistance;
	#endif
	else if (!strcmp(name,"scanNum"))	return (double)head->scanNum;
	else if (!strcmp(name,"beamBad"))	return (double)head->beamBad;
	else if (!strcmp(name,"CCDshutter"))return (double)head->CCDshutter;
	else if (!strcmp(name,"LightOn"))	return (double)head->lightOn;
	else if (!strcmp(name,"hutchTemperature"))	return head->hutchTemperature;
	return NAN;
}

int readHDF5oneHeaderVector(
const char *fileName,			/* full path name of file to use */
char *name,						/* name of value to return */
Dvector *vec)
 {
	hid_t	file_id=0;
	herr_t	err=0;

	if ((file_id=H5Fopen(fileName, H5F_ACC_RDONLY, H5P_DEFAULT))<=0) return file_id;
	(*vec).N = HDF5ReadDoubleVector(file_id, name, &((*vec).v));

	if (file_id>0) {
		if (err = H5Fclose(file_id)) { fprintf(stderr,"ERROR -- readHDF5header(), file close error = %d\n",err); return err; }
	}
	return err;
}


/* NOTE, if dest->XXXX.v is not NULL, then I assume enough space has been allocated, if not there will be trouble!!!!! */
void copyHDF5structure(							/* copy all information from 'in' to 'dest' */
struct HDF5_Header *dest,						/* destination structure */
struct HDF5_Header *in)							/* input structure */
{
	dest->itype		= in->itype;
	dest->isize		= in->isize;
	dest->xDimDet	= in->xDimDet;
	dest->yDimDet	= in->yDimDet;
	dest->xdim		= in->xdim;
	dest->ydim		= in->ydim;
	dest->startx	= in->startx;
	dest->endx		= in->endx;
	dest->groupx	= in->groupx;
	dest->starty	= in->starty;
	dest->endy		= in->endy;
	dest->groupy	= in->groupy;
	dest->gain		= in->gain;
	dest->sampleDistance = in->sampleDistance;
	#ifdef VO2
		dest->VO2_epoch = in->VO2_epoch;
		dest->VO2_current = in->VO2_current;
		dest->VO2_Temperature = in->VO2_Temperature;
		dest->VO2_Volt = in->VO2_Volt;
		dest->VO2_Resistance = in->VO2_Resistance;
	#endif
#ifdef MULTI_IMAGE_FILE
	/* if dest->XXXX.v is not NULL, then assume enough space has been allocated */
	dest->xSample.N = in->xSample.N;
	if (in->xSample.N > 0) {
		if (!(dest->xSample.v)) dest->xSample.v = (double*)calloc(in->xSample.N,sizeof(double));
		memcpy(dest->xSample.v, in->xSample.v, (size_t)(sizeof(double)*(in->xSample.N)));
	}

	dest->ySample.N = in->ySample.N;
	if (in->ySample.N > 0) {
		if (!(dest->ySample.v)) dest->ySample.v = (double*)calloc(in->ySample.N,sizeof(double));
		memcpy(dest->ySample.v, in->ySample.v, (size_t)(sizeof(double)*(in->ySample.N)));
	}

	dest->zSample.N = in->zSample.N;
	if (in->zSample.N > 0) {
		if (!(dest->zSample.v)) dest->zSample.v = (double*)calloc(in->zSample.N,sizeof(double));
		memcpy(dest->zSample.v, in->zSample.v, (size_t)(sizeof(double)*(in->zSample.N)));
	}

	dest->xWire.N = in->xWire.N;
	if (in->xWire.N > 0) {
		if (!(dest->xWire.v)) dest->xWire.v = (double*)calloc(in->xWire.N,sizeof(double));
		memcpy(dest->xWire.v, in->xWire.v, (size_t)(sizeof(double)*(in->xWire.N)));
	}

	dest->yWire.N = in->yWire.N;
	if (in->yWire.N > 0) {
		if (!(dest->yWire.v)) dest->yWire.v = (double*)calloc(in->yWire.N,sizeof(double));
		memcpy(dest->yWire.v, in->yWire.v, (size_t)(sizeof(double)*(in->yWire.N)));
	}

	dest->zWire.N = in->zWire.N;
	if (in->zWire.N > 0) {
		if (!(dest->zWire.v)) dest->zWire.v = (double*)calloc(in->zWire.N,sizeof(double));
		memcpy(dest->zWire.v, in->zWire.v, (size_t)(sizeof(double)*(in->zWire.N)));
	}

	dest->AerotechH.N = in->AerotechH.N;
	if (in->AerotechH.N > 0) {
		if (!(dest->AerotechH.v)) dest->AerotechH.v = (double*)calloc(in->AerotechH.N,sizeof(double));
		memcpy(dest->AerotechH.v, in->AerotechH.v, (size_t)(sizeof(double)*(in->AerotechH.N)));
	}

	dest->energy.N = in->energy.N;
	if (in->energy.N > 0) {
		if (!(dest->energy.v)) dest->energy.v = (double*)calloc(in->energy.N,sizeof(double));
		memcpy(dest->energy.v, in->energy.v, (size_t)(sizeof(double)*(in->energy.N)));
	}

	dest->depth.N = in->depth.N;
	if (in->depth.N > 0) {
		if (!(dest->depth.v)) dest->depth.v = (double*)calloc(in->depth.N,sizeof(double));
		memcpy(dest->depth.v, in->depth.v, (size_t)(sizeof(double)*(in->depth.N)));
	}

	dest->timingClock.N = in->timingClock.N;
	if (in->timingClock.N > 0) {
		if (!(dest->timingClock.v)) dest->timingClock.v = (double*)calloc(in->timingClock.N,sizeof(double));
		memcpy(dest->timingClock.v, in->timingClock.v, (size_t)(sizeof(double)*(in->timingClock.N)));
	}

	dest->exposure.N = in->exposure.N;
	if (in->exposure.N > 0) {
		if (!(dest->exposure.v)) dest->exposure.v = (double*)calloc(in->exposure.N,sizeof(double));
		memcpy(dest->exposure.v, in->exposure.v, (size_t)(sizeof(double)*(in->exposure.N)));
	}

	dest->ringCurrent.N = in->ringCurrent.N;
	if (in->ringCurrent.N > 0) {
		if (!(dest->ringCurrent.v)) dest->ringCurrent.v = (double*)calloc(in->ringCurrent.N,sizeof(double));
		memcpy(dest->ringCurrent.v, in->ringCurrent.v, (size_t)(sizeof(double)*(in->ringCurrent.N)));
	}

	dest->undGap.N = in->undGap.N;
	if (in->undGap.N > 0) {
		if (!(dest->undGap.v)) dest->undGap.v = (double*)calloc(in->undGap.N,sizeof(double));
		memcpy(dest->undGap.v, in->undGap.v, (size_t)(sizeof(double)*(in->undGap.N))); 
	}

	dest->undTaper.N = in->undTaper.N;
	if (in->undTaper.N > 0) {
		if (!(dest->undTaper.v)) dest->undTaper.v = (double*)calloc(in->undTaper.N,sizeof(double));
		memcpy(dest->undTaper.v, in->undTaper.v, (size_t)(sizeof(double)*(in->undTaper.N)));
	}
#else
	dest->exposure	= in->exposure;
	dest->xSample	= in->xSample;
	dest->ySample	= in->ySample;
	dest->zSample	= in->zSample;
	dest->xWire		= in->xWire;
	dest->yWire		= in->yWire;
	dest->zWire		= in->zWire;
	dest->AerotechH	= in->AerotechH;
	dest->energy	= in->energy;
	dest->depth		= in->depth;
	dest->timingClock = in->timingClock;
	dest->ringCurrent = in->ringCurrent;
	dest->undGap = in->undGap;
	dest->undTaper = in->undTaper;
#endif
	dest->wirebaseX	= in->wirebaseX;
	dest->wirebaseY	= in->wirebaseY;
	dest->wirebaseZ	= in->wirebaseZ;
	dest->monitor_I0	= in->monitor_I0;
	dest->monitor_Istart= in->monitor_Istart;
	dest->monitor_Ifinal= in->monitor_Ifinal;
	dest->scanNum	= in->scanNum;
	dest->itype		= in->itype;
	dest->beamBad	= in->beamBad;
	dest->CCDshutter= in->CCDshutter;
	dest->lightOn	= in->lightOn;
	dest->hutchTemperature = in->hutchTemperature;
	strncpy(dest->bkgFile,in->bkgFile,MAX_micro_STRING_LEN);
	strncpy(dest->detector_ID,in->detector_ID,MAX_micro_STRING_LEN);
	strncpy(dest->detector_model,in->detector_model,MAX_micro_STRING_LEN);
	strncpy(dest->detector_vendor,in->detector_vendor,MAX_micro_STRING_LEN);
	strncpy(dest->fileName,in->fileName,MAX_micro_STRING_LEN);
	strncpy(dest->fileTime,in->fileTime,MAX_micro_STRING_LEN);
	strncpy(dest->beamline,in->beamline,MAX_micro_STRING_LEN);
	strncpy(dest->monoMode,in->monoMode,MAX_micro_STRING_LEN);
}


/* NOTE, if dest->XXXX.v is not NULL, then I assume enough space has been allocated, if not there will be trouble!!!!! */
void initHDF5structure(				/* initialize all values to 'empty', particularly strings and vectors */
struct HDF5_Header *h)				/* destination header structure */
{
	h->itype = -1;					/* -1	"an error" */
	h->isize = 0;					/* length in bytes of one element */
	h->xDimDet = h->yDimDet = 0;	/* xy-dimension of chip (pixels) */
	h->xdim = h->ydim = 0;			/* x,y dimensions of image (after any internal binning */
	h->startx = h->starty = 0;		/* start pixel in ROI (unbinned pixels) */
	h->endx = h->endy = 0;			/* highest x pixel value (unbinned pixels) */
	h->groupx = h->groupy = 1;		/* amount x is binned/grouped in hardware */
/*	h->geo_rotate = h->geo_reverse = h->geo_flip = 0;		/* geometric effect applied */
	h->Nimages = 1;					/* number of images stored together */
	h->gain = NAN;					/* actually capacitance (pF) */
#ifdef MULTI_IMAGE_FILE
	init_Dvector(&h->xSample);	init_Dvector(&h->ySample);	init_Dvector(&h->zSample);
	init_Dvector(&h->xWire);	init_Dvector(&h->yWire);	init_Dvector(&h->zWire);
	init_Dvector(&h->AerotechH);
	init_Dvector(&h->exposure);
	init_Dvector(&h->energy);
	init_Dvector(&h->depth);
	init_Dvector(&h->timingClock);
	init_Dvector(&h->ringCurrent);
	init_Dvector(&h->undGap);
	init_Dvector(&h->undTaper);
#else
	h->exposure = NAN;				/* exposure time (seconds) */
	h->xSample = h->ySample = h->zSample = NAN;	/* xyz sample position from PVlist */
	h->xWire = h->yWire = h->zWire = NAN;		/* xyz wire position from PVlist */
	h->AerotechH = NAN;
	h->energy = NAN;				/* monochromator energy (keV) */
	h->depth = NAN;					/* depth of reconstructed image (micron) */
	h->timingClock = NAN;
	h->ringCurrent = NAN;			/* current in storage ring (mA) */
	h->undGap = NAN;				/* undulator gap (mm) */
	h->undTaper = NAN;				/* undulator taper (mm) */
#endif
	h->sampleDistance = NAN;		/* sample distance, from the Keyence (micron) */
	h->wirebaseX = h->wirebaseY = h->wirebaseZ = NAN;		/* wire base (micron) */
	h->scanNum = -1;				/* scan number, not very important */
	h->beamBad = 0;					/* beam is bad (e.g. shutter closed) 1=Bad, 0=OK */
	h->CCDshutter = 1;				/* CCD shutter, 0=IN, 1=OUT */
	h->lightOn = 1;					/* microscope illuminator 1=ON, 0=OFF */
	h->hutchTemperature = NAN;		/* hutch temperature (C) */
	h->monitor_I0 = h->monitor_Istart = h->monitor_Ifinal = -1;
	#ifdef VO2
		h->VO2_epoch = h->VO2_current = h->VO2_Temperature = h->VO2_Volt = 	h->VO2_Resistance = h->VO2_epoch = NAN; /* set all VO2 variable to NAN */
	#endif

	(h->bkgFile)[0] = '\0';			/* name of possible background file */
	(h->detector_ID)[0] = '\0';
	(h->detector_model)[0] = '\0';
	(h->detector_vendor)[0] = '\0';
	(h->fileName)[0] = '\0';
	(h->fileTime)[0] = '\0';
	(h->beamline)[0] = '\0';
	(h->title)[0] = '\0';
	(h->userName)[0] = '\0';
	(h->sampleName)[0] = '\0';
	(h->monoMode)[0] = '\0';
}




hid_t getHDFtype(			/* returns HDF5 number type */
int	itype)					/* the WinView number type */
{
	switch (itype) {
		case 0:  return H5T_INTEL_F32;		/* float (4 byte) */
		case 1:  return H5T_NATIVE_INT32;	/* long integer (4 byte) */
		case 2:  return H5T_NATIVE_INT16;	/* integer (2 byte) */
		case 3:  return H5T_NATIVE_UINT16;	/* unsigned integer (2 byte) */
		case 5:  return H5T_INTEL_F64;		/* double (8 byte) */
		case 6:  return H5T_NATIVE_INT8;	/* signed int8 (1 byte) */
		case 7:  return H5T_NATIVE_UINT8;	/* unsigned int8 (1 byte) */
	}
	return -1;								/* invalid, itype=4 is a character which is also invalid */

/*void testTypes(void);
 *void testTypes(void)
 *{
 *	hid_t	H5class;
 *	int		sz;
 *	hid_t	htype;
 *	char	str[256], buf[255];
 *	int		i;
 *
 *	for (i=0;i<9;i++) {
 *		htype = getHDFtype(i);
 *		sz = H5Tget_size(htype);
 *		H5class = H5Tget_class(htype);
 *		if (H5class == H5T_INTEGER) {
 *			H5T_sign_t sign;
 *			sign = H5Tget_sign(htype);
 *			if (sign == H5T_SGN_NONE && sz==1) strcpy(str,"H unsigned int8 (1 byte)");
 *			else if (sign == H5T_SGN_NONE && sz==2) strcpy(str,"H unsigned integer (2 byte)");
 *			else if (sign == H5T_SGN_2 && sz==4) strcpy(str,"H long integer (4 byte)");
 *			else if (sign == H5T_SGN_2 && sz==2) strcpy(str,"H integer (2 byte)");
 *			else if (sign == H5T_SGN_2 && sz==1) strcpy(str,"H integer (1 byte)");
 *		}
 *		else if (H5class == H5T_FLOAT && sz==4) strcpy(str,"H float (4 byte)");
 *		else if (H5class == H5T_FLOAT && sz==8) strcpy(str,"H float (8 byte)");
 *		else strcpy(str,"H Invalid");
 *		printf("itype = %ld (%s), HDF5 sz=%ld,  H5class=%ld, (%s)\n",i,getFileTypeString(i,buf),sz,H5class,str);
 *	}
 */}


char *getFileTypeString(	/* puts a descriptive string into stype */
int		itype,				/* WinView file type */
char	*stype)				/* to recieve descriptive string make at least 30 long */
{
	char	*name[]={"float (4 byte)","long integer (4 byte)","integer (2 byte)","unsigned integer (2 byte)",\
					"string/char (1 byte)","double (8 byte)","signed int8 (1 byte)","unsigned int8 (1 byte)"};
/*	if (itype<0 || itype>7) stype[0]='\0'; */
	if (itype<0 || itype>7) strcpy(stype,"INVALID type");
	else strncpy(stype,name[itype],30);
	return stype;
}



void InfoAboutData(
hid_t data_id)
{
	hid_t	dataspace=0;
	int		rank;
	hsize_t	dims_out[5];				/* dataset dimensions, 5 is just larger than necessary */
	long	npoints;
	size_t	size;
	hid_t	dataType;

	if ((dataType=H5Dget_type(data_id))>=0) {
		InfoAboutDataType(dataType);
		H5Tclose(dataType);
		dataType = 0;
	}

	if ((dataspace=H5Dget_space(data_id))<=0) return;    /* dataspace identifier */

	if (H5Sis_simple(dataspace)) printf("		simple (not complicated) data\n");
	else printf("		scalar data\n");

	npoints = (long)H5Sget_simple_extent_npoints(dataspace);
	size = (size_t)H5Dget_storage_size(data_id);
	if ((rank = H5Sget_simple_extent_dims(dataspace,dims_out,NULL))>0) {
		printf("		total dataSize = %ld bytes,  and number of points = %ld\n",size,npoints);	/* get total size of data in bytes */
		printf("		rank %d, dimensions %llu", rank, dims_out[0]);
		int i;
		for (i=1;i<rank;i++) printf(" x %llu",dims_out[i]);
		printf("\n");
	}
	else printf("rank = %d\n",rank);

	H5Sclose(dataspace);
	return;
}


void InfoAboutDataType(
hid_t dataType)
{
	hid_t order;
	hid_t H5class;
	H5T_cset_t charSet;
	size_t sz;

	sz = H5Tget_size(dataType);
	printf("		data type = %d,  size of one element = %ld bytes\n",dataType,sz);

	order = H5Tget_order(dataType);
    if (order == H5T_ORDER_LE) printf("		Little endian order\n");
    else if (order == H5T_ORDER_BE) printf("		Big endian order\n");
	else printf("		Unknown order for data type %d,  may be string\n",dataType);

    H5class = H5Tget_class(dataType);
	if (H5class == H5T_INTEGER) {
		H5T_sign_t sign;
		sign = H5Tget_sign(dataType);
		if (sign == H5T_SGN_NONE) printf("		Dataset has UNsigned integers\n");
		else if (sign == H5T_SGN_2) printf("		Dataset has signed integers\n");
		else if (sign = H5T_SGN_ERROR) printf("		sign for this Dataset is an ERROR\n");
		else printf("		for this data type, sign = %d, UNKNOWN integer type\n",sign);

	}
	else if (H5class == H5T_FLOAT) {
		printf("		float or double type\n");
	}
	else if (H5class == H5T_STRING) {
		charSet = H5Tget_cset(dataType);
		printf("		Dataset has String type   ");
		if (charSet == H5T_CSET_ASCII) printf("character set is US ASCII\n");
		else if (charSet == H5T_CSET_UTF8) printf("character set is UTF-8 Unicode encoding\n");
		else printf("Character set error, charSet = %d\n",charSet);
	}
	else if (H5class == H5T_TIME) printf("		Dataset has Time type\n");
	else if (H5class == H5T_NO_CLASS) printf("		Dataset has no known H5class\n");
	else printf("		Dataset has some other strange type of H5class\n");
	return;
}


int	WinView_itype2len_new(		/* convert WinView file type to number of bytes/pixel, 0 is for error*/
int		itype)		/* WinView itype */
{
	int		ilen;

	switch (itype) {
		case 4:			/* string/char (1 byte) */
		case 6:			/* signed int8 (1 byte) */
		case 7:			/* unsigned int8 (1 byte) */
			ilen=1;
			break;
		case 2:			/* integer (2 byte) */
		case 3:			/* unsigned integer (2 byte) */
			ilen=2;
			break;
		case 0:			/* float (4 byte) */
		case 1:			/* long integer (4 byte) */
			ilen=4;
			break;
		case 5:			/* double (8 byte) */
			ilen=8;
			break;
		default:
			ilen = 0;	/* error */
	}
	return ilen;
}


void init_Dvector(
Dvector	*d)
{
	d->N = 0;
	d->v = NULL;
}


void empty_Dvector(
Dvector	*d)
{
	d->N = 0;
	CHECK_FREE(d->v);
}

/*******************************************************************************************/
/*******************************************************************************************/




/*******************************************************************************************
**********************************  Local HDF5 Routines  ***********************************
********************************************************************************************/


#define SKIP 1							/* number at start of a vector to skip, for 1, skip first point of a vector */

/* read in a 1D vector from an HDF5 file.  To get header information, first call HDF5ReadHeader */
/* the image is in vbuf, the result is always returned as type "double" regardless of how it is stored */
/* returns number of values in vector, returns negative on error */
/* CAUTION,  It allocates space for image in vbuf only if vbuf is NULL, so pass it with a null value, and remember to free it later yourself */
/* CAUTION,  if vbuf is not NULL, then there better be enough room for the image */
size_t HDF5ReadDoubleVector(
hid_t	file_id,
const char	*dataName,					/* full path name to data, e.g. "entry1/data/data" */
double	**vbuf)							/* pointer to image, if not NULL, space is allocated here, otherwise assume vbuf is big enough, and it better be too! */
{
	herr_t	i, err=0;
	hid_t	data_id=0;					/* location id of the data in file */
	hid_t	dataspace=0;
	hid_t	memspace=0; 
	hsize_t	dims_out[5];				/* dataset dimensions, 5 is larger than necessary, we should only need 1 */
	int		rank=0;

	hsize_t	dimsm[1]={0};				/* memory space dimensions */
	hsize_t	count[1]={0};				/* size of the hyperslab in the file */
	hsize_t	offset[1]={0};				/* hyperslab offset in the file */
	hsize_t	count_out[1]={0};			/* size of the hyperslab in memory */
	hsize_t	offset_out[1]={0};			/* hyperslab offset in memory */

	size_t	dim=0;						/* number of points in vector (in file) */
	size_t	ilen;						/* number of bytes per value in output array, we are using double here */
	size_t	skip;						/* local copy of SKIP */

	ilen = sizeof(double);				/*	used to be this:	ilen = head->isize; */
	if (!ilen) return 0;								/* no word length in header */

	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<=0) ERROR_PATH(data_id)	/* probably not there, failure, but not a real error */
//	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<=0) { fprintf(stderr,"ERROR -- HDF5ReadDoubleVector(), the data '%s' does not exist\n",dataName); ERROR_PATH(data_id) }
	if ((dataspace=H5Dget_space(data_id))<=0) ERROR_PATH(-1)	/* dataspace identifier */

	/* check for existance of data */
	if ((rank=H5Sget_simple_extent_dims(dataspace,dims_out,NULL))>=0) err = (rank<0 ? -1 : 0);
	if (err) ERROR_PATH(err)
	if (rank != 1) ERROR_PATH(rank)						/* only understand rank==1 data here */
	dim	= (size_t)dims_out[0];							/* length of vector in file */
	skip = dim<=SKIP ? 0 : SKIP;						/* basically ignore SKIP if only one image */
	dim -= skip;										/* reduce length of vector saved by SKIP */
	if (dim<1) ERROR_PATH(dim)							/* nothing here */

	/* allocate space for the vector */
	#ifdef DEBUG
		printf("in HDF5ReadDoubleVector reading '%s':\n",dataName);
		printf("    about to try to allocate image space of %ld bytes\n",dim*ilen);
	#endif
	if (!(*vbuf)) *vbuf = (double*)calloc(dim,ilen);	/* allocate space here if vbuf is NULL, otherwise it better be big enough */
	if (!(*vbuf)) return 0;								/* allocation error */
	#ifdef DEBUG
		printf("    vector buffer = vbuf = %p,   dim = %lu\n",vbuf,dim);
	#endif

	dimsm[0] = dim;										/* memory space dimensions */
	if ((memspace=H5Screate_simple(1,dimsm,NULL))<0) ERROR_PATH(memspace)	/* Define the memory space, to receive data */
//	offset[0] = 0;										/* define which part of the vector in the file to read, want all */
	offset[0] = skip;									/* define which part of the vector in the file to read, want all */
	count[0] = count_out[0] = dim;						/*	size of region to read and save */

	if ((i=H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset_out,NULL,count_out,NULL))<0)	{ fprintf(stderr,"error in H5Sselect_hyperslab(memspace)=%d\n",i); ERROR_PATH(i) }
	if ((i=H5Sselect_hyperslab(dataspace,H5S_SELECT_SET,offset,NULL,count,NULL))<0)			{ fprintf(stderr,"error in H5Sselect_hyperslab(dataspace)=%d\n",i); ERROR_PATH(i) }
	if ((i=H5Dread(data_id,H5T_IEEE_F64LE,memspace,dataspace,H5P_DEFAULT,*vbuf))<0)			{ fprintf(stderr,"error in H5Dread(hyperslab)=%d\n",i); ERROR_PATH(i) }

	error_path:
	if (memspace>0) H5Sclose(memspace);
	if (dataspace>0) H5Sclose(dataspace);
	if (data_id>0) H5Dclose(data_id);
	err = err>0 ? -err : err;
	return (size_t)(err<0 ? 0 : dim);
}



herr_t get1HDF5attr_float(	/* get float value of an attribute, returns TRUE=error */
hid_t	file_id,
char	*groupName,
char	*attrName,
double	*value)				/* holds result */
{
	hid_t	grp=0;
	hid_t	attr_id=0;
	herr_t	tempErr, err=0;
	hid_t	dataType;
	hid_t	H5class;
	size_t	size;
	hsize_t	total;

	if (strlen(groupName)<1 || strlen(attrName)<1) ERROR_PATH(-1)
	if ((grp=H5Gopen(file_id,groupName,H5P_DEFAULT))<0) ERROR_PATH(grp)
	if (H5Aget_num_attrs(grp)<1) ERROR_PATH(-1)

	if ((attr_id=H5Aopen_name(grp,attrName))<=0) { fprintf(stderr,"ERROR-cannot open attribute\n"); ERROR_PATH(attr_id) }
	dataType = H5Aget_type(attr_id);						/* printf("attibute data type = %ld\n",dataType); */
	size = H5Tget_size(dataType);							/* size of one element */
	total = H5Aget_storage_size(attr_id);					/* total number of bytes stored */
	if (total!=size) { fprintf(stderr,"ERROR, get1HDF5attr_float(), %llu != %ld\n",total,size); ERROR_PATH(-1) }

/*	#ifdef VERBOSE
 *	InfoAboutDataType(dataType);
 *	printf("-------------------------\n");
 *	#endif
 */
	H5class = H5Tget_class(dataType);
	if (H5class==H5T_FLOAT && size==4) {
		float val;
		if (err=H5Aread(attr_id,dataType,&val)) {fprintf(stderr,"ERROR-attribute cannot read float\n"); ERROR_PATH(err) }
		*value = val;
	}
	else if (H5class==H5T_FLOAT && size==8) {
		double val;
		if (err=H5Aread(attr_id,dataType,&val)) {fprintf(stderr,"ERROR-attribute cannot read float\n"); ERROR_PATH(err) }
		*value = val;
	}
	else { fprintf(stderr,"ERROR-attribute not an float\n"); ERROR_PATH(-1) }

	error_path:
	if (attr_id>0) {
		if (tempErr=H5Aclose(attr_id)) { fprintf(stderr,"ERROR -- get1HDF5attr_float(), attribute close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (grp>0) {
		if (tempErr=H5Gclose(grp)) { fprintf(stderr,"ERROR -- get1HDF5attr_float(), group close error = %d\n",err); err = err ? err : tempErr; }
	}
	return err;
}




herr_t get1HDF5attr_int(	/* get integer value of an attribute, returns TRUE=error */
hid_t	file_id,
char	*groupName,
char	*attrName,
long	*value)				/* holds result */
{
	hid_t	grp=0;
	hid_t	attr_id=0;
	herr_t	tempErr, err=0;
	hid_t	dataType;
	size_t	size;
	hsize_t	total;
	hid_t	H5class;
	int		un_signed;

	if (strlen(groupName)<1 || strlen(attrName)<1) ERROR_PATH(-1)
	if ((grp=H5Gopen(file_id,groupName,H5P_DEFAULT))<0) ERROR_PATH(grp)
	if (H5Aget_num_attrs(grp)<1) ERROR_PATH(1)

	if ((attr_id=H5Aopen_name(grp,attrName))<=0) { fprintf(stderr,"ERROR-cannot open attribute\n"); ERROR_PATH(attr_id) }
	dataType = H5Aget_type(attr_id);						/* printf("attibute data type = %ld\n",dataType); */
	size = H5Tget_size(dataType);							/* size of one element */
	total = H5Aget_storage_size(attr_id);					/* total number of bytes stored */
	if (total!=size) { fprintf(stderr,"ERROR, get1HDF5attr_intVal(), %llu != %ld\n",total,size); ERROR_PATH(-1) }

/*	#ifdef VERBOSE
 *	InfoAboutDataType(dataType);
 *	printf("-------------------------\n");
 *	#endif
 */
	H5class = H5Tget_class(dataType);
	un_signed = (H5Tget_sign(dataType)==H5T_SGN_NONE);

	if (H5class==H5T_INTEGER && un_signed && size==1) {		/* 1 byte unsigned int */
		unsigned char inum;
		err = H5Aread(attr_id,dataType,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && size==1) {				/* 1 byte signed int */
		char inum;
		err = H5Aread(attr_id,dataType,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && un_signed && size==2) {	/* 2 byte unsigned int */
		unsigned short int inum;
		err = H5Aread(attr_id,dataType,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && size==2) {				/* 2 byte signed int */
		short int inum;
		err = H5Aread(attr_id,dataType,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && un_signed && size==4) {	/* 4 byte unsigned int */
		unsigned long inum;
		err = H5Aread(attr_id,dataType,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && size==2) {				/* 4 byte signed int */
		long inum;
		err = H5Aread(attr_id,dataType,&inum);
		*value = inum;
	}
	else { fprintf(stderr,"ERROR -- get1HDF5attr_int(), data not a valid int type\n"); ERROR_PATH(-1) }
	if (err) { fprintf(stderr,"ERROR, get1HDF5attr_int(), data read error = %d\n",err); ERROR_PATH(err) }

	error_path:
	if (attr_id>0) {
		if (tempErr=H5Aclose(attr_id)) { fprintf(stderr,"ERROR -- get1HDF5attr_int(), attribute close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (grp>0) {
		if (tempErr=H5Gclose(grp)) { fprintf(stderr,"ERROR -- get1HDF5attr_int(), group close error = %d\n",err); err = err ? err : tempErr; }
	}
	return err;
}



herr_t get1HDF5attr_string(	/* get string value of an attribute, returns TRUE=error */
hid_t	file_id,
char	*groupName,
char	*attrName,
char	*value)				/* holds result, should be at least 256 bytes long */
{
	hid_t	grp=0;
	hid_t	attr_id=0;
	herr_t	tempErr, err=0;
	hid_t	dataType;
	size_t	size;
	hsize_t	total;
	char	str[256];

	str[0] = value[0] = '\0';
	if (strlen(groupName)<1 || strlen(attrName)<1) ERROR_PATH(-1)
	if ((grp=H5Gopen(file_id,groupName,H5P_DEFAULT))<0) ERROR_PATH(grp)
	if (H5Aget_num_attrs(grp)<1) ERROR_PATH(-1)

	if ((attr_id=H5Aopen_name(grp,attrName))<=0) { fprintf(stderr,"ERROR-cannot open attribute\n"); ERROR_PATH(attr_id) }
	dataType = H5Aget_type(attr_id);						/* printf("attibute data type = %ld\n",dataType); */
	size = H5Tget_size(dataType);
	total = H5Aget_storage_size(attr_id);					/* total number of bytes stored */
//	if (total!=size) { fprintf(stderr,"ERROR, get1HDF5attr_intVal(), %llu != %ld\n",total,size); ERROR_PATH(-1) }
	size = total>size ? total : size;
	size = size>255 ? 255 : size;							/* must not be too long for str[256] */

/*	#ifdef VERBOSE
 *	InfoAboutDataType(dataType);
 *	printf("-------------------------\n");
 *	#endif
 */
	if (H5Tget_class(dataType)!=H5T_STRING) { fprintf(stderr,"ERROR-attribute not a string\n"); ERROR_PATH(-1) }
	if (err=H5Aread(attr_id,dataType,str)) { fprintf(stderr,"ERROR-attribute cannot read string\n"); ERROR_PATH(err) }
//	strncpy(value,str,255);
	strncpy(value,str,size);
	value[size] = '\0';										/* null terminate the value returned */

	error_path:
	if (attr_id>0) {
		if (tempErr=H5Aclose(attr_id)) { fprintf(stderr,"ERROR -- get1HDF5attr_string(), attribute close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (grp>0) {
		if (tempErr=H5Gclose(grp)) { fprintf(stderr,"ERROR -- get1HDF5attr_string(), group close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (err) printf("ERROR -- get1HDF5attr_string() trying to read group='%s', attribute='%s'\n",groupName,attrName);
	return err;
}



int get1HDF5data_float(		/* read one float (or the first one of a vector), returns TRUE=ERROR, False is OK */
hid_t	file_id,
char	*dataName,			/* full data name, includes groups from root */
double	*value)				/* holds result */
{
	hid_t	data_id=0;
	hid_t	dataType=0;
	hid_t	H5class;
	herr_t	tempErr, err=0;
	hsize_t	total;
	size_t	size;
	hid_t	scalarSpace;	/* set scalarSpace to H5S_ALL to read the whole vector or array, here we only want just the first value */

	*value = NAN;
	if (strlen(dataName)<1) ERROR_PATH(-1);

	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<0) ERROR_PATH(data_id)	/* probably not there, failure, but not a real error */
	if ((dataType=H5Dget_type(data_id))<0) { fprintf(stderr,"ERROR -- get1HDF5data_float cannot get dataType\n"); ERROR_PATH(dataType) }
	size = H5Tget_size(dataType);						/* size of one element */
	total = H5Dget_storage_size(data_id);				/* get total size of data in this data set */
	if (total<size) { fprintf(stderr,"ERROR, get1HDF5data_float(%s), %llu != %ld\n",dataName,total,size); ERROR_PATH(-1) }

	#ifdef VERBOSE
	InfoAboutDataType(dataType);
	printf("		total data size = %ld bytes\n",total);
	#endif

	scalarSpace = H5Screate(H5S_SCALAR);
//	scalarSpace = H5S_ALL;
	H5class = H5Tget_class(dataType);
	if (H5class==H5T_FLOAT && size==4) {					/* single precision float */
		float fnum;
		if (err=H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&fnum)) { fprintf(stderr,"data read error = %d\n",err); ERROR_PATH(err) }
		*value = fnum;
	}
	else if (H5class==H5T_FLOAT && size==8) {				/* double precision float */
		double dnum;
		if (err=H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&dnum)) { fprintf(stderr,"data read error = %d\n",err); ERROR_PATH(err) }
		*value = dnum;
	}
	else { fprintf(stderr,"ERROR-get1HDF5data_float(), data not a valid float type\n"); ERROR_PATH(-1) }

	error_path:
	#ifdef VERBOSE
	printf("-------------------------\n");
	#endif
	if (dataType>0) {
		if (tempErr=H5Tclose(dataType)) { fprintf(stderr,"ERROR -- get1HDF5data_float(), dataType close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (data_id>0) {
		if (tempErr=H5Dclose(data_id)) { fprintf(stderr,"ERROR -- get1HDF5data_float(), data close error = %d\n",err); err = err ? err : tempErr; }
	}
	return err;
}



int get1HDF5data_int(		/* read one int (or the first one of a vector), returns TRUE=ERROR, False is OK */
hid_t	file_id,
char	*dataName,			/* full data name, includes groups from root */
long	*value)				/* holds result */
{
	hid_t	data_id=0;
	hid_t	dataType=0;
	hid_t	H5class;
	herr_t	tempErr, err=0;
	hsize_t	total;
	size_t	size;
	int		un_signed;
	hid_t	scalarSpace;	/* set scalarSpace to H5S_ALL to read the whole vector or array, here we only want just the first value */

	*value = 0;
	if (strlen(dataName)<1) ERROR_PATH(-1)

	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<0) ERROR_PATH(data_id)	/* probably not there, failure, but not a real error */
	if ((dataType=H5Dget_type(data_id))<0) { fprintf(stderr,"ERROR -- get1HDF5data_int cannot get dataType\n"); ERROR_PATH(dataType) }

	size = H5Tget_size(dataType);						/* size of one element */
	total = H5Dget_storage_size(data_id);				/* get total size of data in this data set */
	if (total<size) { fprintf(stderr,"ERROR, get1HDF5data_int(%s), %llu != %ld\n",dataName,total,size); ERROR_PATH(-1) }
	H5class = H5Tget_class(dataType);
	un_signed = (H5Tget_sign(dataType)==H5T_SGN_NONE);
	scalarSpace = H5Screate(H5S_SCALAR);
//	scalarSpace = H5S_ALL;

	#ifdef VERBOSE
	InfoAboutDataType(dataType);
	printf("		total data size = %ld bytes\n",total);
	#endif

	if (H5class==H5T_INTEGER && un_signed && size==1) {		/* 1 byte unsigned int */
		unsigned char inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && size==1) {				/* 1 byte signed int */
		char inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && un_signed && size==2) {	/* 2 byte unsigned int */
		unsigned short int inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && size==2) {				/* 2 byte signed int */
		short int inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && un_signed && size==4) {	/* 4 byte unsigned int */
		unsigned long inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		*value = inum;
	}
	else if (H5class==H5T_INTEGER && size==4) {				/* 4 byte signed int */
		long inum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum);
		*value = inum;
	}
	else if (H5class==H5T_FLOAT && size==4) {				/* single precision float */
		float fnum;
		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&fnum);
		*value = (long)fnum;
	}
	else if (H5class==H5T_FLOAT && size==8) {				/* double precision float */
		double dnum;

		err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&dnum);
		*value = (long)dnum;
	}
	else { fprintf(stderr,"ERROR-get1HDF5data_int(), data not a valid int type\n"); ERROR_PATH(-1) }
	if (err) { fprintf(stderr,"ERROR, get1HDF5data_int(), data read error = %d\n",err); ERROR_PATH(err) }


	error_path:
	#ifdef VERBOSE
	printf("-------------------------\n");
	#endif
	if (dataType>0) {
		if (tempErr=H5Tclose(dataType)) { fprintf(stderr,"ERROR -- get1HDF5data_int(), dataType close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (data_id>0) {
		if (tempErr=H5Dclose(data_id)) { fprintf(stderr,"ERROR -- get1HDF5data_int(), data close error = %d\n",err); err = err ? err : tempErr; }
	}
	return err;
}


int get1HDF5data_string(
hid_t	file_id,
char	*dataName,
char	*value,				/* holds result, should be at least (N+1) bytes long */
long	N)					/* max length to try and put into value */
{
	hid_t	data_id=0;
	hid_t	dataType=0;
	hsize_t	total;
	herr_t	tempErr, err=0;
	char	str[MAX_micro_STRING_LEN+1];

	value[0] = '\0';
	if (strlen(dataName)<1 || N<1) ERROR_PATH(-1)

	if ((data_id=H5Dopen(file_id,dataName,H5P_DEFAULT))<0) ERROR_PATH(data_id)	/* probably not there, not a real error */
	if ((dataType=H5Dget_type(data_id))<0) { fprintf(stderr,"ERROR -- get1HDF5data_string cannot get dataType\n"); ERROR_PATH(dataType) }
	total = H5Dget_storage_size(data_id);			/* get total size of data in this data set */
	#ifdef VERBOSE
	InfoAboutDataType(dataType);
	printf("		total data size = %ld bytes\n",total);
	#endif

	if (H5Tget_class(dataType) != H5T_STRING) { fprintf(stderr,"ERROR, '%s' not string in get1HDF5data_string()\n",dataName); ERROR_PATH(-1) }

	if (total<MAX_micro_STRING_LEN) {
		if (err=H5Dread(data_id,dataType,H5S_ALL,H5S_ALL,H5P_DEFAULT,str)) fprintf(stderr,"data read error = %d\n",err);
		str[total] = '\0';
	}
	else {
		str[0] = '\0';
		fprintf(stderr,"string is too long for the buffer\n");
	}
	strncpy(value,str,MIN((size_t)N,MAX_micro_STRING_LEN));

	error_path:
	#ifdef VERBOSE
	printf("-------------------------\n");
	#endif
	if (dataType>0) {
		if (tempErr=H5Tclose(dataType)) { fprintf(stderr,"ERROR -- get1HDF5data_int(), dataType close error = %d\n",err); err = err ? err : tempErr; }
	}
	if (data_id>0) {
		if (tempErr=H5Dclose(data_id)) { fprintf(stderr,"ERROR -- get1HDF5data_string(), data close error = %d\n",err); err = err ? err : tempErr; }
	}
	return err;
}






int get1HDF5attr_tagVal(
hid_t	file_id,
char	*groupName,
char	*attrName,
char	*tagName,
char	result1[256])
{
	hid_t	grp;
	herr_t	err=0;
	hid_t	attr_id;
	hid_t	dataType;
	size_t	size;
	char	*tag;
	char	str[256];

	str[0] = result1[0] = '\0';
	if (strlen(groupName)<1 || strlen(attrName)<1) goto return_path;
	if (tagName && tagName[0]) tag = tagName;
	else tag = attrName;

	if ((err=grp=H5Gopen(file_id,groupName,H5P_DEFAULT))<0) goto return_path;
	if (H5Aget_num_attrs(grp)<1) { err=1; goto return_path; }

	attr_id = H5Aopen_name(grp,attrName);						/* printf("attr_id = %ld\n",attr_id); */
	if (attr_id>0) {
		dataType = H5Aget_type(attr_id);						/* printf("attibute data type = %ld\n",dataType); */
		size = H5Tget_size(dataType);
		#ifdef VERBOSE
		InfoAboutDataType(dataType);
		printf("-------------------------\n");
		#endif
		if (H5Tget_class(dataType)==H5T_STRING) {
			err = H5Aread(attr_id,dataType,str);				/* printf("attr read = %ld\n",err); */
			str[size] = '\0';
		}
		err = H5Aclose(attr_id);								/* printf("attr close =%s\n",err); */
		if (strlen(str)>0) sprintf(result1,"%s=%s",tag,str);
	}
	if (err=H5Gclose(grp)) { fprintf(stderr,"group close error = %d\n",err); goto return_path; }

	return_path:
	return err;
}


int get1HDF5data_tagVal(
hid_t	file_id,
char	*groupName,
char	*dataName,
char	*tagName,
char	result1[256])
{
	hid_t grp=0;
	hid_t data_id=0;
	hid_t dataType=0;
	hid_t H5class;
	hsize_t dataSize;
	H5T_cset_t charSet;
	herr_t err=0;
	hid_t	scalarSpace;	/* set scalarSpace to H5S_ALL to read the whole vector or array, here we only want just the first value */
	char	*tag;
	result1[0] = '\0';
	if (strlen(groupName)<1 || strlen(dataName)<1) goto return_path;
	if (tagName && tagName[0]) tag = tagName,255;
	else tag = dataName;

	scalarSpace = H5Screate(H5S_SCALAR);
	//	scalarSpace = H5S_ALL;

	err = grp = H5Gopen(file_id, groupName,H5P_DEFAULT);
	#ifdef VERBOSE
	printf("after group open, grp = %ld\n",grp);
	#endif
	if (err<0) goto return_path;

	err = data_id = H5Dopen(grp,dataName,H5P_DEFAULT);
	#ifdef VERBOSE
	printf("after data open, data_id = %ld\n",data_id);
	#endif
	if (err<0) goto return_path;

	dataType = H5Dget_type(data_id);
	#ifdef VERBOSE
	InfoAboutDataType(dataType);
	#endif

	dataSize = H5Dget_storage_size(data_id);			/* get total size of data in this data set */
	#ifdef VERBOSE
	printf("		total data size = %ld bytes\n",dataSize);
	#endif

    H5class = H5Tget_class(dataType);
	if (H5class == H5T_INTEGER) {							/* an integer */
		int	inum;
		if (dataSize<=4) {
			inum = 0;
			if (err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&inum)) fprintf(stderr,"data read error = %d\n",err);
			sprintf(result1,"%s=%d",tag,inum);
		}
	}
	else if (H5class == H5T_FLOAT) {
		if (dataSize<=4) {								/* a single precision float */
			float fnum;
			if (err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&fnum)) fprintf(stderr,"data read error = %d\n",err);
			sprintf(result1,"%s=%g",tag,fnum);
		}
		if (dataSize<=8) {								/* a double precision float */
			double dnum;
			if (err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,&dnum)) fprintf(stderr,"data read error = %d\n",err);
			sprintf(result1,"%s=%lg",tag,dnum);
		}
	}
	else if (H5class == H5T_STRING) {						/* a string */
		char str[256];
		charSet = H5Tget_cset(dataType);
		if (dataSize<255) {
			if (err = H5Dread(data_id,dataType,scalarSpace,scalarSpace,H5P_DEFAULT,str)) fprintf(stderr,"data read error = %d\n",err);
			str[dataSize] = '\0';
			sprintf(result1,"%s=%s",tag,str);
		}
		else {
			str[0] = '\0';
			fprintf(stderr,"string is too long to show\n");
		}
	}
	else if (H5class == H5T_TIME) fprintf(stderr,"Dataset has Time type\n");
	else if (H5class == H5T_NO_CLASS) fprintf(stderr,"Dataset has no known H5class\n");
	else fprintf(stderr,"Dataset has some other strange type of H5class\n");

	if (err = H5Dclose(data_id)) { fprintf(stderr,"data close error = %d\n",err); goto return_path; }
	data_id = 0;
	if (err = H5Gclose(grp)) { fprintf(stderr,"group close error = %d\n",err); goto return_path; }
	grp = 0;

	return_path:
	#ifdef VERBOSE
	printf("-------------------------\n");
	#endif

	if (dataType>0) H5Tclose(dataType);
	if (data_id>0) H5Dclose(data_id);
	if (grp>0) H5Gclose(grp);
	return err;
}



herr_t groupExists(	/* tests if a group exists, returns -1 if group OK, 0 if group NOT found */
hid_t	file_id,
 char	*groupName)
{
	hid_t	grp=0;
	
	if (strlen(groupName)<1) return 0;				/* no group name passed, so assume does not exist */
	grp = H5Gopen(file_id,groupName,H5P_DEFAULT);	/* try to open groupName, grp<0 is error */
	if (grp>0) {									/* the group was successfully opened, so close it */
		H5Gclose(grp);								/* close it and return -1 meaning group exists */
		return -1;
	}
	else return 0;									/* group could not be opened, assume that it does not exist */
}

/*******************************************************************************************/
/*******************************************************************************************/





/*******************************************************************************************
*************************************  Utility Section  ************************************
************************************  all are External  ************************************
********************************************************************************************/

int copyFile(				/* copy a file, sends a system call */
const char *source,			/* path to source file */
const char *dest,				/* path to destination file */
int		overWrite)			/* TRUE-overwrite existing file, FALSE-do not overwire (return failure if it exists) */
{
	int		err;
	char	*cmdbuf;
	size_t	is, id, len;

	if (!source || !dest) return -1;					/* file name not present */
	is = strlen(source);
	id = strlen(dest);
	len = is + id + 15;
	if (id<1 || is<1 || id>FILENAME_MAX || is>FILENAME_MAX) return -1;	/* bad file name */
	cmdbuf = (char*)calloc(len,sizeof(char));			/* space for value and terminating NULL */
	if (!cmdbuf) return -1;								/* allocation failed */

	if (overWrite)	sprintf(cmdbuf,"cp -f \"%s\" \"%s\"",source,dest);
	else			sprintf(cmdbuf,"cp -n \"%s\" \"%s\"",source,dest);
	if (err=system(cmdbuf)) fprintf(stderr,"ERROR -- copyFile()\n\t%s\n\t%s\n",source,dest);
	free(cmdbuf);
	return err;
}

int repackFile(				/* make a duplicate of a file, but re-pack it.  This will reclaim lost space & it always overwites existing file.  Sends a system call */
const char *source,			/* path to source file */
const char *dest)			/* path to destination file */
{
	int		err;
	char	*cmdbuf;
	char	repack[]=repackPATH;
	size_t	is, id, len;

	if (!source || !dest) return -1;					/* file name not present */
	is = strlen(source);
	id = strlen(dest);
	len = is + id + strlen(repack) + 10;
	if (id<1 || is<1 || id>FILENAME_MAX || is>FILENAME_MAX) return -1;	/* bad file name */
	cmdbuf = (char*)calloc(len,sizeof(char));			/* space for value and terminating NULL */
	if (!cmdbuf) return -1;								/* allocation failed */

	sprintf(cmdbuf,"%s \"%s\" \"%s\"",repack,source,dest);
	if (err=system(cmdbuf)) fprintf(stderr,"ERROR -- repackFile()\n\t%s\n\t%s\n",source,dest);
	free(cmdbuf);
	return err;
}



int deleteFile(				/* delete a file, sends a system call */
const char *fileName)			/* path to file that is to be deleted */
{
	int		err;
	char	*cmdbuf;
	size_t	i;

	if (!fileName) return -1;							/* file name not present */
	i = strlen(fileName);
	if (i<1 || i>FILENAME_MAX) return -1;				/* bad file name */
	cmdbuf = (char*)calloc(i+10,sizeof(char));			/* space for value and terminating NULL */
	if (!cmdbuf) return -1;								/* allocation failed */
	sprintf(cmdbuf,"rm -f \"%s\"",fileName);
	if (err=system(cmdbuf)) fprintf(stderr,"ERROR -- deleteFile()\t%s\n",fileName);
	free(cmdbuf);
	return err;
}


/*
Returns a number from a keyed list.  If the key does not exist in the list, then NaN is returned.  
Each item in the list is of the form key=value.  Where the equal sign is acutally the character 
given by keySepChar.  If keySepChar is less then or equal to 0, then an equal sign is assumed. 
Each key=value pair is separated by the listSepChar character, so the list is a string that 
looks like:   key1=val1;key2=val2;...keyN=valN.  The key string is limited to 200 characters 
long (limitation is from StringByKey()).  An example of how it might be used is:

	printf("Y1 = %g\n",NumberByKey("Y1",PVlist,0,0));

if PVlist = "X1=123;Y1=567.200;Z1=678.9", the result will be:
	
 >	Y1 = 567.2
*/
double NumberByKey_new(		/* return a number from a keyed list "key1=val1;key2=val2;...keyN=valN"*/
char	*key,				/* key string to match */
char	*list,				/* list */
char	keySepChar,			/* separator character between key and value, <= means use "=" */
char	listSepChar)		/* separator character between list elements, <=0 means use ";" */
{
		double	value;				/* the result */
		char	*ptr;				/* pointer into list, points to start of value */

		ptr = StringByKey_new(key,list,keySepChar,listSepChar,30);
		if (ptr) {
			value = atof(ptr);
			free(ptr);
		}
		else value=NAN;				/* if NAN not defined, you can use sqrt(-1.0) */
		return value;
}


/*
Returns a long int from a keyed list.  The returned value is rounede, not truncated.  If the key 
does not exist in the list, then the integer version of NaN is returned (-2147483648 ?).  
Each item in the list is of the form key=value.  Where the equal sign is acutally the character 
given by keySepChar.  If keySepChar is less then or equal to 0, then an equal sign is assumed. 
Each key=value pair is separated by the listSepChar character, so the list is a string that 
looks like:   key1=val1;key2=val2;...keyN=valN.  The key string is limited to 200 characters 
long (limitation is from StringByKey()).  An example of how it might be used is:

	printf("Y1 = %ld\n",IntByKey("Y1",PVlist,0,0));

if PVlist = "X1=123;Y1=567.800;Z1=678.9", the result will be:
	
 >	Y1 = 568
*/
long IntByKey_new(			/* return an integer from a keyed list "key1=val1;key2=val2;...keyN=valN"*/
char	*key,				/* key string to match */
char	*list,				/* list */
char	keySepChar,			/* separator character between key and value, <= means use "=" */
char	listSepChar)		/* separator character between list elements, <=0 means use ";" */
{
		return (long)floor(NumberByKey_new(key,list,keySepChar,listSepChar)+0.5);
}


/*
Return a pointer to the value in a keyed list.  This routine allocates the space for the result.  So do not 
forget to free the space when you are done!.  Each item in the list is of the form key=value.  Where the equal 
sign is acutally the character given by keySepChar.  If keySepChar is less then or equal to 0, then an equal 
sign is assumed. Each key=value pair is separated by the listSepChar character, so the list is a string that
looks like:   key1=val1;key2=val2;...keyN=valN.  The key string is limited to 200 characters long (limitation 
is from size of search),  and the returned value is limited to maxLen-1 characters (maxLen if you count 
terminating null).  An example of how it might be used is:

	char *temp;
	temp = StringByKey_new("Z1",PVlist,0,0,30);
	if (temp) {
		printf(" as a string, Z1 = '%s'\n",temp);
		free(temp);
	}
	else printf(" as a string, Z1 is NULL\n");
*/
char* StringByKey_new(		/* return a pointer to a string from a keyed list "key1=val1;key2=val2;...keyN=valN"*/
char	*key,				/* key string to match */
char	*list,				/* list */
char	keySepChar,			/* separator character between key and value, <= means use "=" */
char	listSepChar,		/* separator character between list elements, <=0 means use ";" */
int		maxLen)				/* maximum length allowed for the result */
{
	char	search[256];		/* search string */
	long	n;					/* length of search string */
	char	*ptr;				/* pointer into list, points to start of value */
	char	*ptr2;				/* pointer to end of value */
	char	*result;			/* answer, allocate space for it too */
	long	nr;					/* length of result, not > maxLen */

	if (maxLen<1) return NULL;
	if (strlen(key)>250) return NULL;		/* key string is too long */
	listSepChar  = (listSepChar<=0) ? ';' : listSepChar;
	keySepChar  = (keySepChar<=0) ? '=' : keySepChar;
	result = NULL;

	search[0] = listSepChar;				/* search = ";key=" */
	search[1] = '\0';
	strcat(search,key);
	n = strlen(search);
	search[n++] = keySepChar;
	search[n] = '\0';

	if ((ptr=strstr(list,search))) ptr += n;			/* in the middle of the list, found ";key=" */
	else if (list==(ptr=strstr(list,search+1))) ptr += (n-1);	/* at the beginning of the list, found "key=" */
	else return NULL;

	ptr2 = strchr(ptr,listSepChar);						/* points to list separator after value */
	if (!ptr2) nr = strlen(ptr);						/* assume value goes to end of string */
	else nr = ptr2-ptr;
	if (nr < 1) return NULL;							/* nothing there */
	if ((nr+1)>maxLen) return NULL;						/* value too long */

	result = (char*)calloc((size_t)nr+1,1);				/* space for value and terminating NULL */
	if (!result) exit(ENOMEM);							/* allocation error */
	strncpy(result,ptr,(size_t)nr);
	return result;
}
/*******************************************************************************************/
/*******************************************************************************************/
