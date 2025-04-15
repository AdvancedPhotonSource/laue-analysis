/*
 *  microHDF5.h
 *  
 *
 *  Created by Jon Tischler on 1/6/09.
 *  Copyright 2009 ORNL. All rights reserved.
 *
 */

#ifndef _MICRO_HDF5_READ_
#define _MICRO_HDF5_READ_

#include "/opt/local/include/hdf5.h"
#include "/opt/local/include/hdf5_hl.h"		/* hdf lite, I can probably get rid of this with a little effort */

/* #define MAX_DETECTOR_ID_LEN 255		/* max length of string with detector ID */
#define MAX_micro_STRING_LEN 1023		/* max length of a string easily read from a data */


#ifndef MAX
#define MAX(X,Y) ( ((X)<(Y)) ? (Y) : (X) )
#endif
#ifndef MIN
#define MIN(X,Y) ( ((X)>(Y)) ? (Y) : (X) )
#endif
#ifndef ERROR_PATH
#define ERROR_PATH(A) { err=(A); goto error_path; }
#endif

#ifndef _HDF5_Header_	/* HDF5 header values */
#define _HDF5_Header_
struct HDF5_Header {
	int		itype;			/* from WinView .spe image types
								-1	"an error"
								0	"float (4 byte)"
								1	"long integer (4 byte)"
								2	"integer (2 byte)"
								3	"unsigned integer (2 byte)"
								4	"string/char (1 byte)",  NOT USED with HDF5 here
								5	"double (8 byte)"
								6	"signed int8 (1 byte)"
								7	"unsigned int8 (1 byte)" */
	int		isize;			/* length in bytes of one element */
	size_t 	xDimDet;		/* x-dimension of chip (pixels) */
	size_t	yDimDet;		/* y-dimension of chip (pixels) */
	size_t	xdim;			/* x,y dimensions of image (after any internal binning */
	size_t	ydim;
	size_t	startx;			/* start pixel in ROI (unbinned pixels) */
	size_t	endx;			/* highest x pixel value (unbinned pixels) */
	size_t	groupx;			/* amount x is binned/grouped in hardware */
	size_t	starty;
	size_t	endy;
	size_t	groupy;
/*	int		geo_rotate;		/* geometric effect applied, rotate */
/*	int		geo_reverse;	/* geometric effect applied, reverse */
/*	int		geo_flip;		/* geometric effect applied, flipped */
	double	exposure;		/* exposure time (seconds) */
	char	bkgFile[MAX_micro_STRING_LEN+1];	/* name of possible background file */
	double	xSample;		/* x sample position from PVlist */
	double	ySample;		/* y sample position from PVlist */
	double	zSample;		/* z sample position from PVlist */
	double	xWire;			/* x wire position from PVlist */
	double	yWire;			/* y wire position from PVlist */
	double	zWire;			/* z wire position from PVlist */
	double	energy;			/* monochromator energy (keV) */
	double	depth;			/* depth of reconstructed image (micron) */
	double	sampleDistance;	/* sample distance, from the Keyence (micron) */
	int		scanNum;		/* scan number, not very important */
	char	detector_ID[MAX_micro_STRING_LEN+1];
	char	detector_model[MAX_micro_STRING_LEN+1];
	char	fileName[MAX_micro_STRING_LEN];		/* name of original file */
	char	fileTime[MAX_micro_STRING_LEN+1];	/* time when original file was written */
	char	beamline[MAX_micro_STRING_LEN+1];
	char	title[MAX_micro_STRING_LEN+1];		/* title of scan */
	char	userName[MAX_micro_STRING_LEN+1];	/* name of user */
	char	sampleName[MAX_micro_STRING_LEN+1];	/* name of sample */
	};
#endif


int HDF5WriteROI(const char *fileName, const char *dataName, void *vbuf, size_t xlo, size_t xhi, size_t ylo, size_t yhi, hid_t dataType, struct HDF5_Header *head);
//int HDF5WriteROI(const char *fileName, const char *dataName, void *vbuf, size_t xlo, size_t xhi, size_t ylo, size_t yhi, struct HDF5_Header *head);
int HDF5ReadROI(const char *fileName, const char *dataName, void **vbuf, size_t xlo, size_t xhi, size_t ylo, size_t yhi, struct HDF5_Header *head);
int HDF5ReadROIdouble(const char *fileName, const char *dataName, double **vbuf, size_t xlo, size_t xhi, size_t ylo, size_t yhi, struct HDF5_Header *head);
int createNewData(const char *fileName, const char *dataName, int rank, int *dims, int dataType);
int readHDF5header(const char *fileName, struct HDF5_Header *head);
int printHeader(struct HDF5_Header *h);
double readHDF5oneValue(const char *fileName, const char *dataName);
double readHDF5oneHeaderValue(struct HDF5_Header *head, char *name);
herr_t writeDepthInFile(const char *fileName, double depth);
herr_t deleteDataFromFile(hid_t file_id, char *groupName, char *dataName);

char *getFileTypeString(int itype, char *stype);
hid_t getHDFtype(int itype);
void InfoAboutData(hid_t data_id);
void InfoAboutDataType(hid_t dataType);
int WinView_itype2len(int itype);


int repackFile(const char *source, const char *dest);
int copyFile(const char *source, const char *dest, int overWrite);
int deleteFile(const char *fileName);
void copyHDF5structure(struct HDF5_Header *dest, struct HDF5_Header *in);
double NumberByKey(char *key, char *list, char keySepStr, char listSepStr);
long IntByKey(char *key, char *list, char keySepChar, char listSepChar);
char* StringByKey(char *key, char *list, char keySepChar, char listSepChar, int maxLen);

/*
int WinViewReadHeader(FILE *fid, struct WinViewHeader *head);
int WinViewParseHeader(char *buf, struct WinViewHeader *head);
int WinViewReadROI(FILE *fid, void  **vbuf, size_t xlo, size_t xhi, size_t ylo, size_t yhi, struct WinViewHeader *head);
int WinViewWriteROI(FILE *fid, char *vbuf, int itype, size_t xdim, size_t xlo, size_t xhi, size_t ylo, size_t yhi);
char *onlyNamePartOfFile(char *full, char *name);
char *WinViewControllers(int itype, char *stype);
char *WinViewFileTypeString(int itype,char *stype);
int WinView_itype2len(int itype);
int checkTypeSizes();
void printBufferHex(void *buffer, size_t bufSizeOf, size_t n, size_t offset);
long byteSwap2(void *j);
void byteSwapArray(size_t wordLen, size_t words, void *a);
*/
/* for .spe itype==
  0	"float (4 byte)"
  1	"long integer (4 byte)"
  2	"integer (2 byte)"
  3	"unsigned integer (2 byte)"
  4	"string/char (1 byte)"
  5	"double (8 byte)"
  6	"signed int8 (1 byte)"
  7	"unsigned int8 (1 byte)"
*/


#endif	/* _MICRO_HDF5_READ_ */
