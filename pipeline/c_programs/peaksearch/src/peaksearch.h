

#include <stdbool.h>
#include <limits.h>
#include <math.h>

#include "WinViewImage.h"
#include "grid.h"
#include "grid_operations.h"
#include "point.h"
#include "list.h"
#include "fitToFunction.h"
#include "genfileinf.h"
#include "peak.h"
#include "calibparam.h"
#include "ccdTable.h"

#include "minmax.h"

#include "microHDF5.h"

#ifndef _PEAKSEARCH_H_
#define _PEAKSEARCH_H_


struct ExtraOutput_Header {
	double	depth;					/* depth for depth resolved images (micron) */
	double	energy;					/* monchromator energy */
	long	scanNum;				/* scan number */
	int		beamBad;				/* beam bad flag (TRUE==bad) */
	int		CCDshutterIN;			/* CCD shutter, 1=IN, 0=OUT */
	int		lightOn;				/* flag, TRUE=illuminator ON */
	double	hutchTemperature;		/* hutch temperature (C) */
	double	sampleDistance;			/* Keyence measure of sample posiiton (micron) */
	double	sum;					/* total of all pixels in image */
	double	sumAboveThreshold;		/* sum of all pixels above threshold */
	size_t	numAboveThreshold;		/* number of pixels above threshold */
	char	monoMode[MAX_micro_STRING_LEN+1];
	char	dateExposed[MAX_micro_STRING_LEN+1];
	char	userName[MAX_micro_STRING_LEN+1];
	char	title[MAX_micro_STRING_LEN+1];
	char	sampleName[MAX_micro_STRING_LEN+1];
	char	beamline[MAX_micro_STRING_LEN+1];
	char	detector_ID[MAX_micro_STRING_LEN+1];
	int		NpeakMax;				/* only search the first NpeakMax peaks, this limits the search */
	char	maskFile[MAX_micro_STRING_LEN+1];	/* name of mask file, only use pixels with mask==0 */
	};



List* boxsearch(Grid* imageRaw, GridB* mask, int boxsize, long ipeakMax, bool smooth, Genfileinf *ginf);
//List * processBlobs(List *blobs, WinViewImage *wimage,Genfileinf *ginf);
List * processBlobs(List *blobs, WinViewImage *wimage,Genfileinf *ginf, int NpeakMax);
//List*	blobsearch(Grid* image, double threshold, int min_size, bool maxima_search, double saturation_level);
List*	blobsearch(Grid* image, double threshold, int min_size, bool maxima_search);
List * removeNearbyPeaks(List *peaks, int minSeparation);

//List*	find_maximas(Grid* image, double threshold, int npix, double saturation_level, int shiftx, int shifty);

//List*	get_blob_list(Grid* image, Grid* bitmap);
//void	get_blob_points(Grid* image, Grid* bitmap, List* list, int x, int y);
//void peakCorrection(double *fitX, double *fitY,Genfileinf *ginf);
//bool peakQulify(double fitX,double fitY,double centX,double centY,double widthx, double widthy, double chisq,Genfileinf *ginf);
//double peakIntegral(Grid *image,double fitX,double fitY,Genfileinf *ginf,double originalIntens);
//double peakIntegral(Grid *image,double fitX,double fitY,Genfileinf *ginf);
void savePeaksIDL(List *peaks, char * filename);
void savePeaks(List *peaks, char * filename, WinViewHeader* header, char * inFileName, Genfileinf *ginf, 
	double threshold, double seconds, int minSeparation, bool smooth, struct ExtraOutput_Header *exH, char *pgm);
void sorListPoints(List *blobs);
#endif
