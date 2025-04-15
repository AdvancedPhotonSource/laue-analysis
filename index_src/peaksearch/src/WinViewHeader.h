/**********************************************************

	Data Structure and Routines for working with
	WinViewHeader data types. This is a collection of
	values describing an SPE file's properties

/**********************************************************/


#include <stdlib.h>
#include <string.h>
#include <errno.h>


#ifndef _WINVIEW_HEADER_H_
#define _WINVIEW_HEADER_H_

#define EPICS_MAX_STRLEN	40

typedef struct {
	int		itype;			/* WinView number type of image */
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
	int		ADCrate;		/* ADC rate */
	int		ADCtype;		/* ADC type */
	int		ADCresolution;	/* ADC resolution */
	int		geo_rotate;		/* geometric effect applied, rotate */
	int		geo_reverse;	/* geometric effect applied, reverse */
	int		geo_flip;		/* geometric effect applied, flipped */
	double	exposure;		/* exposure time (seconds) */
	int		controllerType;	/* flag indicating controller type  (see WinViewControllers) */
	char	bkgFile[256];	/* name of possible background file upto 255 chars long */
	double	xSample;		/* x sample position from PVlist */
	double	ySample;		/* y sample position from PVlist */
	double	zSample;		/* z sample position from PVlist */
	double	xWire;			/* x wire position from PVlist */
	double	yWire;			/* y wire position from PVlist */
	double	zWire;			/* z wire position from PVlist */
	double	CCDy;			/* CCD y position from PVlist */
	char	PVlist[5*EPICS_MAX_STRLEN];	/* pointer to list of PV1=val1;PV2=val2;PV3=val3;... */
							/* no space allocated in the structure for this list */
	} WinViewHeader;
	
WinViewHeader* winview_header_new(void);
WinViewHeader* winview_header_copy(WinViewHeader* h);
void winview_header_delete(WinViewHeader* h);

#endif
