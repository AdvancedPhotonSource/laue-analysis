/*
 *  misc.c
 *  reconstruct
 *
 *  Created by Jon Tischler on 2/12/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include "mathUtil.h"

#include "WireScanDataTypesN.h"
#include "WireScan.h"
#include "readGeoN.h"
#include "checkFileType.h"
#include "misc.h"


/* read in all of the parameters from an ASCII file, this is only used for the -F flag on input */
int readAllParameters(
char	*paramFile,		/* name of parameter file to read */
char	*infile,		/* root name of input images */
char	*outfile,		/* root name of output images */
char	*normalization,	/* tag to use for normalizing incident intensity */
int		*first_image,	/* index of first input image */
int		*last_image,	/* index of last input image */
double	*depth_start,	/* lowest depth of reconstructed images	(micron) */
double	*depth_end,		/* greatest depth of reconstructed images (micron) */
double	*resolution,	/* depth resolution (micron) */
int		*out_pixel_type,/* number type for the output pixels */
int		*wireEdge,		/* 1=leading edge of wire, 0=trailing edge of wire, -1=both edges */
int		*detector,		/* detector number */
char distortionFilePath[FILENAME_MAX])
{
	FILE *f;
	char	*buf=NULL;								/* string with tag values */
	size_t	len;									/* allocated length of buf[] */
	char	*p;										/* generic pointer into a string */
	char	line[502];								/* line of data read from file */
	int		n=0;									/* flags geometry parameters as they are read */
	int		err = 1;
	char	tag[256];								/* tag for geo file, needed because of id[] */
	double	x,y,z;									/* used for reading 3-vectors */

	len = 200*1024;									/* allocate space for the header, 200KB should be more than enough */
	buf = (char*)calloc(len,sizeof(char));
	if (!buf) { fprintf(stderr,"unable to allocate space for 'buf' in readGeoFromFile()\n"); goto exitPoint; }
	if ((f = fopen(paramFile, "r")) == NULL) { fprintf(stderr,"Can't open file '%s' in readAllParameters()\n",paramFile);  return(err); }
	len = fread(buf,sizeof(char),len-1,f);
	fclose(f);
	if (len<1) { fprintf(stderr,"unable to read buffer in readGeoFromFile()\n"); goto exitPoint; }

	buf[len-1] = '\0';								/* ensure null terminated */
	p = buf;
	while(p=strchr(p,'\r')) *p = '\n';				/* convert all carriage returns to new lines */

	if (!(p = strchr(buf,'\n'))) goto exitPoint;	/* p points to the first new-line (if none found --> invalid file) */
//	if (!checkFileType(buf,p-buf,"depthSortedInfo")) goto exitPoint;	/* check the file type */
	if (!checkFileTypeLine(buf,"depthSortedInfo")) goto exitPoint;	/* check the file type */

	/* read program parameters from input file */
	n = 0;
	if (!strFromTagBuf(buf,"ws_infile",line,250))			{ strncpy(infile,line,FILENAME_MAX); n=n|1<<0; }	/* input file root */
	if (!strFromTagBuf(buf,"ws_outfile",line,250))			{ strncpy(outfile,line,FILENAME_MAX); n=n|1<<1; }	/* output file root */
	if (!strFromTagBuf(buf,"ws_depthStart",line,250))		{ *depth_start = atof(line); n=n|1<<2; }			/* first depth relative to Si (micron) */
	if (!strFromTagBuf(buf,"ws_depthEnd",line,250))			{ *depth_end = atof(line); n=n|1<<3; }				/* last depth relative to Si (micron) */
	if (!strFromTagBuf(buf,"ws_depthResolution",line,250))	{ *resolution = atof(line); n=n|1<<4; }				/* depth resolution (micron) */
	if (!strFromTagBuf(buf,"ws_firstInputIndex",line,250))	{ *first_image = atol(line); n=n|1<<5; }			/* index to first input image */
	if (!strFromTagBuf(buf,"ws_lastInputIndex",line,250))	{ *last_image = atol(line); n=n|1<<6; }				/* index to last input image */
	if (!strFromTagBuf(buf,"ws_detectorNumber",line,250))	{ *detector = atol(line); n=n|1<<7; }				/* detector number */

	if (!strFromTagBuf(buf,"ws_distortionMap",line,250))	strncpy(distortionFilePath,line,250);				/* path to distortion map */
	if (!strFromTagBuf(buf,"ws_normalization",line,250))	strncpy(normalization,line,FILENAME_MAX);			/* normalization tag, not required */
	if (!strFromTagBuf(buf,"ws_percentOfPixels",line,250))	percent = (float)atof(line);						/* % of pixels used */
	if (!strFromTagBuf(buf,"ws_wireEdge",line,250))			*wireEdge = atol(line);								/* edge of wire used (1=leading, 0=trailing, -1=both) */
	if (!strFromTagBuf(buf,"ws_outputPixelType",line,250))	*out_pixel_type = atol(line);						/* nunmber type of output pixels */
	if (!strFromTagBuf(buf,"ws_MiB_RAM",line,250))			AVAILABLE_RAM_MiB = atol(line);						/* MiB of RAM used */
	if (!strFromTagBuf(buf,"ws_verbose",line,250))			verbose = atol(line);								/* verbose flag */
	if (n != (1<<8)-1) {
		error("-F when reading file, some of the required program parameters were missing\n   must have {ws_infile, ws_outfile, ws_depthStart, ws_depthEnd, ws_depthResolution, ws_firstInputIndex, ws_lastInputIndex, ws_detectorNumber}\n");
		goto exitPoint;
	}
	else if (*detector < 0 || *detector > 2) {
		error("-F when reading file, ws_detectorNumber must be 0, 1, or 2\n");
		goto exitPoint;
	}

	/* read geometry parameters from input file */
	n = 0;
	sprintf(tag,"d%d_Nx",*detector);		if (!strFromTagBuf(buf,tag,line,250))	{ geoIn.d[*detector].Nx = atol(line); n=n|1<<0; }	/* detector description */
	sprintf(tag,"d%d_Ny",*detector);		if (!strFromTagBuf(buf,tag,line,250))	{ geoIn.d[*detector].Ny = atol(line); n=n|1<<1; }
	sprintf(tag,"d%d_sizeX",*detector);		if (!strFromTagBuf(buf,tag,line,250))	{ geoIn.d[*detector].sizeX = atof(line); n=n|1<<2; }
	sprintf(tag,"d%d_sizeY",*detector);		if (!strFromTagBuf(buf,tag,line,250))	{ geoIn.d[*detector].sizeY = atof(line); n=n|1<<3; }
	sprintf(tag,"d%d_R",*detector);
	if (!strFromTagBuf(buf,tag,line,250)) {
		if (sscanf(line,"{%lg,%lg,%lg}",&x,&y,&z)==3) {
			geoIn.d[*detector].R[0] = x;
			geoIn.d[*detector].R[1] = y;
			geoIn.d[*detector].R[2] = z;
			n=n|1<<4;
		}
	}
	sprintf(tag,"d%d_P",*detector);
	if (!strFromTagBuf(buf,tag,line,250)) {
		if (sscanf(line,"{%lg,%lg,%lg}",&x,&y,&z)==3) {
			geoIn.d[*detector].P[0] = x;
			geoIn.d[*detector].P[1] = y;
			geoIn.d[*detector].P[2] = z;
			n=n|1<<5;
		}
	}
	if (!strFromTagBuf(buf,"wireDia",line,250))		{ geoIn.wire.dia = atof(line); n=n|1<<6; }	/* wire positions */
	if (!strFromTagBuf(buf,"wireKnife",line,250))	{ geoIn.wire.knife = atol(line); n=n|1<<7; }/* wire type (knife_edge=1, or stand_alone_wire=0) */
	if (!strFromTagBuf(buf,"wireOrigin",line,250)) {
		if (sscanf(line,"{%lg,%lg,%lg}",&x,&y,&z)==3) {
			geoIn.wire.origin[0] = x;
			geoIn.wire.origin[1] = y;
			geoIn.wire.origin[2] = z;
			n=n|1<<8;
		}
	}
	if (!strFromTagBuf(buf,"wireRot",line,250)) {			/* rotation of wire PM500 is optional */
		if (sscanf(line,"{%lg,%lg,%lg}",&x,&y,&z)==3) {
			geoIn.wire.R[0] = x;
			geoIn.wire.R[1] = y;
			geoIn.wire.R[2] = z;
		}
	}
	if (!strFromTagBuf(buf,"wireAxis",line,250)) {			/* wire.axis is optional */
		if (sscanf(line,"{%lg,%lg,%lg}",&x,&y,&z)==3) {
			geoIn.wire.axis[0] = x;
			geoIn.wire.axis[1] = y;
			geoIn.wire.axis[2] = z;
		}
	}
	if (!strFromTagBuf(buf,"wireF",line,250))		{ geoIn.wire.dia = atof(line); n=n|1<<9; }	/* F of wire during a scan */
	if (n != (1<<10)-1) {									/* check for needed geometry parameters */
		error("-F when reading file, some of the required geometry parameters were missing\n");
		sprintf(line,"   must have {d%d_Nx, d%d_Ny, d%d_sizeX, d%d_sizeY, d%d_R, d%d_P, wireDia, wireKnife, wireOrigin}\n",*detector,*detector,*detector,*detector,*detector,*detector);
		error(line);
		goto exitPoint;
	}

	GeometryStructureUpdate(&geoIn);						/* set the computed geometry parameters ki[], rde[][], rded[][] */
	geo2calibration(&geoIn, *detector);

	if (*wireEdge<0 && *out_pixel_type<0) *out_pixel_type = 1;	/* when using both edges of wire, need pixels of type long */
	err = 0;
	exitPoint:
		free(buf);											/* true says that all parameters were read in */
	return err;
}




void geo2calibration(struct geoStructure *geo, int detector)/* take values from geo structure and put them into global "calibration" structure */
{
	point_xyz	Rvec;										/* rotation vector that pust axis of wire along {1,0,0} */
	double		theta_convert;								/* used to make length of Rvec the rotation angle (radian) */

	calibration.ccd_pixels_i = geo->d[detector].Nx;			/* total number of raw un-binned in detector */
	calibration.ccd_pixels_j = geo->d[detector].Ny;

	calibration.ccd_dimension_i = (geo->d[detector].sizeX);	/* already micron */
	calibration.ccd_dimension_j = (geo->d[detector].sizeY);

	calibration.pixel_size_i = calibration.ccd_dimension_i/calibration.ccd_pixels_i;	/* detector size (micron) / number of full chip pixels */
	calibration.pixel_size_j = calibration.ccd_dimension_j/calibration.ccd_pixels_j;	/* size of un-binned pixels (micron) */

	calibration.P.x = geo->d[detector].P[0]; calibration.P.y = geo->d[detector].P[1]; calibration.P.z = geo->d[detector].P[2];

	calibration.ki.x = calibration.ki.y = 0;  calibration.ki.z = 1;						/* this is different than calibration.wire.ki */

	calibration.detector_rotation[0][0] = geo->d[detector].rho00;
	calibration.detector_rotation[0][1] = geo->d[detector].rho01;
	calibration.detector_rotation[0][2] = geo->d[detector].rho02;

	calibration.detector_rotation[1][0] = geo->d[detector].rho10;
	calibration.detector_rotation[1][1] = geo->d[detector].rho11;
	calibration.detector_rotation[1][2] = geo->d[detector].rho12;

	calibration.detector_rotation[2][0] = geo->d[detector].rho20;
	calibration.detector_rotation[2][1] = geo->d[detector].rho21;
	calibration.detector_rotation[2][2] = geo->d[detector].rho22;

	calibration.wire.diameter = geo->wire.dia;
	calibration.wire.F = geo->wire.F;

	calibration.wire.rotation[0][0] = geo->wire.R00;		/* rotation matrix from R[3] internally calculated */
	calibration.wire.rotation[0][1] = geo->wire.R01;		/* this is rotation between PM500 and beam line coords (does not include wire axis direction) */
	calibration.wire.rotation[0][2] = geo->wire.R02;

	calibration.wire.rotation[1][0] = geo->wire.R10;
	calibration.wire.rotation[1][1] = geo->wire.R11;
	calibration.wire.rotation[1][2] = geo->wire.R12;

	calibration.wire.rotation[2][0] = geo->wire.R20;
	calibration.wire.rotation[2][1] = geo->wire.R21;
	calibration.wire.rotation[2][2] = geo->wire.R22;

	calibration.wire.centre_at_si_xyz.x = geo->wire.origin[0];
	calibration.wire.centre_at_si_xyz.y = geo->wire.origin[1];
	calibration.wire.centre_at_si_xyz.z = geo->wire.origin[2];

	calibration.wire.axis.x = geo->wire.axis[0];			/* assumed to be already normalized */
	calibration.wire.axis.y = geo->wire.axis[1];
	calibration.wire.axis.z = geo->wire.axis[2];

	calibration.wire.axisR.x = geo->wire.axisR[0];			/* assumed to be already normalized */
	calibration.wire.axisR.y = geo->wire.axisR[1];
	calibration.wire.axisR.z = geo->wire.axisR[2];

	/* #warning "moved computation of rho to here, do not need to do this every time, and pre-calculate ki rotated by rho" */
	Rvec.x=0; Rvec.y=calibration.wire.axisR.z, Rvec.z=-calibration.wire.axisR.y;		/* just cross product, axisR x {1,0,0} */
	theta_convert = NORM(Rvec);								/* current length of Rvec which is sin(theta) */
	theta_convert = asin(theta_convert) / theta_convert;
	VEC3_MULTIPLY(Rvec,theta_convert)						/* makes length of |Rvec|=theta */
	rotationMatFromAxis(Rvec,NAN,calibration.wire.rho);		/* compute rotation matrix that puts wire axis along {1,0,0} */
	calibration.wire.ki.x = calibration.wire.ki.y = 0;		/* set ki = {0,0,1} */
	calibration.wire.ki.z = 1;
	calibration.wire.ki = MatrixMultiply31(calibration.wire.rho,calibration.wire.ki);	/* ki = rho x ki, ki in rotated frame */
}



void printCalibration(int more)		/* print formated values of the calibration structure to the scren */
{
	printf("\n\n	  orientation of this detector (from geometry) parameters:");
	printf("\n	Nx = %lg, Ny = %lg									// number of full chip un-binned pixels in detector",calibration.ccd_pixels_i,calibration.ccd_pixels_j);
	printf("\n	detector size:  %g x %g						// size of detector (mm)",calibration.ccd_dimension_j/1000,calibration.ccd_dimension_i/1000);
	printf("\n	pixel size:  %g x %g								// size of a detector pixel (micron)",calibration.pixel_size_i,calibration.pixel_size_j);
	printf("\n	P = {%g, %g, %g}						// detector displacement vector (mm)",calibration.P.x/1000,calibration.P.y/1000,calibration.P.z/1000);
	if (more) {
		printf("\n\tki =\t{%.4f, %.4f, %.4f}					// incident beam direction (Beam Line coordinates)",calibration.ki.x,calibration.ki.y,calibration.ki.z);
		printf("\n\t\t\t{%+.4f, %+.4f, %+.4f}					// rotation matrix for detector (Beam Line coordinates)",calibration.detector_rotation[0][0],calibration.detector_rotation[0][1],calibration.detector_rotation[0][2]);
		printf("\n\trded =\t{%+.4f, %+.4f, %+.4f}",calibration.detector_rotation[1][0],calibration.detector_rotation[1][1],calibration.detector_rotation[1][2]);
		printf("\n\t\t\t{%+.4f, %+.4f, %+.4f}",calibration.detector_rotation[2][0],calibration.detector_rotation[2][1],calibration.detector_rotation[2][2]);
	}
	printf("\n\n	  for the wire:");
	printf("\n	Si = {%.2f, %.2f, %.2f}					// wire (PM500 wire frame) of the origin (the Si position) (micron)",
			calibration.wire.centre_at_si_xyz.x,calibration.wire.centre_at_si_xyz.y,calibration.wire.centre_at_si_xyz.z);
	printf("\n	diameter = %.2f										// diameter of wire (micron)",calibration.wire.diameter);
	printf("\n	wireAxis = {%.6f, %.6f, %.6f}			// direction of wire axis (PM500 frame)",calibration.wire.axis.x,calibration.wire.axis.y,calibration.wire.axis.z);
	if (more) {
		printf("\n	wireAxisR = {%.6f, %.6f, %.6f}		// direction of wire axis (beam line coords)",calibration.wire.axisR.x,calibration.wire.axisR.y,calibration.wire.axisR.z);
		printf("\n	ki (rotated) = {%.6f, %.6f, %.6f}	// ki when rotated to be parallel to wire axis",calibration.wire.ki.x,calibration.wire.ki.y,calibration.wire.ki.z);
		printf("\n\t\t\t{%+.4f, %+.4f, %+.4f}					// rotation matrix for wire (Beam Line coordinates)",calibration.wire.rotation[0][0],calibration.wire.rotation[0][1],calibration.wire.rotation[0][2]);
		printf("\n\trded =\t{%+.4f, %+.4f, %+.4f}",calibration.wire.rotation[1][0],calibration.wire.rotation[1][1],calibration.wire.rotation[1][2]);
		printf("\n\t\t\t{%+.4f, %+.4f, %+.4f}",calibration.wire.rotation[2][0],calibration.wire.rotation[2][1],calibration.wire.rotation[2][2]);
	}
}



/* write first part of summary file */
void writeSummaryHead(
FILE *f,
char *infile,						/* base name of input image files */
char *outfile,						/* base name of output image files */
char *geofile,						/* full path to geometry file */
double depth_start,					/* first depth in reconstruction range (micron) */
double depth_end,					/* last depth in reconstruction range (micron) */
double resolution,					/* depth resolution (micron) */
int first_image,					/* index to first input image file */
int last_image,						/* index to last input image file */
int out_pixel_type,					/* type to use for the output pixel */
int wireEdge,						/* 1=leading edge of wire, 0=trailing edge of wire, -1=both edges */
char *normalization)				/* optional tag for normalization */
{
	/* globals printed here:	percent, AVAILABLE_RAM_MiB, verbose, distortionPath */
	if (!f) return;

	fprintf(f,"$filetype	geometryFileN;depthSortedInfo\n");
	printGeometry(f,&geoIn);

	fprintf(f,"\n");
	fprintf(f,"$ws_infile				%s\n",infile);
	fprintf(f,"$ws_outfile				%s\n",outfile);
	fprintf(f,"$ws_geofile				%s\n",geofile);
	fprintf(f,"$ws_fileExtension		%s\n","h5");
	if (strlen(distortionPath)) fprintf(f,"$ws_distortionMap		%s\n",distortionPath);
	fprintf(f,"$ws_depthStart			%g				// first depth relative to Si (micron)\n",depth_start);
	fprintf(f,"$ws_depthEnd			%g				// last depth relative to Si (micron)\n",depth_end);
	fprintf(f,"$ws_depthResolution		%g				// depth resolution (micron)\n",resolution);
	fprintf(f,"$ws_firstInputIndex		%d				// index of first raw image\n",first_image);
	fprintf(f,"$ws_lastInputIndex		%d				// index of last raw image\n",last_image);
	if (normalization[0]) fprintf(f,"$ws_normalization		%s				// tag used to normalize incident intensity\n",normalization);
	fprintf(f,"$ws_wireEdge			%d				// edge of wire to use, 1=leading, 0=trailing, -1=both\n",wireEdge);
	if (out_pixel_type>=0) fprintf(f,"$ws_outputPixelType		%d				// nunmber type of output pixels (1=long)\n",out_pixel_type);
	fprintf(f,"$ws_percentOfPixels		%g				// %% of pixels used\n",percent);
	fprintf(f,"$ws_MiB_RAM				%d				// MiB of RAM used\n",AVAILABLE_RAM_MiB);
	fprintf(f,"$ws_verbose				%d				// verbose flag\n",verbose);
}

/* write last part of summary file */
// void writeSummaryTail(
// FILE *f,
// double	seconds)					/* execution time (seconds) */
// {
// 	/* globals printed here:	first_header, imaging_parameters, image_set, user_preferences */
// 	double	H1, F1, Y1, Z1;
// 	double	keV, value;
// 	long	scanNum;
// 
// 	if (!f) return;
// 	Y1 = first_header.ySample;
// 	Z1 = first_header.zSample;
// 	H1 = (Z1+Y1)/sqrt(2.0);
// 	F1 = (Z1-Y1)/sqrt(2.0);
// 	value = first_header.scanNum;
// 
// 	if (value==value) {
// 		scanNum = (int)round(value);
// 		fprintf(f,"$scanNum				%ld			// scan number used to take this data\n",scanNum);
// 	}
// 	fprintf(f,"$startx					%ld				// x start of ROI (unbinned pixels)\n",first_header.startx);
// 	fprintf(f,"$endx					%ld			// x end of ROI (unbinned pixels)\n",first_header.endx);
// 	fprintf(f,"$groupx					%ld				// binning in x\n",first_header.groupx);
// 	fprintf(f,"$starty					%ld				// y start of ROI (unbinned pixels)\n",first_header.starty);
// 	fprintf(f,"$endy					%ld			// y end of ROI (unbinned pixels)\n",first_header.endy);
// 	fprintf(f,"$groupy					%ld				// binning in y\n",first_header.groupy);
// 
// 	if (first_header.xSample == first_header.xSample)		/* not NAN */
// 	fprintf(f,"$X1						%g			// X sample position of PM500 beam line coords (micron)\n",first_header.xSample);
// 	if ((Y1+Z1) == (Y1+Z1)) {
// 		fprintf(f,"$Y1						%g			// Y sample position (micron)\n",Y1);
// 		fprintf(f,"$Z1						%g			// Z sample position (micron)\n",Z1);
// 		fprintf(f,"$H1						%g		// H sample position (micron)\n",H1);
// 		fprintf(f,"$F1						%g			// F sample position (micron)\n",F1);
// 	}
// 	keV = first_header.energy;
// 	if (keV==keV) fprintf(f,"$keV					%g			// energy of monochromator (keV)\n",keV);
// 	fprintf(f,"$rows_at_one_time		%lu			// rows to process at one time (out of %lu)\n", imaging_parameters.rows_at_one_time,first_header.ydim);
// 
// 	if (seconds > 2.) fprintf(f,"$executionTime			%.1f			// execution time (sec)\n",seconds);
// 	else fprintf(f,"$executionTime			%.3f			// execution time (sec)\n",seconds);
// 
// 	double depth, maxIntenity=image_set.depth_intensity.v[0];
// 	int i, imax=0;
// 	for (i=0,depth=user_preferences.depth_start; depth <= user_preferences.depth_end; i++, depth=user_preferences.depth_start+i*user_preferences.depth_resolution) {
// 		if (image_set.depth_intensity.v[i] > maxIntenity) {
// 			imax = i;
// 			maxIntenity = image_set.depth_intensity.v[i];
// 		}
// 	}
// 	depth = user_preferences.depth_start+imax*user_preferences.depth_resolution;
// 	fprintf(f,"\n$array0peakIndex		%d				// index of the peak intensity\n",imax);
// 	fprintf(f,"$array0peakDepth		%g				// depth of the peak intensity (micron)\n",depth);
// 	fprintf(f,"$array0\t3,%d,Index,depth(micron),Intensity\n",(int)(floor((user_preferences.depth_end - user_preferences.depth_start)/user_preferences.depth_resolution)+1));
// 	for (i=0,depth=user_preferences.depth_start; depth <= user_preferences.depth_end; i++, depth=user_preferences.depth_start+i*user_preferences.depth_resolution) {
// 		fprintf(f,"%d\t%g\t%g\n",i,depth,image_set.depth_intensity.v[i]);
// 	}
// }



void error(char *error) {
	fprintf(stderr,"\nERROR -- ");
	fprintf(stderr,error);
	fprintf(stderr,"\n");
	fflush(stdout);
}
