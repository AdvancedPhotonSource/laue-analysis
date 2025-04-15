/*
 *  misc.h
 *  reconstruct
 *
 *  Created by Jon Tischler on 2/12/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

int readAllParameters(char *paramFile, char *infile, char *outfile, char *normalization, int *first_image, int *last_image, double *depth_start, double *depth_end, \
	double *resolution, int *out_pixel_type, int *wireEdge, int *detector, char distortionFilePath[FILENAME_MAX]);
void error (char *error);
void printCalibration(int more);
void geo2calibration(struct geoStructure *geo, int detector);
void writeSummaryHead(FILE *f, char *infile, char *outfile, char *geofile, double depth_start, double depth_end, double resolution, \
	int first_image, int last_image, int out_pixel_type, int wireEdge, char *normalization);
void writeSummaryTail(FILE *f, double seconds);

