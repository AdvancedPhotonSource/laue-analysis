/*
 *  depth_corection.c
 *  reconstruct
 *
 *  Created by Jon Tischler on 3/30/12.
 *  Copyright 2012 ANL. All rights reserved.
 *
 */

#include "WireScanDataTypesN.h"
#include "WireScan.h"
#include "readGeoN.h"
#include "misc.h"
#include "depth_correction.h"
#include <string.h>

//gsl_matrix_float * depthCorrection_map = NULL;


/* reads the depth correction of the detector (in pixels).  true pixel is itrue = i+depthCorrection_map(i,j),   and jtrue = j+distortion_map_j(i,j) */
/* the depth correction is a correction to the pixel */
gsl_matrix_float * load_depth_correction_map(char *filename)
{
	int		i, j;
	int		ilen, jlen;
	size_t	nlen;
	float	*buffer=NULL;
	gsl_matrix_float * depthCorrection_map = NULL;

	if (strlen(filename)<1) return NULL;

	if (verbose > 0) printf("\nloading depth correction map from: %s",filename);
	fflush(stdout);

	FILE *readfile;
	readfile = fopen(filename, "r");
	if (!readfile) {
		error("\nCannot read depth correction file");
		exit(1);
	}
	fread(&ilen,sizeof(ilen),1,readfile);
	fread(&jlen,sizeof(jlen),1,readfile);
	nlen = (size_t)ilen*jlen;
	
	buffer = (float*)malloc(sizeof(float)*nlen);
	if (!buffer) {
		fprintf(stderr,"\nCould not allocate buffer %lu bytes in load_depth_correction_map()\n",sizeof(float)*nlen);
		fclose(readfile);
		exit(1);
	}

	fread(buffer, sizeof(float), nlen, readfile);
	if (verbose > 1) { printf("\n   read the depth correction map"); fflush(stdout); }
	fclose(readfile);

	depthCorrection_map = gsl_matrix_float_alloc(ilen,jlen);
	for (j=0; j<jlen; j++) {
		for (i=0; i<ilen; i++) {
			gsl_matrix_float_set(depthCorrection_map, i, j, buffer[j*ilen + i]);
		}
	}
	if (verbose > 2) printf("\n   transfered depth correction map to a gsl matrix");
	free (buffer);
	return depthCorrection_map;
}

