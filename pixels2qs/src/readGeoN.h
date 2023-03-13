#ifndef READGEON_H
#define READGEON_H

/* #define	aoSi 5.43102088			// lattice constant of Silicon (Å) */
#define MAX_Ndetectors 3			/* maximum number of detectors to permitted */
#define READGEON_MAXlen 100		/* max length of a string in the structures */


struct sampleGeometry {				/* structure definition */
	double O[3];					/* raw PM500 coordinates where sample is at origin, (the Si position), (micron) */
	double R[3];					/* rotation vector for the sample positioner (length is angle in radians) */
	double Rmag;					/* magnitude of | Rs[3] |, in degrees (computed internally) */
	double R00, R01, R02;			/* rotation matrix from R[3] internally calculated */
	double R10, R11, R12;
	double R20, R21, R22;
};

struct wireGeometry {				/* structure definition */
	double origin[3];				/* raw PM500 coordinates that would put wire center at Origin, (the Si position), (micron) */
	double dia;						/* diameter of wire (micron) */
	int knife;						/* true if wire on a knife edge, false for free-standing wire */
	double F;						/* F of the wire for the wire scan in raw PM500 units (not very important) (micron) */
	double axis[3];					/* unit vector in direction of wire afis in positioner frame, e.g. (0,0,1) is along positioner x-axis */
	double axisR[3];				/* vector axis rotated by R, now direction of wire in beam line frame (calculated internally) */
	double R[3];					/* rotation vector for the wire positioner (length is angle in radians) */
	double Rmag;					/* magnitude of | R[3] |, in degrees (computed internally) */
	double R00, R01, R02;			/* rotation matrix from R[3] internally calculated */
	double R10, R11, R12;
	double R20, R21, R22;
};

/* Position of detector: first translate by P, and then rotate detector around R.  Since rho is rotation matrix calculated from R:
 *		{X,Y,Z} = rho x [P+{x',y',z'}] ,   where XYZ are beam line coords, and {x'y'z'} are coordinates in detector reference frame.
 * Size of detector is measured to the outer edge of the outer most pixels.  So conversion from position to pixel for the x direction is:
 *		x' = ( pixel - (Nx-1)/2 )*pitch,   where sizeX = Nx*pitch.  This puts the coordinate of pixel (i,j) at the center of the pixel.
 */
struct detectorGeometry {			/* structure definition for a detector */
	int used;						/* TRUE=detector used, FALSE=detector un-used */
	long Nx, Ny;					/* # of un-binned pixels in full detector */
	double sizeX,sizeY;				/* outside size of detector (sizeX = Nx*pitchX), measured to outer edge of outer pixels (micron) */
	double R[3];					/* rotation vector (length is angle in radians) */
	double P[3];					/* translation vector (micron) */

	char timeMeasured[READGEON_MAXlen+1];		/* when this geometry was calculated */
	char geoNote[READGEON_MAXlen+1];			/* note */
	char detectorID[READGEON_MAXlen+1];			/* unique detector ID */
	char distortionMapFile[READGEON_MAXlen+1];	/* name of file with distortion map */

	double rho00, rho01, rho02;		/* rotation matrix internally calculated from R[3] */
	double rho10, rho11, rho12;
	double rho20, rho21, rho22;
};

struct geoStructure {				/* structure definition */
	int	Ndetectors;					/* number of detectors, must be <= MAX_Ndetectors */
	struct detectorGeometry d[MAX_Ndetectors];	/* geometry parameters for each detector */
	struct wireGeometry wire;
	struct sampleGeometry s;
};


long readGeoFromFile(char *fname, struct geoStructure *geo);
void printGeometry(FILE *f, struct geoStructure *geo);
int checkFileTypeOLD(char *lineIn, size_t Nline, char *type);
void GeometryStructureUpdate(struct geoStructure *geo);
int printDetector(FILE *f, struct detectorGeometry *d);
int MicroGeometryBad(struct geoStructure *g);
void copyMicroGeometryStructure(struct geoStructure *dest, struct geoStructure *in);
int strFromTagBuf(char *buffer, char *tagIn, char *value, long maxLen);

#endif
