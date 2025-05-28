#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "readGeoN.h"
#include "keyLists.h"
#include "checkFileType.h"
#include "xmlUtility.h"

/* #define VERBOSE */

#ifndef MAX
#define MAX(X,Y) ( ((X)<(Y)) ? (Y) : (X) )
#endif
#ifndef MIN
#define MIN(X,Y) ( ((X)>(Y)) ? (Y) : (X) )
#endif

#ifndef CHECK_FREE
#define CHECK_FREE(A)   { if(A) free(A); (A)=NULL; }
#endif

int xmlBuf2GeoN(char *buf, struct geoStructure *geo);
int tagValBuf2GeoN(char *buf, struct geoStructure *geo);
int strFromTagFile(FILE *f,char *tagIn, char *value, long maxLen);
// int strFromTagBuf(char *buffer, char *tagIn, char *value, long maxLen);
int DetectorUpdateCalc(struct detectorGeometry *d);
void SampleUpdateCalc(struct sampleGeometry *sa);
void WireUpdateCalc(struct wireGeometry *w);
long stringTo3Vector(char *in, double vec[3], double factor);
int DetectorBad(struct detectorGeometry *d);
int WireBad(struct wireGeometry *w);
int SampleBad(struct sampleGeometry *s);
void copySampleGeometry(struct sampleGeometry *f, struct sampleGeometry *i);
void copyWireGeometry(struct wireGeometry *f, struct wireGeometry *i);
void copyDetectorGeometry(struct detectorGeometry *f, struct detectorGeometry *i);


/*
This is an example of what the input file looks like (this one only has 2 of the three detectors):

 $filetype		geometryFileN
 $dateWritten	Sat, Feb 20, 2010
 $timeWritten	23:19:41.5 (-6)
 $EPOCH			3349552782					// seconds from midnight January 1, 1904
 $fileNote		measured for all 3, first time ever
 
 // Sample
 $SampleOrigin	{8200.00,-4758.83,-4476.00}	// sample origin in raw PM500 units (micron)
 $SampleRot		{-0.006,0.006,-0.000018}	// sample positioner rotation vector (length is angle in radians)
 
 // Detectors
 $Ndetectors		2						// number of detectors in use, must be <= MAX_Ndetectors
 
 $d0_Nx			2048						// number of un-binned pixels in full detector
 $d0_Ny			2048
 $d0_sizeX		409.6						// size of CCD (mm)
 $d0_sizeY		409.6
 $d0_R			{-1.20310066,-1.21179927,-1.21933886}	// rotation vector (length is angle in radians)
 $d0_P			{25.826,-0.728,510.812}		// translation vector (mm)
 $d0_timeMeasured	Sat, Feb 20, 2010, 13:25:52 (-6)	// when this geometry was calculated
 $d0_geoNote	Optimized using CalibrationList_Orange0
 $d0_detectorID	PE1621 723-3335				// unique detector ID
 
 $d1_Nx			1024						// number of un-binned pixels in full detector
 $d1_Ny			1024
 $d1_sizeX		204.8						// size of CCD (mm)
 $d1_sizeY		204.8
 $d1_R			{-1.76742461,-0.72877176,-1.75941268}	// rotation vector (length is angle in radians)
 $d1_P			{-142.549,-1.003,412.988}	// translation vector (mm)
 $d1_timeMeasured	Sat, Feb 20, 2010, 14:23:05 (-6)	// when this geometry was calculated
 $d1_geoNote	Optimized using CalibrationList_yellow1
 $d1_detectorID	PE0820 763-1807				// unique detector ID
 
 // Wire
 $wireDia		52.00						// diameter of wire (micron)
 $wireKnife		0							// true if wire on a knife edge, false for free-standing wire
 $wireOrigin		{4.41,0.00,0.00}			// wire origin in raw PM500 frame (micron)
 $wireRot		{0.00449992,-0.015,-0.00003375}	// wire positioner rotation vector (length is angle in radians)
 $wireAxis		{1.0,0.0,0.0}				// unit vector along wire axis, usually close to (1,0,0)
 $wireF			-200.0						// F of wire for a constant F wire scan (raw PM500 units)
*/



/* read geometry parameters from front of a file, returns 0 if OK */
long readGeoFromFile(
char	*fname,
struct geoStructure *geo)
{
	char	*buf=NULL;								/* string with tag values */
	FILE	*f=NULL;								/* file descriptor */
	size_t	len;									/* allocated length of buf[] */
	char	*p;										/* generic pointer into a string */
	int		err=1;
	int		n=0;									/* flags geometry parameters as they are read */
	int		itype;									/* file type, 1=old, 2=xml */
	
	/* some defaults */
	geo->Ndetectors = 1;							/* # of detectors in structure */

	/* pre-sets for wire */
	geo->wire.origin[0] = geo->wire.origin[1] = geo->wire.origin[2] = 0;	/* equivalent to old xyzSi */
	geo->wire.dia = 52;								/* diameter of wire (micron) */
	geo->wire.knife = 0;							/* default to free-standing wire (true means true knife-edge) */
	geo->wire.F = 3000;								/* F of the wire for the wire scan in raw PM500 units (not very important) (micron) */
	geo->wire.axis[0] = 1;							/* default to wire.axis = {1,0,0} */
	geo->wire.axis[1] = geo->wire.axis[2] = 0;
	geo->wire.R[0] = geo->wire.R[1] = geo->wire.R[2] = 0;	/* rotation vector for the wire positioner (length is angle in radians) */

	/* pre-sets for sample */
	geo->s.O[0] = geo->s.O[0] = geo->s.O[0] = 0;	/* PM500 frame coordinates where sample is at origin, (the Si position), (micron) */
	geo->s.R[0] = geo->s.R[0] = geo->s.R[0] = 0;	/* rotation vector for the sample positioner (length is angle in radians) */

	/* pre-sets for first detector */
	geo->d[0].used = 1;								/* TRUE=detector used, FALSE=detector un-used */
	geo->d[1].used = geo->d[2].used = 0;
	geo->d[0].Nx = geo->d[0].Ny = 2048;				/* # of un-binned pixels in full detector */
	geo->d[0].sizeX = geo->d[0].sizeY = 409.6e3;	/* outside size of detector (sizeX = Nx*pitchX), measured to outer edge of outer pixels (micron) */
	geo->d[0].R[0] = geo->d[0].R[1] = geo->d[0].R[2] =  -2/3*M_PI/sqrt(3.);	/* rotation of detector, theta = -120° about (111) */
	geo->d[0].P[0]=25e3;							/* offset to detector (micron) */
	geo->d[0].P[1]=0;
	geo->d[0].P[2]=510e3;
	strcpy(geo->d[0].timeMeasured,"Dec 4, 2008, 3:33pm"); 
	strcpy(geo->d[0].geoNote,"reference orientation"); 
	strcpy(geo->d[0].detectorID,"PE1621, 723-3335"); 
	geo->d[0].distortionMapFile[0] = '\0';

	len = 200*1024;									/* allocate space for the header, 200KB should be more than enough */
	buf = (char*)calloc(len,sizeof(char));
	if (!buf) { fprintf(stderr,"unable to allocate space for 'buf' in readGeoFromFile()\n"); goto exitPoint; }
	
	itype = checkFileType(fname,"geometryFileN","geoN");	/* return 1 for old style '$', return 2 for xml, return 0 if not OK */
	if (!itype) { fprintf(stderr,"Bad geometry file '%s' in readGeoFromFile()\n",fname);  goto exitPoint; }
	if (( f = fopen(fname, "r")) == NULL) { fprintf(stderr,"Can't re-open file '%s' in readGeoFromFile()\n",fname);  goto exitPoint; }
	len = fread(buf,sizeof(char),len-1,f);
	fclose(f);
	f = NULL;
	if (len<1) { fprintf(stderr,"unable to read buffer in readGeoFromFile()\n"); goto exitPoint; }
	buf[len-1] = '\0';								/* ensure null terminated */
	p = buf;
	while(((p=strchr(p,'\r')))) *p = '\n';			/* convert all carriage returns to new lines */

	if (itype==1) n = tagValBuf2GeoN(buf,geo);		/* interpret old file type with "$tag value" pairs */
	else if(itype==2) n = xmlBuf2GeoN(buf,geo);		/* interpret new xml file type */
	else n = 0;
	if (n != (1<<10)-1) goto exitPoint;				/* did not find all of the required parameters */

	GeometryStructureUpdate(geo);					/* set the computed geometry parameters */
	err = 0;
	exitPoint:
		CHECK_FREE(buf);							/* free allocated space */
		if (f != NULL) fclose(f);					/* ensure an opened file gets closed */
	return err;
}


/* the contents of the xml geometry file is in buf, put values into the geoStructure, return n, flag showing what was read */
int xmlBuf2GeoN(
char	*buf,										/* string containing xml */
struct geoStructure *geo)
{
	int 	n;										/* flags geometry parameters as they are read */
	char	*geoN;
	char	*keyVals;
	int		Ndetectors;
	char	*Detectors=NULL, *Wire=NULL, *Sample=NULL, *detector;
	char	*str;
	int		i, id;
	
	n = 0;
	geoN = XMLtagContents("geoN",buf,0);			/* allocates space for geoN, does NOT free it */
	XMLattibutes2KeyList("Detectors",geoN,0,&keyVals);
	Ndetectors = (int)IntByKey("Ndetectors",keyVals,'=',';');
	CHECK_FREE(keyVals)
	if (Ndetectors>=1 && Ndetectors<=3)		{ geo->Ndetectors = Ndetectors; n = n|1<<0; }	/* # of detectors */
	Detectors = XMLtagContents("Detectors",geoN,0);
	geo->d[0].used = geo->d[1].used = geo->d[2].used = 0;	/* init to all unused */

	for (i=0;i<Ndetectors;i++) {
		XMLattibutes2KeyList("Detector",Detectors,i,&keyVals);
		id = (int)IntByKey("N",keyVals,'=',';');
		CHECK_FREE(keyVals)
		
		detector = XMLtagContents("Detector",Detectors,i);
		if (id>=0 && id <=3) {
			/* detector id description */
			geo->d[id].used = 1;
			if ((str=XMLtagContents("Npixels",detector,0))) {
				if (sscanf(str,"%ld %ld",&(geo->d[id].Nx),&(geo->d[id].Ny))==2) { n = n|1<<1; n = n|1<<2; }
			}
			CHECK_FREE(str);
			
			if ((str=XMLtagContents("size",detector,0))) {
				if (sscanf(str,"%lg %lg",&(geo->d[id].sizeX),&(geo->d[id].sizeY))==2) { n = n|1<<3; n = n|1<<4; }
				geo->d[id].sizeX *= 1000.;					/* file is in mm, I need micron in the geoN structure */
				geo->d[id].sizeY *= 1000.;
			}
			CHECK_FREE(str);
			
			if ((str=XMLtagContents("R",detector,0))) {
				if (!stringTo3Vector(str,geo->d[id].R,1.)) n = n|1<<5;
			}
			CHECK_FREE(str);
			
			if ((str=XMLtagContents("P",detector,0))) {
				if (!stringTo3Vector(str,geo->d[id].P,1e3)) n = n|1<<6;
			}
			CHECK_FREE(str);

			if ((str=XMLtagContents("timeMeasured",detector,0))) strncpy(geo->d[id].timeMeasured,str,READGEON_MAXlen);
			CHECK_FREE(str);
			if ((str=XMLtagContents("geoNote",detector,0))) strncpy(geo->d[id].geoNote,str,READGEON_MAXlen);
			CHECK_FREE(str);
			if ((str=XMLtagContents("ID",detector,0))) strncpy(geo->d[id].detectorID,str,READGEON_MAXlen);
			CHECK_FREE(str);
			if ((str=XMLtagContents("distortionMap",detector,0))) strncpy(geo->d[id].distortionMapFile,str,READGEON_MAXlen);
			CHECK_FREE(str);
		}
		CHECK_FREE(detector);
	}
	CHECK_FREE(Detectors);

	Wire = XMLtagContents("Wire",geoN,0);
	if (Wire) {
		str = XMLtagContents("R",Wire,0);
		if (str) stringTo3Vector(str,geo->wire.R,1.);
		CHECK_FREE(str);
		
		str = XMLtagContents("Axis",Wire,0);
		if (str) stringTo3Vector(str,geo->wire.axis,.1);
		CHECK_FREE(str);
		
		str = XMLtagContents("dia",Wire,0);
		if (str) {
			if (sscanf(str,"%lg",&(geo->wire.dia))==1) n = n|1<<7;
		}
		CHECK_FREE(str);
		
		str = XMLtagContents("F",Wire,0);
		if (str) sscanf(str,"%lg",&(geo->wire.F));
		CHECK_FREE(str);
		
		str = XMLtagContents("Knife",Wire,0);
		if (str) {
			if (sscanf(str,"%d",&(geo->wire.knife))==1) n = n|1<<8;
			}
		CHECK_FREE(str);

		str = XMLtagContents("Origin",Wire,0);
		if (str) {
			if (!stringTo3Vector(str,geo->wire.origin,1.)) n = n|1<<9;
		}
		CHECK_FREE(str);
		CHECK_FREE(Wire);
	}

	Sample = XMLtagContents("Sample",geoN,0);
	if (Sample) {
		str = XMLtagContents("Origin",Sample,0);
		if (str) stringTo3Vector(str,geo->s.O,1.);
		CHECK_FREE(str);
		
		str = XMLtagContents("R",Sample,0);
		if (str) stringTo3Vector(str,geo->s.O,1.);
		CHECK_FREE(str);
		CHECK_FREE(Sample);
	}
	CHECK_FREE(geoN);

	return n;
}


/* the contents of the old $tag value pair geometry file is in buf, put values into the geoStructure, return n, flag showing what was read */
int tagValBuf2GeoN(
char	*buf,										/* string containing xml */
struct geoStructure *geo)
{
	char	line[READGEON_MAXlen+2];				/* line of data read from file */
	int		n;										/* a bit flag used to make sure that essential values have been read */
	
	n = 0;
	/*	1<<2 == 4 */
	if (!strFromTagBuf(buf,"Ndetectors",line,READGEON_MAXlen))	{ geo->Ndetectors = (int)strtol(line,NULL,10); n=n|1<<0; }	/* # of detectors */

	if (!strFromTagBuf(buf,"d0_Nx",line,READGEON_MAXlen))		{ geo->d[0].Nx = strtol(line,NULL,10); n=n|1<<1; }	/* detector 0 description */
	if (!strFromTagBuf(buf,"d0_Ny",line,READGEON_MAXlen))		{ geo->d[0].Ny = strtol(line,NULL,10); n=n|1<<2; }
	if (!strFromTagBuf(buf,"d0_sizeX",line,READGEON_MAXlen))	{ geo->d[0].sizeX = strtod(line,NULL)*1000.; n=n|1<<3; }
	if (!strFromTagBuf(buf,"d0_sizeY",line,READGEON_MAXlen))	{ geo->d[0].sizeY = strtod(line,NULL)*1000.; n=n|1<<4; }
	if (!strFromTagBuf(buf,"d0_R",line,READGEON_MAXlen))		{ if (!stringTo3Vector(line,geo->d[0].R,1.)) n=n|1<<5; }
	if (!strFromTagBuf(buf,"d0_P",line,READGEON_MAXlen))		{ if (!stringTo3Vector(line,geo->d[0].P,1e3)) n=n|1<<6; }
	if (!strFromTagBuf(buf,"d0_timeMeasured",line,READGEON_MAXlen))		{ strncpy(geo->d[0].timeMeasured,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d0_geoNote",line,READGEON_MAXlen))			{ strncpy(geo->d[0].geoNote,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d0_detectorID",line,READGEON_MAXlen))		{ strncpy(geo->d[0].detectorID,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d0_distortionMapFile",line,READGEON_MAXlen))	{ strncpy(geo->d[0].distortionMapFile,line,READGEON_MAXlen); }
	geo->d[0].used = (strlen(geo->d[0].detectorID)>0);

	if (!strFromTagBuf(buf,"d1_Nx",line,READGEON_MAXlen))		{ geo->d[1].Nx = strtol(line,NULL,10); n=n|1<<1; }	/* detector 1 description */
	if (!strFromTagBuf(buf,"d1_Ny",line,READGEON_MAXlen))		{ geo->d[1].Ny = strtol(line,NULL,10); n=n|1<<2; }
	if (!strFromTagBuf(buf,"d1_sizeX",line,READGEON_MAXlen))	{ geo->d[1].sizeX = strtod(line,NULL)*1000.; n=n|1<<3; }
	if (!strFromTagBuf(buf,"d1_sizeY",line,READGEON_MAXlen))	{ geo->d[1].sizeY = strtod(line,NULL)*1000.; n=n|1<<4; }
	if (!strFromTagBuf(buf,"d1_R",line,READGEON_MAXlen))		{ if (!stringTo3Vector(line,geo->d[1].R,1.)) n=n|1<<5; }
	if (!strFromTagBuf(buf,"d1_P",line,READGEON_MAXlen))		{ if (!stringTo3Vector(line,geo->d[1].P,1e3)) n=n|1<<6; }
	if (!strFromTagBuf(buf,"d1_timeMeasured",line,READGEON_MAXlen))		{ strncpy(geo->d[1].timeMeasured,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d1_geoNote",line,READGEON_MAXlen))			{ strncpy(geo->d[1].geoNote,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d1_detectorID",line,READGEON_MAXlen))		{ strncpy(geo->d[1].detectorID,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d1_distortionMapFile",line,READGEON_MAXlen))	{ strncpy(geo->d[1].distortionMapFile,line,READGEON_MAXlen); }
	geo->d[1].used = (strlen(geo->d[1].detectorID)>0);

	if (!strFromTagBuf(buf,"d2_Nx",line,READGEON_MAXlen))		{ geo->d[2].Nx = strtol(line,NULL,10); n=n|1<<1; }	/* detector 2 description */
	if (!strFromTagBuf(buf,"d2_Ny",line,READGEON_MAXlen))		{ geo->d[2].Ny = strtol(line,NULL,10); n=n|1<<2; }
	if (!strFromTagBuf(buf,"d2_sizeX",line,READGEON_MAXlen))	{ geo->d[2].sizeX = strtod(line,NULL)*1000.; n=n|1<<3; }
	if (!strFromTagBuf(buf,"d2_sizeY",line,READGEON_MAXlen))	{ geo->d[2].sizeY = strtod(line,NULL)*1000.; n=n|1<<4; }
	if (!strFromTagBuf(buf,"d2_R",line,READGEON_MAXlen))		{ if (!stringTo3Vector(line,geo->d[2].R,1.)) n=n|1<<5; }
	if (!strFromTagBuf(buf,"d2_P",line,READGEON_MAXlen))		{ if (!stringTo3Vector(line,geo->d[2].P,1e3)) n=n|1<<6; }
	if (!strFromTagBuf(buf,"d2_timeMeasured",line,READGEON_MAXlen))		{ strncpy(geo->d[2].timeMeasured,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d2_geoNote",line,READGEON_MAXlen))			{ strncpy(geo->d[2].geoNote,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d2_detectorID",line,READGEON_MAXlen))		{ strncpy(geo->d[2].detectorID,line,READGEON_MAXlen); }
	if (!strFromTagBuf(buf,"d2_distortionMapFile",line,READGEON_MAXlen))	{ strncpy(geo->d[2].distortionMapFile,line,READGEON_MAXlen); }
	geo->d[2].used = (strlen(geo->d[2].detectorID)>0);

	if (!strFromTagBuf(buf,"wireDia",line,READGEON_MAXlen))		{ geo->wire.dia = strtod(line,NULL); n=n|1<<7; }	/* wire description */
	if (!strFromTagBuf(buf,"wireKnife",line,READGEON_MAXlen))	{ geo->wire.knife = (int)strtol(line,NULL,10); n=n|1<<8; }
	if (!strFromTagBuf(buf,"wireOrigin",line,READGEON_MAXlen))	{ if (!stringTo3Vector(line,geo->wire.origin,1.)) n=n|1<<9; }
	if (!strFromTagBuf(buf,"wireRot",line,READGEON_MAXlen))		{ stringTo3Vector(line,geo->wire.R,1.); }
	if (!strFromTagBuf(buf,"wireAxis",line,READGEON_MAXlen))	{ stringTo3Vector(line,geo->wire.axis,.1); }
	if (!strFromTagBuf(buf,"wireF",line,READGEON_MAXlen))		{ geo->wire.F = strtod(line,NULL); }

	return n;
}


long stringTo3Vector(						/* given a string such as "{123,3.55,17}" fill vec with values */
char	*in,								/* input string */
double	vec[3],
double	factor)								/* multiply result by factor, use 1 to do nothing */
{
	double x,y,z;
	int i;

	i = (sscanf(in,"{%lg, %lg, %lg}",&x,&y,&z) != 3);
	if (i) { i = (sscanf(in,"%lg %lg %lg",&x,&y,&z) != 3); }

	if (i==0) {
		vec[0] = x*factor;
		vec[1] = y*factor;
		vec[2] = z*factor;
	}
	return i;
}


/* returns value of associated with a tag from a buffer, each tag value pair is terminated by a new line
   and tag is separated from its value by white space.  anything after a // is ignored, and can be used as comments in the file
   tags are limited to 120 characters, and the tag+value+comment is limited to 500 characters */
int strFromTagBuf(	/* look for the tag and return 0 if it finds the tag, 1 if not found */
char	*buffer,	/* char buffer with all of the tagged strings */
char	*tagIn,		/* the tag, this should NOT have the leading '$' */
char	*value,		/* put the rest of the line following tag here, line should be at least maxLen long */
long	maxLen)		/* max length of line to pass back */
{
	char	tag[128];		/* local version of tag with leading $ */
	size_t	tagLen;			/* length of tag */
	char	*p;				/* generic pointer for string stuff */
	long	i;				/* generic index */

	tagLen = strlen(tagIn);
	if (tagLen<1 || tagLen>120) return 1;			/* empty tag is not allowed, not too long either */
	for (p=tagIn;*p>0;p++) if(*p <= ' ') return 1;	/* tag may not contain control characters or spaces */
	strcpy(tag,"$");
	strncat(tag,tagIn,125);							/* now tag[] has the leading '$' */
	tagLen++;

	/* search buffer for $tag followed by character <= 32, also tag must start a line */
	p = buffer;
	while((p=strstr(p,tag))) {						/* find $tag, then check for white space after */
		if (!p) {value=NULL; return 1; }			/* NULL pointer, $tag was not found */
/*		if (p!=buffer && *(p-1)!='\n') continue;	// check if $tag at start or preceeded by new line */
		if (p!=buffer && *(p-1)!='\n') {p++; continue;}	/* check if $tag at start or preceeded by new line */
		p += tagLen;								/* points to first character after $tag */
		if (*p <= ' ') break;						/* $tag followed by white space, found it */
	}
	if (!p) { value[0]='\0'; return 0; }			/* $tag not found */

	/* p now points to first character after $tag */
	i = 0;
	while (*p>0 && *p<=' ' && *p!='\n' && i<500) { p++; i++; }	/* find the start of the value, skip white space */
	if (*p=='\n' || *p==0 || i>=500) { value[0]='\0'; return 0; } /* the tag was there, but no value part found */

	strncpy(value,p,(size_t)(maxLen-1));			/* now value contains the value part */
	value[maxLen-1] = '\0';							/* and it is definitely terminated */
	if ((p=strchr(value,'\n'))) *p = '\0';			/* trim at new line */
	if ((p=strstr(value,"//"))) *p = '\0';			/* strip off any following comment */

	/* trim off any trailing white space */
	p = value+strlen(value)+1;						/* p points to null after last character */
	while(p>value && *(p-1)<=' ') p--;
	*p = '\0';
	return 0;
}

/* returns value of associated with a tag from a file, each tag value pair is terminated by a new line
   and tag is separated from its value by white space.  anything after a // is ignored, and can be used as comments in the file
   tags are limited to 120 characters, and the tag+value is limited to 500 characters 
   leaves file pointer at start of next line (the line after the one with the tag) */
int strFromTagFile(	/* look for the tag and return 0 if it finds the tag, 1 if not found */
FILE	*f,			/* file id */
char	*tagIn,		/* the tag, this should NOT have the leading '$' */
char	*value,		/* put the rest of the line following tag here, line should be at least maxLen long */
long	maxLen)		/* max length of line to pass back */
{
	char	line[512];		/* contains one line from the file */
	char	tag[128];		/* local version of tag with leading $ */
	size_t	tagLen;			/* length of tag */
	char	*p;				/* generic pointer for string stuff */
	long	i;				/* generic index */

	tagLen = strlen(tagIn);
	if (tagLen<1 || tagLen>120) return 1;			/* empty tag is not allowed, not too long either */
	for (p=tagIn;*p>0;p++) if(*p <= ' ') return 1;	/* tag may not contain control characters or spaces */
	strcpy(tag,"$");
	strncat(tag,tagIn,125);							/* now tag[] has the leading '$' */
	tagLen++;

	fseek(f,0,SEEK_SET); 							/* always start searching from start of file */
	while ((p=fgets(line,511,f))) {					/* break on EOF */
		if (strncmp(line,tag,tagLen)) continue;		/* wrong tag, continue checking */
		else if (line[tagLen]>' ') continue;		/* the tag in line is longer than tag[] */
		p = &(line[tagLen]);						/* points to start of value part */
		break;
	}
	if (!p) {value[0]='\0'; return 1; }				/* NULL pointer, this mainly catches the EOF */

	i = tagLen;
	while (*p>0 && *p<=' ' && i<500) { p++; i++; }	/* find the start of the value, skip white space */
	if (*p==0 || i>=500) { value[0]='\0'; return 0; } /* the tag was there, but no value part found */

	strncpy(value,p,(size_t)(maxLen-1));			/* now value contains the value part */
	value[maxLen-1] = '\0';							/* and it is terminated */
	if ((p=strstr(value,"//"))) *p = '\0';			/* strip off any following comment */

	/* trim off any trailing white space */
	p = value+strlen(value)+1;						/* p points to null after last character */
	while(p>value && *(p-1)<=' ') p--;
	*p = '\0';
	return 0;
}



/* first check if this is the correct type of file.  There are two ways to check.
 *	1:  if the first line starts with $type then it is OK
 *	2:  if the first line has a tag of $filetype, and one of the values in the following semi-colon separated list is type then it is OK
 *	lineIn and type are unchanged
 */
int checkFileTypeOLD(								/* return true if file is of type 'type' */
char	*lineIn,									/* first line of a file (check for $type in here) */
size_t	Nline,										/* max length of lineIn */
char	*type)										/* desired file type */
{
	char	*p;										/* generic pointer into a string */
	char	typeList[256];							/* holds value of $filetype, a list of file types read from file */
	size_t	maxLen=251;								/* maximum length of typeList[] */
	size_t	i;										/* generic index */
	char	localtype[256];							/* local version of type */
	char	line[512];								/* local version of lineIn */

	Nline = MIN(Nline,strlen(lineIn));				/* maximum number of characters in line to consider */
	if (lineIn[0]!='$' || Nline<9) return(0);		/* completely wrong type of file */
	if (strlen(type)>maxLen-1) return(0);			/* type is too long */

	Nline = MIN(511,Nline);
	strncpy(line,lineIn,(size_t)Nline);				/* need a local copy because we will change it (using p) */
	line[Nline] = '\0';

	strcpy(localtype,"$");
	strncat(localtype,type,(size_t)(maxLen-4));
	i = strlen(localtype);
	if (Nline<i) return(0);
	if (!strncmp(line,localtype,i)) return(1);		/* (1) check for the correct type of file, =0 means starts with $type */

	if (strncmp(line,"$filetype",9)) return(0);		/* definitely wrong kind of file, does not start with "$filetype" either  */
	p = line + 9;									/* points to first character after the tag */
	i = 9;
	while (*p>0 && *p<=' ' && i<Nline) { p++; i++; }/* find the start of the typeList, skip white space */
	if (*p==0 || i>=Nline) return(0);				/* the tag was there, but no typeList part found */

	typeList[0] = (*p != ';') ? ';' : '\0';			/* ensure that typeList will start with exactly one semi-colon */
	typeList[1] = '\0';
	strncat(typeList,p,(size_t)(maxLen-1));			/* now typeList contains the typeList part with a leading semi-colon */
	typeList[maxLen] = '\0';						/* and it is definitely terminated */
	if ((p=strstr(typeList,"//"))) *p = '\0';			/* strip off any following comment */

	/* trim off any trailing white space */
	p = typeList+strlen(typeList)+1;				/* p points to null after last character */
	while(p>typeList && *(p-1)<=' ') p--;
	if (*p != ';') { *p = ';';	p++; }				/* ensure one trailing semi-colon */
	*p = '\0';										/* and terminate */
	/* typeList is now a semi-colon separated list of filetypes, where it is guaranteed to have both leading and trailing semi-colons */
	sprintf(localtype,";%s;",type);
	if (!strstr(typeList,localtype)) return(0);		/* if type not in value of $filetype list, then wrong type of file */
	return(1);										/* the right kind of file */
}





/* calculate rotation matrix from detector tilts angles */
void GeometryStructureUpdate(						/* update all internally calculated things in the structure */
struct geoStructure *geo)
{
	int		i, N=0;

	if (geo->Ndetectors>MAX_Ndetectors) {
		printf("\nERROR, geo0->Ndetectors is %d, which is bigger than max value of %d. Reducing it\n",geo->Ndetectors,MAX_Ndetectors);
		geo->Ndetectors = MAX_Ndetectors;
	}
	SampleUpdateCalc(&(geo->s));					/* update all internally calculated things in the sample structure */
	WireUpdateCalc(&(geo->wire));					/* update all internally calculated things in the wire structure */
	for (i=0;i<MAX_Ndetectors;i+=1) {
		if (geo->d[i].used) {
			DetectorUpdateCalc(&(geo->d[i]));		/* update all internally calculated things in the detector structures */
			N += 1;
		}
	}
	geo->Ndetectors = N;
}



int DetectorUpdateCalc(								/* update all internally calculated things in the detector structure */
struct detectorGeometry *d)
{
	double Rx, Ry, Rz;								/* used to make the rotation matrix rho from vector R */
	double theta, c, s, c1;

	if (!(d->used)) return 1;

	Rx=d->R[0]; Ry=d->R[1]; Rz=d->R[2];				/* make the rotation matrix rho from vector R */
	theta = sqrt(Rx*Rx+Ry*Ry+Rz*Rz);
	if (theta != theta || theta==0) {				/* no rotation, set to identity matrix */
		d->rho00 = 1;		d->rho01 = 0;		d->rho02 = 0;
		d->rho10 = 0;		d->rho11 = 1;		d->rho12 = 0;
		d->rho20 = 0;		d->rho21 = 0;		d->rho22 = 1;
		return 0;
	}
	c=cos(theta);
	s=sin(theta);
	c1 = 1-c;
	Rx /= theta;	Ry /= theta;	Rz /= theta;	/* make |{Rx,Ry,Rz}| = 1 */

	d->rho00 = c + Rx*Rx*c1;		d->rho01 = Rx*Ry*c1 - Rz*s;	d->rho02 = Ry*s + Rx*Rz*c1;		/* this is the Rodrigues formula from: */
	d->rho10 = Rz*s + Rx*Ry*c1;		d->rho11 = c + Ry*Ry*c1;	d->rho12 = -Rx*s + Ry*Rz*c1;	/* http://mathworld.wolfram.com/RodriguesRotationFormula.html */
	d->rho20 = -Ry*s + Rx*Rz*c1;	d->rho21 = Rx*s + Ry*Rz*c1;	d->rho22 = c + Rz*Rz*c1;
	return 0;
}




void SampleUpdateCalc(								/* update all internally calculated things in the structure */
struct sampleGeometry *sa)
{
	double Rx, Ry, Rz;								/* used to make the rotation matrix rho from vector R */
	double theta, c, s, c1;
	Rx=sa->R[0];	Ry=sa->R[1];	Rz=sa->R[2];
	theta = sqrt(Rx*Rx+Ry*Ry+Rz*Rz);				/* angle in radians */

	if (theta!=theta || theta==0) {					/* rotation of sample stage */
		sa->R[0] = 0;	sa->R[1] = 0;	sa->R[2] = 0;
		sa->Rmag = 0;
		sa->R00 = 1;	sa->R01 = 0;	sa->R02 = 0;/* matrix is identity */
		sa->R10 = 0;	sa->R11 = 1;	sa->R12 = 0;
		sa->R20 = 0;	sa->R21 = 0;	sa->R22 = 1;
	}
	else {
		sa->Rmag = theta * 180/M_PI;
		c=cos(theta);
		s=sin(theta);
		c1 = 1-c;
		Rx /= theta;	Ry /= theta;	Rz /= theta;/* make |{Rx,Ry,Rz}| = 1 */
		sa->R00 = c + Rx*Rx*c1;		sa->R01 = Rx*Ry*c1 - Rz*s;	sa->R02 = Ry*s + Rx*Rz*c1;		/* this is the Rodrigues formula from: */
		sa->R10 = Rz*s + Rx*Ry*c1;	sa->R11 = c + Ry*Ry*c1;		sa->R12 = -Rx*s + Ry*Rz*c1;	/* http://mathworld.wolfram.com/RodriguesRotationFormula.html */
		sa->R20 = -Ry*s + Rx*Rz*c1;	sa->R21 = Rx*s + Ry*Rz*c1;	sa->R22 = c + Rz*Rz*c1;
	}
}


void WireUpdateCalc(								/* update parts of the wire structure */
struct wireGeometry *w)
{
	double Rx, Ry, Rz;								/* used to make the rotation matrix rho from vector R */
	double theta, c, s, c1;							/* used to make rotation matrix from R */
	double xx, yy, zz;
	double len;

	/* normalize wire.axis */
	len = sqrt(w->axis[0]*w->axis[0] + w->axis[1]*w->axis[1] + w->axis[2]*w->axis[2]);
	if (len==len && len>0) {
		w->axis[0] /= len ;		w->axis[1] /= len;		w->axis[2] /= len;
	}
	else {
		w->axis[0] = 1 ; 		w->axis[1] = w->axis[2] = 0;
	}

	/* fix up the rotation of wire frame */
	Rx=w->R[0];	Ry=w->R[1];	Rz=w->R[2];
	theta = sqrt(Rx*Rx+Ry*Ry+Rz*Rz);				/* angle in radians */

	if (w->Rmag != w->Rmag || theta==0) {			/* rotation of wire stage */
		w->R[0] = 0;	w->R[1] = 0;	w->R[2] = 0;
		w->Rmag = 0;
		w->R00 = 1;		w->R01 = 0;		w->R02 = 0;	/* matrix is identity */
		w->R10 = 0;		w->R11 = 1;		w->R12 = 0;
		w->R20 = 0;		w->R21 = 0;		w->R22 = 1;
		w->axisR[0] = w->axis[0];					/* both axes the same, no rotation */
		w->axisR[1] = w->axis[1];
		w->axisR[2] = w->axis[2];
	}
	else {
		w->Rmag = theta * 180/M_PI;
		c=cos(theta);
		s=sin(theta);
		c1 = 1-c;
		Rx /= theta;	Ry /= theta;	Rz /= theta;/* make |{Rx,Ry,Rz}| = 1 */
		w->R00 = c + Rx*Rx*c1;		w->R01 = Rx*Ry*c1 - Rz*s;	w->R02 = Ry*s + Rx*Rz*c1;	/* this is the Rodrigues formula from: */
		w->R10 = Rz*s + Rx*Ry*c1;	w->R11 = c + Ry*Ry*c1;		w->R12 = -Rx*s + Ry*Rz*c1;	/* http://mathworld.wolfram.com/RodriguesRotationFormula.html */
		w->R20 = -Ry*s + Rx*Rz*c1;	w->R21 = Rx*s + Ry*Rz*c1;	w->R22 = c + Rz*Rz*c1;

		xx=w->axis[0]; yy=w->axis[1]; zz=w->axis[2];
		w->axisR[0] = w->R00*xx + w->R01*yy + w->R02*zz;	/* axisR = w->Rij x axis,   rotate by R (a small rotation) */
		w->axisR[1] = w->R10*xx + w->R11*yy + w->R12*zz;
		w->axisR[2] = w->R20*xx + w->R21*yy + w->R22*zz;

		/* normalize wire.axisR */
		len = sqrt(w->axisR[0]*w->axisR[0] + w->axisR[1]*w->axisR[1] + w->axisR[2]*w->axisR[2]);
		if (len==len && len>0) {
			w->axisR[0] /= len ;		w->axisR[1] /= len;			w->axisR[2] /= len;
		}
		else {
			w->axisR[0] = w->axis[0];	w->axisR[1] = w->axis[1];	w->axisR[2] = w->axis[2];
		}
	}
}




void printGeometry(										/* print the details for passed geometry */
FILE	*f,												/* file descriptor, use stdout to get this on the screen */
struct	geoStructure *geo)
{
	int i;

	if (f == NULL) f = stdout;

	if (f == stdout) {
		fprintf(f,"current geomery parameters  (using %d detectors)\n",geo->Ndetectors);
		if (!SampleBad(&(geo->s))) {
			fprintf(f,"Sample\n");
			fprintf(f,"	Origin = {%g, %g, %g}					// PM500 coordinate frame where sample is at origin (micron)\n",geo->s.O[0],geo->s.O[1],geo->s.O[2]);
			fprintf(f,"	R = {%g, %g, %g}	// (= %g°) sample positioner rotation vector (radian)\n",geo->s.R[0],geo->s.R[1],geo->s.R[2],geo->s.Rmag);
			#ifdef VERBOSE
				fprintf(f,"			{%+.6f, %+.6f, %+.6f}	// rotation matrix from sample R\n",geo->s.R00, geo->s.R01, geo->s.R02);
				fprintf(f,"	Rs =	{%+.6f, %+.6f, %+.6f}\n",geo->s.R10, geo->s.R11, geo->s.R12);
				fprintf(f,"			{%+.6f, %+.6f, %+.6f}\n",geo->s.R20, geo->s.R21, geo->s.R22);
			#endif
		}
		for (i=0;i<MAX_Ndetectors;i+=1) {				/* info about all of the detectors */
			if (geo->d[i].used) {
				fprintf(f,"Detector %d\n",i);
				printDetector(f,&(geo->d[i]));
			}
		}
		if (WireBad(&(geo->wire))) fprintf(f,"Wire UN-Defined *****\n");		/* info about the wire */
		else
		{
			fprintf(f,"Wire:\n");						/* info about the wire */
			fprintf(f,"	Origin = {%.2f, %.2f, %.2f}	// PM500 coordinate frame to put wire at Origin (Si position) (µm)\n",geo->wire.origin[0],geo->wire.origin[1],geo->wire.origin[2]);
			fprintf(f,"	diameter=%.2f				// diameter of wire (µm)\n",geo->wire.dia);
			if (geo->wire.knife) fprintf(f,"\t wire mounted on a knife edge");
			else fprintf(f,"\tfree standing wire");
			fprintf(f,"	wire axis direction = {%.6f, %.6f, %.6f}	// direction of wire axis in PM500 wire coordinates (µm)\n",geo->wire.axis[0],geo->wire.axis[1],geo->wire.axis[2]);
			if (geo->wire.Rmag > 0) {
				fprintf(f,"	R = {%g, %g, %g}, a rotation of %g°	// wire positioner rotation vector\n",geo->wire.R[0],geo->wire.R[1],geo->wire.R[2],geo->wire.Rmag);
				#ifdef VERBOSE
					fprintf(f,"			{%+.6f, %+.6f, %+.6f}	// rotation matrix from wire Rw\n",geo->wire.R00, geo->wire.R01, geo->wire.R02);
					fprintf(f,"	Rw =	{%+.6f, %+.6f, %+.6f}\n",geo->wire.R10, geo->wire.R11, geo->wire.R12);
					fprintf(f,"			{%+.6f, %+.6f, %+.6f}\n",geo->wire.R20, geo->wire.R21, geo->wire.R22);
					fprintf(f,"	wire axis direction = {%.6f, %.6f, %.6f}	// direction of wire axis in Beam LIne coordinates (µm)\n",geo->wire.axisR[0],geo->wire.axisR[1],geo->wire.axisR[2]);
				#endif
			}
			fprintf(f,"	F=%.2f						// F of wire for a constant F wire scan, raw PM500 units (µm)\n",geo->wire.F);
		}
	}
	else {											/* formatted to look like a geometry file */
		char pre[100];
		if (!SampleBad(&(geo->s))) {
			fprintf(f,"\n// Sample\n");
			fprintf(f,"$SampleOrigin	{%.2f,%.2f,%.2f}			// sample origin in PM500 frame (micron)\n",geo->s.O[0],geo->s.O[1],geo->s.O[2]);
			fprintf(f,"$SampleRot		{%.8f,%.8f,%.8f}	// sample positioner rotation vector (length is angle in radians)\n",geo->s.R[0],geo->s.R[1],geo->s.R[2]);
		}
		fprintf(f,"\n// Detectors\n");
		fprintf(f,"$Ndetectors		%d							// number of detectors, must be <= MAX_Ndetectors\n",geo->Ndetectors);
		for (i=0;i<MAX_Ndetectors;i+=1) {
			if (!(geo->d[i].used)) continue;
			sprintf(pre,"d%d_",i);
			fprintf(f,"\n");
			fprintf(f,"$%sNx			%ld						// number of un-binned pixels in full detector\n",pre,geo->d[i].Nx);
			fprintf(f,"$%sNy			%ld\n",pre,geo->d[i].Ny);
			fprintf(f,"$%ssizeX		%.3f						// size of detector (mm)\n",pre,(geo->d[i].sizeX)/1000);
			fprintf(f,"$%ssizeY		%.3f\n",pre,(geo->d[i].sizeY/1000));
			fprintf(f,"$%sR			{%.8f,%.8f,%.8f}	// rotation vector (length is angle in radians)\n",pre,geo->d[i].R[0],geo->d[i].R[1],geo->d[i].R[2]);
			fprintf(f,"$%sP			{%.3f,%.3f,%.3f}		// translation vector (mm)\n",pre,(geo->d[i].P[0]/1000),(geo->d[i].P[1]/1000),(geo->d[i].P[2]/1000));
			if (strlen(geo->d[i].timeMeasured)) fprintf(f,"$%stimeMeasured	%s	// when this geometry was calculated\n",pre,geo->d[i].timeMeasured);
			if (strlen(geo->d[i].geoNote)) fprintf(f,"$%sgeoNote	%s\n",pre,geo->d[i].geoNote);
			if (strlen(geo->d[i].detectorID)) fprintf(f,"$%sdetectorID	%s			// unique detector ID\n",pre,geo->d[i].detectorID);
			if (strlen(geo->d[i].distortionMapFile)) fprintf(f,"$%sdistortionMapFile	%s			// name of file with distortion map\n",pre,geo->d[i].distortionMapFile);
		}
		if (!WireBad(&(geo->wire))) {
			fprintf(f,"\n// Wire\n");
			fprintf(f,"$wireDia		%.2f						// diameter of wire (micron)\n",geo->wire.dia);
			fprintf(f,"$wireKnife		%d							// true if wire on a knife edge, false for free-standing wire\n",geo->wire.knife);
			fprintf(f,"$wireOrigin		{%.2f,%.2f,%.2f}	// wire origin in PM500 frame (micron)\n",geo->wire.origin[0],geo->wire.origin[1],geo->wire.origin[2]);
			if (geo->wire.Rmag>0) fprintf(f,"$wireRot		{%.8f,%.8f,%.8f}	// wire positioner rotation vector (length is angle in radians)\n",geo->wire.R[0],geo->wire.R[1],geo->wire.R[2]);
			fprintf(f,"$wireAxis		{%.6f,%.6f,%.6f}	// unit vector along wire axis, usually close to (1,0,0)\n",geo->wire.axis[0],geo->wire.axis[1],geo->wire.axis[2]);
			fprintf(f,"$wireF			%.2f						// F of wire for a constant F wire scan, raw PM500 units (µm)\n",geo->wire.F);
		}
	}
}

int printDetector(									/* print the details for passed detector geometry to the history window */
FILE	*f,											/* file descriptor, use stdout to get this on the screen */
struct detectorGeometry *d)
{
	if (!(d->used)) return 1;

	fprintf(f,"	Nx=%ld, Ny=%ld			// number of un-binned pixels in detector\n",d->Nx,d->Ny);
	fprintf(f,"	sizeX=%g, sizeY=%g		// size of detector (mm)\n",(d->sizeX/1000), (d->sizeY/1000));
	fprintf(f,"	R = {%.7g, %.7g, %.7g}, a rotation of %.7g°	// rotation vector\n",d->R[0],d->R[1],d->R[2],sqrt(d->R[0]*d->R[0] + d->R[1]*d->R[1] + d->R[2]*d->R[2])*180/M_PI);
	fprintf(f,"	P = {%g, %g, %g}					// translation vector (mm)\n",(d->P[0])/1000,(d->P[1])/1000,(d->P[2])/1000);

	fprintf(f,"	geometry measured on  '%s'\n",d->timeMeasured);
	if (strlen(d->geoNote)) fprintf(f,"	detector note = '%s'\n",d->geoNote);
	if (strlen(d->distortionMapFile)) fprintf(f,"	detector distortion file = '%s'\n",d->distortionMapFile);
	fprintf(f,"	detector ID = '%s'\n",d->detectorID);
	#ifdef VERBOSE
		fprintf(f,"			{%+.6f, %+.6f, %+.6f}	// rotation matrix from R\n",d->rho00, d->rho01, d->rho02);
		fprintf(f,"	rho =	{%+.6f, %+.6f, %+.6f}\n",d->rho10, d->rho11, d->rho12);
		fprintf(f,"			{%+.6f, %+.6f, %+.6f}\n",d->rho20, d->rho21, d->rho22);
	#endif
	return 0;
}



int MicroGeometryBad(							/* check for a valid or Invalid structure */
struct geoStructure *g)							/* f is the destination, i is source */
{
	int	bad;
	long N;
	long m, i;

	bad = SampleBad(&(g->s));
	N = g->Ndetectors;							/* Ndetectors must be 1, 2, or 3 */
	N = N<1 ? 1 : N;
	N = N>3 ? 3 : N;
	bad += (g->Ndetectors != N) || (g->Ndetectors < 1);
	for (m=0,i=0;m<MAX_Ndetectors;m+=1) {		/* check the detectors */
		if (g->d[m].used) {
			bad += DetectorBad(&(g->d[m]));
			i += 1;
		}
	}
	bad += (i != N);
	bad += WireBad(&(g->wire));					/* check the wire */
	return (bad>0);
}

int DetectorBad(
struct detectorGeometry *d)
{
	int bad=0;
	double value;

	if (!(d->used)) return 1;
	value = d->sizeX + d->sizeY + d->R[0] + d->R[1] + d->R[2] + d->P[0] + d->P[1] + d->P[2];
	bad += (value != value);
	bad = (d->Nx<1 || d->Ny<1);
	bad += (d->Nx<1 || d->Nx>5000);												/* detector cannot have more than 5000 pixels along one edge */
	bad += (d->Ny<1 || d->Ny>5000);
	bad += (d->sizeX<1e3 || d->sizeX>1e6);										/* detector cannot be larger than 1m */
	bad += (d->sizeY<1e3 || d->sizeY>1e6);
	bad += (fabs(d->R[0])>2*M_PI || fabs(d->R[1])>2*M_PI || fabs(d->R[2])>2*M_PI);	/* rotation cannot be more than 2π */
	bad += (fabs(d->P[0])>2e6 || fabs(d->P[0])>2e6 || fabs(d->P[0])>2e6);		/* P cannot be more than 2m in any direction */
	return (bad>0);
}

int WireBad(
struct wireGeometry *w)
{
	double sxyz;
	int bad;

	sxyz = fabs(w->origin[0])+fabs(w->origin[1])+fabs(w->origin[2]);
	bad = (sxyz!=sxyz) || fabs(sxyz)>50e3;
	bad += (w->dia<5 || w->dia>1000);
	bad += !(w->knife==0 || w->knife==1);										/* knife must be 0 or 1 */
	bad += (fabs(w->axis[0])+fabs(w->axis[1])+fabs(w->axis[2])) > 2;
	return (bad>0);
}

int SampleBad(
struct sampleGeometry *s)						/* sample strucure */
{
	double sxyz, value;
	int bad;

	sxyz = fabs(s->O[0])+fabs(s->O[1])+fabs(s->O[2]);
	bad = (sxyz!=sxyz) || fabs(sxyz)>50e3;
	value = ((s->R[0] + s->R[1] + s->R[2]) > 0);
	bad += (value!=value);
	return (bad>0);
}


void copyMicroGeometryStructure(				/* copy all information from 'in' to 'dest' */
struct geoStructure *dest,						/* destination structure */
struct geoStructure *in)						/* input structure */
{
	int i;
	dest->Ndetectors = in->Ndetectors;
	for (i=0;i<MAX_Ndetectors;i++) {
		copyDetectorGeometry(&(dest->d[i]),&(in->d[i]));	/* copy geometry parameters for each detector */
	}
	copySampleGeometry(&(dest->s), &(in->s));	/* copy Sample geometry */
	copyWireGeometry(&(dest->wire),&(in->wire));/* copy the wire */
}

void copySampleGeometry(						/* copy a Sample geometry structure, set f = i */
struct sampleGeometry *f,						/* destination structure */
struct sampleGeometry *i)						/* input structure */
{
	f->O[0] = i->O[0];		f->O[1] = i->O[1];		f->O[2] = i->O[2];
	f->R[0] = i->R[0];		f->R[1] = i->R[1];		f->R[2] = i->R[2];
	f->Rmag = i->Rmag;
	f->R00=i->R00;			f->R01=i->R01;			f->R02=i->R02;
	f->R10=i->R10;			f->R11=i->R11;			f->R12=i->R12;
	f->R20=i->R20;			f->R21=i->R21;			f->R22=i->R22;
}

void copyWireGeometry(							/* copy a wire geometry structure, set f = i */
struct wireGeometry *f,							/* destination structure */
struct wireGeometry *i)							/* input structure */
{
	f->origin[0]=i->origin[0];	f->origin[1]=i->origin[1];	f->origin[2]=i->origin[2];
	f->F = i->F;
	f->dia = i->dia;
	f->knife = i->knife;
	f->axis[0]=i->axis[0];		f->axis[1]=i->axis[1];		f->axis[2]=i->axis[2];
	f->axisR[0]=i->axisR[0];	f->axisR[1]=i->axisR[1];	f->axisR[2]=i->axisR[2];
	f->R[0] = i->R[0];			f->R[1] = i->R[1];			f->R[2] = i->R[2];
	f->Rmag = i->Rmag;
	f->R00=i->R00;				f->R01=i->R01;				f->R02=i->R02;
	f->R10=i->R10;				f->R11=i->R11;				f->R12=i->R12;
	f->R20=i->R20;				f->R21=i->R21;				f->R22=i->R22;
}

void copyDetectorGeometry(						/* copy a detector structure */
struct detectorGeometry *f,						/* destination structure */
struct detectorGeometry *i)						/* input structure */
{
	f->used = i->used;
	f->Nx = i->Nx;			f->Ny = i->Ny;
	f->sizeX = i->sizeX;	f->sizeY = i->sizeY;
	f->R[0] = i->R[0];		f->R[1] = i->R[1];			f->R[2] = i->R[2];
	f->P[0] = i->P[0];		f->P[1] = i->P[1];			f->P[2] = i->P[2];

	strncpy(f->timeMeasured,i->timeMeasured,100);
	strncpy(f->geoNote,i->geoNote,100);
	strncpy(f->detectorID,i->detectorID,100);
	strncpy(f->distortionMapFile,i->distortionMapFile,100);

	f->rho00 = i->rho00;	f->rho01 = i->rho01;		f->rho02 = i->rho02;
	f->rho10 = i->rho10;	f->rho11 = i->rho11;		f->rho12 = i->rho12;
	f->rho20 = i->rho20;	f->rho21 = i->rho21;		f->rho22 = i->rho22;
}

