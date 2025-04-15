#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <math.h> 
#include <sys/stat.h>
#include "readGeoN.h"
#include "checkFileType.h"
#include "lattice.h"
#include "xmlUtility.h"
#include "keyLists.h"

#ifndef MAX_FILE_LENGTH
#define MAX_FILE_LENGTH 2048
#endif

#define EXIT_WITH_HELP { for(i=0;help[i][0];i++) fprintf(stderr,"%s\n",help[i]); exit(1); }


int readCrystalFile(char *fname, struct crystalStructure *xtal);
char *readCrystalFileHeader(char *fname);
char *readPixelPeakFileHeader(char *fname);
//double *pixels2qhatsTable(struct geoStructure *geo, char *buf);
double *pixels2qhatsTable(struct detectorGeometry *d, char *buf);
double pixel2q(struct detectorGeometry *d, double px, double py, double depth, double *qx, double *qy, double *qz);
void pixel2XYZ(struct detectorGeometry *d, double px, double py, double *x, double *y, double *z);
void writeOutput(long Nq, double *qhats, char *xtalHeaderBuf, char *pixelHeaderBuf, char *geoFile, char *xtalFile, char *outName, const char *pgm);
int detectorIDfromGeo(struct geoStructure *geo, char *detector_ID);

int main (int argc, const char * argv[]) {
	char	geoFile[MAX_FILE_LENGTH+1]="geometry.txt";
	char	xtalFile[MAX_FILE_LENGTH+1];
	char	peaksFile[MAX_FILE_LENGTH+1];
	char	outFile[MAX_FILE_LENGTH+1];
	struct geoStructure geo;
	char	*xtalHeaderBuf=NULL;			/* string with tag values to pass on to the output flie */
	char	*pixelHeaderBuf=NULL;			/* string with tag values to pass on to the output flie */
	double	*qhats=NULL;
	long	Npeaks=-1;
	int		detectorID;						/* detector ID number [0,1,2,3] */
	char	line[512];
	long	i;
	static char *help[] = {"pixels2qs [-g geometryFile] -x crystalDescriptionFile input_peaks_file output_qs_file",
							"\tinput_peaks_file    result from peak search", "\toutput_qs_file      name for new output file that holds the result",
							"\tswitches are:","\t\t-g geometry file (defaults to geometry.txt)", "\t\t-x crystal description file", "" };

	strncpy(geoFile,"geometry.txt",MAX_FILE_LENGTH);
	xtalFile[0] = peaksFile[0] = outFile[0] = '\0';
	for (i=1;i<argc;i++) {
		if (!strncmp(argv[i],"-g",2)) {
			if ((++i)>=argc || argv[i][0]=='-') { fprintf(stderr,"ERROR: -g not follwed by the geometry file name \n"); EXIT_WITH_HELP }
			if (strlen(argv[i])<1) { fprintf(stderr,"ERROR: -g cannot read name of geometry file from argv[i]='%s'\n",argv[i]); EXIT_WITH_HELP }
			strncpy(geoFile,argv[i],MAX_FILE_LENGTH);
			continue;
		}
		else if (!strncmp(argv[i],"-x",2)) {
			if ((++i)>=argc || argv[i][0]=='-') { fprintf(stderr,"ERROR: -x not follwed by name of the crystal file\n"); EXIT_WITH_HELP }
			if (strlen(argv[i])<1) { fprintf(stderr,"ERROR: -x cannot readd name of crystal file argv[i]='%s'\n",argv[i]); EXIT_WITH_HELP }
			strncpy(xtalFile,argv[i],MAX_FILE_LENGTH);
			continue;
		}
		if (strlen(peaksFile)==0) { strncpy(peaksFile,argv[i],MAX_FILE_LENGTH); continue; }	/* first arg is input file */
		if (strlen(outFile)==0) { strncpy(outFile,argv[i],MAX_FILE_LENGTH); continue; }		/* second arg is output file */

		EXIT_WITH_HELP							/* there should be nothing else */
	}
	if (!geoFile[0] || !xtalFile[0] || !peaksFile[0] || !outFile[0]) {
		fprintf(stderr,"ERROR: missing input files\n"); 
		EXIT_WITH_HELP
	}

	if (readGeoFromFile(geoFile, &geo)) { fprintf(stderr,"ERROR: unable to read geometry from file '%s'\n",geoFile); exit(1); }
	xtalHeaderBuf = readCrystalFileHeader(xtalFile);		/* get the lattice information from a 'CrystalStructure' file */
	pixelHeaderBuf = readPixelPeakFileHeader(peaksFile);	/* get the fitted peak positions */
	#ifdef DEBUG
	printf("peaksIn = '%s'\n",pixelHeaderBuf);
	#endif
	if (!pixelHeaderBuf) exit(1);

	if (!strFromTagBuf(pixelHeaderBuf,"Npeaks",line,500)) Npeaks = atoi(line);
	if (Npeaks<1) {fprintf(stderr,"ERROR: no peaks in file '%s'\n",peaksFile); exit(1); }
	if (strFromTagBuf(pixelHeaderBuf,"detector_ID",line,500)) line[0]='\0';
	detectorID = detectorIDfromGeo(&geo,line);

	#ifdef DEBUG
	printf("detectorID = '%d',  using line = '%s'\n",detectorID,line);
	printf("detector.used = %d, %d, %d\n",geo.d[0].used,geo.d[1].used,geo.d[2].used);
	#endif
	
	qhats = pixels2qhatsTable(&(geo.d[detectorID]), pixelHeaderBuf);		/* convert each pixel to qhat */
	if (!qhats) {fprintf(stderr,"ERROR: cannot form pixel list from contents of file '%s'\n",peaksFile); exit(1); }

	writeOutput(Npeaks,qhats,xtalHeaderBuf,pixelHeaderBuf,geoFile,xtalFile,outFile,argv[0]);	/* write the 'PeaksFile' file suitable for indexing */
	CHECK_FREE(xtalHeaderBuf);
	CHECK_FREE(pixelHeaderBuf);
	CHECK_FREE(qhats);
	return 0;
}



void writeOutput(
long	Nq,					/* number of qhats to write */
double	*qhats,				/* list of qhats, with intensity */
char	*xtalHeaderBuf,		/* string with tag values to pass on to the output flie */
char	*pixelHeaderBuf,	/* string with tag values to pass on to the output flie */
char	*geoFile,			/* name of file that provided the geometry */
char	*xtalFile,			/* name of the xtl file */
char	*outName,			/* name of output file */
const char	*pgm)			/* name of this program */
{
	FILE	*f;
	char	*p;
	long	i, m;

	if (!qhats || Nq<1) {
		fprintf(stderr,"ERROR: no data to write in writeOutput()\n");
		exit(1);
	}
	if ( !(f=fopen(outName,"w")) ) {
		fprintf(stderr,"ERROR: cannot open file '%s' to write\n",outName);
		exit(1);
	}

	fprintf(f,"$filetype		PeaksFile\n");
	if (xtalHeaderBuf) {
		fprintf(f,"// parameters defining the crystal structure:\n");
		p = strtok(xtalHeaderBuf,"\n");			/* write out the crystal structure */
		p = strtok(NULL,"\n");					/* skip the first line which is the file type */
		while (p) {
			if (*p =='$') fprintf(f,"%s\n",p);
			p = strtok(NULL,"\n"); 
		}
	}
	if (xtalFile) fprintf(f,"$xtalFileName\t%s\n",xtalFile);

	if (pixelHeaderBuf) {
		fprintf(f,"// parameters from the peak fitting:\n");
		p = strtok(pixelHeaderBuf,"\n");		/* write out the pixel header info */
		p = strtok(NULL,"\n");					/* skip the first line which is the file type */
		while (p) {
			if (*p =='$') {
				if (!strstr(p,"$Npeaks\t") && !strstr(p,"$peakList\t") && !strstr(p,"$executionTime\t")) fprintf(f,"%s\n",p);
			}
			p = strtok(NULL,"\n"); 
		}
	}
	if (strlen(geoFile)) fprintf(f,"$geoFileName	%s\n",geoFile);
	if (strlen(pgm)) fprintf(f,"$programName	%s\n",pgm);

	fprintf(f,"\n// the following table contains xyz compotnents of G^ and the integral of the peak\n");
	fprintf(f,"$N_Ghat+Intens 	%ld		// number of G^ vectors\n",Nq);
	for (i=0;i<Nq;i++) {
		m = 4*i;
		fprintf(f, "% .7f, % .7f, % .7f,     %g\n",qhats[m],qhats[m+1],qhats[m+2],qhats[m+3]);
	}
	fclose(f);
}


int readCrystalFile(
char *fname,
struct crystalStructure *xtal)
{
	char	*xtalHeaderBuf;		/* string with tag values to pass on to the output flie */
	char	line[512];
	double	units2Angstrom;
	int		SpaceGroup;
	int		err;

	xtalHeaderBuf = readCrystalFileHeader(fname);
	if (!xtalHeaderBuf) return 1;

	InitCleanCrystalStructure(xtal);
	strFromTagBuf(xtalHeaderBuf,"structureDesc",line,500);	line[256]='\0';
	strncpy(xtal->desc,line,255); 

	xtal->lengthUnits = 1.e10;											/* default is Angstrom */
	if (!strFromTagBuf(xtalHeaderBuf,"lengthUnit",line,500)) {			/* search for units of the lattice parameters */
		if (!strcmp(line,"nm")) xtal->lengthUnits = 1.e9;				/* using nm */
		else if (strstr(line,"Ang")==line) xtal->lengthUnits = 1.e10;	/* using Angstrom */
		else if (strstr(line,"micron")==line) xtal->lengthUnits = 1.e6;	/* using micron */
		else { fprintf(stderr,"ERROR: invalid $lengthUnit = '%s',  in readDataFromJZT()\n",line); goto badFile; }
	}
	units2Angstrom = 1.e9 /xtal->lengthUnits;			/* set conversion factor based on xtal.lengthUnit */

	if (strFromTagBuf(xtalHeaderBuf,"latticeParameters",line,500)) { fprintf(stderr,"ERROR: cannot find tag for lattice parameters in file '%s'\n",fname); goto badFile; }
	else {
		double a=0,b=0,c=0,alpha=0,beta=0,gam=0;
		size_t Ntype, Nalloc;
		if (sscanf(line,"{%lg, %lg, %lg, %lg, %lg, %lg}",&a,&b,&c,&alpha,&beta,&gam)!=6) {
			fprintf(stderr,"ERROR: unable to read 6 values from $latticeParameters in file '%s'\n",fname);
			goto badFile;
		}
		if (a<=0 || b<=0 || c<=0 || alpha<=0 || beta<=0 || gam<=0 || alpha>=180 || beta>=180 || gam>=180) {
			fprintf(stderr,"ERROR: invalid lattice parameters in file '%s'\n",fname);
			fprintf(stderr,"ERROR: lattice parameters %g, %g, %g, %g, %g, %g invalid\n",a,b,c,alpha,beta,gam);
			goto badFile;
		}
		xtal->lengthUnits = 1.e10;						/* switching to Angstrom */
		xtal->a = a*units2Angstrom;						/* everything is OK, set lattice */
		xtal->b = b*units2Angstrom;
		xtal->c = c*units2Angstrom;
		xtal->alpha = alpha*M_PI/180.;
		xtal->beta = beta*M_PI/180.;
		xtal->gamma = gam*M_PI/180.;

		Nalloc = 100;									/* read the atomTypes */
		xtal->atomType = calloc(Nalloc,sizeof(xtal->atomType[0]));
		if (!(xtal->atomType)) { fprintf(stderr,"ERROR: unable to allocate space for atomTypes in readDataFromJZT()\n"); goto badFile; }
		for(Ntype=0;Ntype<STRUCTURE_ATOMS_MAX;Ntype++) {
			double	x,y,z,occ;
			char	ele[61];
			char	tagName[120];
			sprintf(tagName,"AtomDesctiption%ld",Ntype+1);	/* tag for next atom type */
			if (strFromTagBuf(xtalHeaderBuf,tagName,line,500)) break;	/* get next atom type, break if none found */
			if (sscanf(line,"{%s  %lg %lg %lg %lg}",ele,&x,&y,&z,&occ)!=5) break;
			ele[59] = '\0';
			if (Ntype>=Nalloc) {						/* need more room, extend ->xtal.atomType[] */
				Nalloc = Ntype+100;
				xtal->atomType = realloc(xtal->atomType,Nalloc*sizeof(xtal->atomType[0]));
				if (!(xtal->atomType)) { fprintf(stderr,"ERROR: unable to re-allocate space for atomTypes in readDataFromJZT()\n"); goto badFile; }
			}
			strncpy(xtal->atomType[Ntype].name,ele,60);
			xtal->atomType[Ntype].x = x;
			xtal->atomType[Ntype].y = y;
			xtal->atomType[Ntype].z = z;
			xtal->atomType[Ntype].occ = occ;
			xtal->atomType[Ntype].Zatom = atomicNumber(ele);
		}
		xtal->atomType = realloc(xtal->atomType,Ntype*sizeof(xtal->atomType[0]));
		xtal->Ntype = Ntype;
	}

	err = strFromTagBuf(xtalHeaderBuf,"SpaceGroup",line,250);/* search for the lattice SpaceGroup, set to default of FCC above */
	if (err) err = strFromTagBuf(xtalHeaderBuf,"latticeStructure",line,250);	/* old for backward compatibility, should use $SpaceGroup */
	if (err) fprintf(stderr,"ERROR: cannot find tag for SpaceGroup in file '%s', \n",fname);
	else {
		int		i;
		SpaceGroup = 0;
		i = sscanf(line,"%d",&SpaceGroup);				/* get SpaceGroup, and set it if valid */
		if(i==1 && SpaceGroup>=1 && SpaceGroup<=230) xtal->SpaceGroup = SpaceGroup;
		else fprintf(stderr,"ERROR: in file '%s', $SpaceGroup is '%s', it must be number in range [1,230], defaulting to FCC\n",fname,line);
	}

	return 0;
	badFile:
		CHECK_FREE(xtalHeaderBuf);						/* free allocated space */
		freeCrystalStructure(xtal);
	return 1;

}


char *readCrystalFileHeader(
char *fname)
{
	FILE	*f=NULL;
	char	*xtalHeaderBuf=NULL;						/* string with tag values to pass on to the output flie */
	long	flen;
	long	i;
	int		isXtl=0;
	char	*cif=NULL;
	long	cif_version = 1;
	long	dim = 3;

	if ( !(f=fopen(fname,"r")) ) {
		fprintf(stderr,"ERROR: cannot open file '%s' to read crystal structure\n",fname);
		return NULL;
	}
	fseek(f,0,SEEK_END);								/* move to end */
	flen = ftell(f);
	if (flen<0) { fprintf(stderr,"ERROR: crystal file '%s' is too long\n",fname); goto badFile; }
	fseek(f,0,SEEK_SET);								/* re-posiiton to start */
	
	xtalHeaderBuf = calloc((size_t)flen+1,sizeof(char));/* allocate space for the header */
	if (!(xtalHeaderBuf)) { fprintf(stderr,"ERROR: unable to allocate space for 'xtalHeaderBuf' when reading '%s'\n",fname); goto badFile; }
	xtalHeaderBuf[0] = '\0';							/* start this empty */
	fread(xtalHeaderBuf, sizeof(char), (size_t)flen, f);/* read the tagged header values into xtalHeaderBuf[] */
	fclose(f);
	for (i=0;i<flen;i++) if (xtalHeaderBuf[i]=='\r') xtalHeaderBuf[i]='\n';	/* change all <cr> to <nl> */

	isXtl = checkFileTypeLine(xtalHeaderBuf,"CrystalStructure");			/* return TRUE proper type of xtl file */
	if (!isXtl) {
		cif = XMLtagContents("cif",xtalHeaderBuf,0);						/* contents of the cif tag */
		char	*cifVals;
		if (XMLattibutes2KeyList("cif",xtalHeaderBuf,0, &cifVals)) {		/* try to get version & dim from <cif> */
			cif_version = IntByKey("version",cifVals,'=',';');
			dim = IntByKey("dim",cifVals,'=',';');
			cif_version = cif_version ? cif_version : 1;					/* default values are version 1 and 3D */
			dim = dim ? dim : 3;
			CHECK_FREE(cifVals);
		}
	}
	if (cif) {												/* translate the xml into $tag data format */
		char	*cell=NULL;
		char	line[256];
		char	*str=NULL;
		char	*space_group=NULL;
		char	*site=NULL;
		int		iatom;
		char	*label=NULL;
		char	*fract_xyz=NULL;							/* use either fract_xyz or fract_x fract_y fract_z */
		char	*fract_x=NULL;
		char	*fract_y=NULL;
		char	*fract_z=NULL;
		char	*occupancy=NULL;

		xtalHeaderBuf[0] = '\0';							/* restart this empty, fill with result from cif */

		strcat(xtalHeaderBuf,"$filetype\t\t\tCrystalStructure\n");
		if ((str=XMLtagContents("chemical_name_common",cif,0))) {
			strcat(xtalHeaderBuf,"$structureDesc\t\t");
			strcat(xtalHeaderBuf,str);
			strcat(xtalHeaderBuf,"\n");
			CHECK_FREE(str);
		}

		if (cif_version==2 && dim==3) {				/* version 2 of xml xtal file, and 3D data */
			if ((space_group=XMLtagContents("space_group",cif,0))) {
				if ((str=XMLtagContents("id",space_group,0))) {
					char *c = strchr(str,':');		/* find a possible colon */
					if (c) *c = '\0';				/* trim off the colon and everything after it */
				}
				else {								/* ailed to find <id>, get SpaceGroup from <IT_number> */
					str = XMLtagContents("IT_number",space_group,0);
				}
			}
		}
		else {										/* version 1 of xml xtal file */
			str = XMLtagContents("space_group_IT_number",cif,0);
		}
		if (str) {
			strcat(xtalHeaderBuf,"$SpaceGroup\t\t");
			strcat(xtalHeaderBuf,str);
			strcat(xtalHeaderBuf,"\n");
		}
		CHECK_FREE(str);
		CHECK_FREE(space_group);					/* this is safe even if space_group was not allocated */

		cell = XMLtagContents("cell",cif,0);
		if (cell) {
			char lat[512];
			char unit[256];
			unit[0] = lat[0] = '\0';
			if ((str=XMLtagContents("a",cell,0))) {
				char	*keyVals;
				char	*p = NULL;;
				strcat(lat,str);
				strcat(lat,", ");
				if (XMLattibutes2KeyList("a",cell,0, &keyVals)) {
					if (keyVals==strstr(keyVals,"unit=")) p = keyVals + 5;	/* at the start of keyVals */
					else if ((p=strstr(keyVals,";unit="))) p += 6;			/* in keyVals, but not at start */
					if (p) {												/* p is at start of value for unit */
						strcat(unit,p);
						if ((p=strchr(unit,';'))) *p = '\0';				/* end just before following semi-colon */
					}
				}
				CHECK_FREE(keyVals);
				CHECK_FREE(str);
			}
			if ((str=XMLtagContents("b",cell,0))) {
				strcat(lat,str);
				strcat(lat,", ");
				CHECK_FREE(str);
			}
			if ((str=XMLtagContents("c",cell,0))) {
				strcat(lat,str);
				strcat(lat,", ");
				CHECK_FREE(str);
			}
			if ((str=XMLtagContents("alpha",cell,0))) {
				strcat(lat,str);
				strcat(lat,", ");
				CHECK_FREE(str);
			}
			if ((str=XMLtagContents("beta",cell,0))) {
				strcat(lat,str);
				strcat(lat,", ");
				CHECK_FREE(str);
			}
			if ((str=XMLtagContents("gamma",cell,0))) {
				strcat(lat,str);
				CHECK_FREE(str);
			}
			strcat(xtalHeaderBuf,"$latticeParameters\t{ ");
			strcat(xtalHeaderBuf,lat);
			strcat(xtalHeaderBuf," }\n");

			if ((str=XMLtagContents("alphaT",cell,0))) {
				strcat(xtalHeaderBuf,"$latticeAlphaT\t\t");
				strcat(xtalHeaderBuf,str);
				strcat(xtalHeaderBuf,"\n");
				CHECK_FREE(str);
			}

			if (strlen(unit)) {
				strcat(xtalHeaderBuf,"$lengthUnit\t\t");	/* length unit for lattice constants a,b,c */
				strcat(xtalHeaderBuf,unit);
				strcat(xtalHeaderBuf,"\n");
			}
			
			CHECK_FREE(cell);
		}

		site = label = fract_xyz = fract_x = fract_y = fract_z = occupancy = NULL;
		for (iatom=0; iatom<STRUCTURE_ATOMS_MAX; iatom++) {
			if ((site=XMLtagContents("atom_site",cif,iatom))) {
				label = XMLtagContents("label",site,0);
				if (cif_version == 2) fract_xyz = XMLtagContents("fract",site,0);		/* for version 2, use fract instead of 'fract_xyz' */
				else	fract_xyz = XMLtagContents("fract_xyz",site,0);
				if (!fract_xyz) {
					fract_x = XMLtagContents("fract_x",site,0);
					fract_y = XMLtagContents("fract_y",site,0);
					fract_z = XMLtagContents("fract_z",site,0);
				}
				occupancy = XMLtagContents("occupancy",site,0);
				if (label && (fract_xyz || (fract_x && fract_y && fract_z))) {
					if (strlen(label)>0) {
						if (fract_xyz && occupancy) sprintf(line,"$AtomDesctiption%d	{%s  %s %s}\n",iatom+1,label,fract_xyz,occupancy);
						else if (occupancy) sprintf(line,"$AtomDesctiption%d	{%s  %s %s %s %s}\n",iatom+1,label,fract_x,fract_y,fract_z,occupancy);
						else if (fract_xyz) sprintf(line,"$AtomDesctiption%d	{%s  %s 1}\n",iatom+1,label,fract_xyz);
						else sprintf(line,"$AtomDesctiption%d	{%s  %s %s %s 1}\n",iatom+1,label,fract_x,fract_y,fract_z);
						strcat(xtalHeaderBuf,line);
					}
				}
				CHECK_FREE(site);
				CHECK_FREE(label);
				CHECK_FREE(fract_xyz);
				CHECK_FREE(fract_x);
				CHECK_FREE(fract_y);
				CHECK_FREE(fract_z);
				CHECK_FREE(occupancy);
			}
			else break;
		}

		if ((str=XMLtagContents("citation",cif,0))) {
			strcat(xtalHeaderBuf,"$citation ");
			strcat(xtalHeaderBuf,str);
			strcat(xtalHeaderBuf,"\n");
			CHECK_FREE(str);
		}
		CHECK_FREE(cif);
	}

	return xtalHeaderBuf;
	badFile:
		CHECK_FREE(xtalHeaderBuf);						/* free allocated space */
		if (f != NULL) fclose(f);						/* ensure an opened file gets closed */
	return NULL;
}


char *readPixelPeakFileHeader(
char *fname)
{
	FILE	*f=NULL;
	char	*headerBuf;									/* string with tag values to pass on to the output flie */
	char	line[512];
	long	flen=0;
	long	i;

	if ( !(f=fopen(fname,"r")) ) {
		fprintf(stderr,"ERROR: cannot open file '%s' to read pixel peak positions\n",fname);
		return NULL;
	}
	fseek(f,0,SEEK_END);								/* move to end */
	flen = ftell(f);			
	if (flen<0) { fprintf(stderr,"ERROR: pixel peak file '%s' is too long\n",fname); goto badFile; }
	fseek(f,0,SEEK_SET);								/* re-posiiton to start */

	headerBuf = calloc((size_t)flen+1,sizeof(char));	/* allocate space for the header */
	if (!(headerBuf)) { fprintf(stderr,"ERROR: unable to allocate space for 'headerBuf' when reading '%s'\n",fname); goto badFile; }
	headerBuf[0] = '\0';								/* start this empty */
	fread(headerBuf, sizeof(char), (size_t)flen, f);	/* read the tagged header values into headerBuf[] */
	fclose(f);
	for (i=0;i<flen;i++) if (headerBuf[i]=='\r') headerBuf[i]='\n';		/* change all <cr> to <nl> */

	strFromTagBuf(headerBuf,"filetype",line,500);
	if (strcmp(line,"PixelPeakList"))  { fprintf(stderr,"ERROR: the file '%s' is not a 'PixelPeakList' file type\n",fname); goto badFile;; }

	return headerBuf;
	badFile:
		CHECK_FREE(headerBuf);							/* free allocated space */
		if (f != NULL) fclose(f);						/* ensure an opened file gets closed */
	return NULL;
}



double *pixels2qhatsTable(
struct detectorGeometry *d,				/* geometry parameters for this detector */
char *buf)
{
	double *peaks=NULL;
	double *pk;							/* a pointer used to access peaks */
	long	Npeaks;
	long	xDimDet, yDimDet;
	long	startx, endx, groupx;
	long	starty, endy, groupy;
	char	line[512];
	char	*p;
	double	x,y,pkIntens,area;
	double	qx,qy,qz;					/* outgoing q hat */
	double	depth=NAN;					/* depth from file */
	long	i;

	if (strFromTagBuf(buf,"Npeaks",line,500)) goto badFile;
	Npeaks = atoi(line);
	if (strFromTagBuf(buf,"xDimDet",line,500)) goto badFile;
	xDimDet = atoi(line);
	if (strFromTagBuf(buf,"yDimDet",line,500)) goto badFile;
	yDimDet = atoi(line);
	if (strFromTagBuf(buf,"startx",line,500)) goto badFile;
	startx = atoi(line);
	if (strFromTagBuf(buf,"endx",line,500)) goto badFile;
	endx = atoi(line);
	if (strFromTagBuf(buf,"groupx",line,500)) goto badFile;
	groupx = atoi(line);
	if (strFromTagBuf(buf,"starty",line,500)) goto badFile;
	starty = atoi(line);
	if (strFromTagBuf(buf,"endy",line,500)) goto badFile;
	endy = atoi(line);
	if (strFromTagBuf(buf,"groupy",line,500)) goto badFile;
	groupy = atoi(line);
	if (strFromTagBuf(buf,"depth",line,500)) depth=NAN;
	else depth = atof(line);

	if (strFromTagBuf(buf,"peakList",line,500)) goto badFile;

	p = strstr(buf,"\n$peakList\t");
	if (!p) p = strstr(buf,"\n$peakList ");	/* try again but with a space this time */
	if (!p) goto badFile;

	if (Npeaks<1) goto badFile;

	p = strtok(p,"\n");						/* write out the crystal structure */
	p = strtok(NULL,"\n");					/* skip the first line which is the file type */

	peaks = calloc((size_t)Npeaks,4*sizeof(double));
	if (!peaks) { fprintf(stderr,"ERROR: cannot allocate space for table of peak positions"); goto badFile; }

	for (i=0,pk=peaks;i<Npeaks && p;i++) {
		if (!p) break;
		if (sscanf(p,"%lg\t%lg\t%lg\t%lg",&x,&y,&pkIntens,&area)!=4) break;
		/* convert binned pixel (x,y) to full-chip unbinned pixels */
		x = startx + x*groupx + (double)(groupx-1)/2.0;	/* pixel is zero based here & startx is zero based */
		y = starty + y*groupy + (double)(groupy-1)/2.0;	/* groupx=1 is un-binned */
		pixel2q(d,x,y,depth,&qx,&qy,&qz);
		*pk = qx;		pk++;
		*pk = qy;		pk++;
		*pk = qz;		pk++;
/*		*pk = pkIntens;	pk++;			changed Nov 6, 2015, now area, not peak intensity */
		*pk = area;	pk++;
		p = strtok(NULL,"\n"); 
	}

	return peaks;
	badFile:
		CHECK_FREE(peaks);							/* free allocated space */
	return NULL;
}

// convert px,py positions on detector into Q vector, assumes ki={0,0,1}, // returns theta (rad)
double pixel2q(				/* returns theta (rad) */
struct detectorGeometry *d,	/* geometry parameters for this detector */
double	px,					/* pixel position (full-chip, unbinned, zero based) */
double	py,
double	depth,				/* sample depth measured along the beam */
double	*qx,				/* outgoing q hat */
double	*qy,
double	*qz)
{
	double	x,y,z;								/* position on detector, also direction of kf */
	double	kf;									/* length of {x,y,z} */
	double	theta;								/* Bragg angle (rad) */
	double	dot;								/* dot(kf,ki)/(|kf|*|ki|) */
	double	qlen;

	/* this all assumes that ki = {0,0,1} */
	pixel2XYZ(d,px,py,&x,&y,&z);				/* kf is in direction of pixel in beam line coords */

	if (depth==depth) z -= depth;				/* kfDepth = d*ki + kfZero */
	kf = sqrt(x*x + y*y + z*z);
	x /= kf;									/* normalize kf = {x,y,z} */
	y /= kf;
	z /= kf;
	dot = z;									/* MatrixDot(kf,ki), very simple when ki={0,0,1} */

	if (dot>1) theta=0;							/* catch case when dot is slightly outside of [-1,1] the domain of acos */
	else if(dot<-1) theta=M_PI;
	else theta = acos(dot);						/* ki.kf = cos(2theta), (radians) */

	if (qx && qy && qz) {						/* compute q if valid pointers passed */
		z -= 1.0;								/* (x,y,z) is now kf - ki,  this is simples since ki=(001) */
		qlen = sqrt(x*x + y*y + z*z);			/*    and (kf-ki) || q */
		*qx = x/qlen;							/* normalize q */
		*qy = y/qlen;
		*qz = z/qlen;
	}
	return theta;
}

void pixel2XYZ(
struct detectorGeometry *d,	/* geometry parameters for this detector */
double	px,					/* pixel position (full-chip, unbinned, zero based) */
double	py,
double	*x,					/* point on detector */
double	*y,
double	*z)
{
	double	xp,yp, zp;							/* x' and y' (requiring z'=0), detector starts centered on origin and perpendicular to z-axis */

//	peakCorrect(d,px,py)						/* convert pixel on detector to an undistorted distance in pixels (takes zero based pixels) */
	xp = (px - 0.5*(d->Nx-1)) * d->sizeX/d->Nx;	/* (x' y' z'), position on detector */
	yp = (py - 0.5*(d->Ny-1)) * d->sizeY/d->Ny;

	xp += d->P[0];								/* translate by P */
	yp += d->P[1];
	zp = d->P[2];

	*x = d->rho00*xp + d->rho01*yp + d->rho02*zp;	/* xyz = rho x [ (x' y' z') + P ] */
	*y = d->rho10*xp + d->rho11*yp + d->rho12*zp;	/* rho is pre-calculated from vector d->R */
	*z = d->rho20*xp + d->rho21*yp + d->rho22*zp;
}


int detectorIDfromGeo(		/* finds the detector with corresponding id in the geo structure */
struct geoStructure *geo,
char	*detector_ID)
{
	int		i;
	for (i=0; i < geo->Ndetectors; i++) {
		if (strcmp(detector_ID,geo->d[i].detectorID)==0) return i;	/* found a match */
	}
	return 0;				/* default value is 0 */
}

