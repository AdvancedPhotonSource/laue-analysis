#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _MSC_VER						/* identifies this as a Microsoft compiler */
#define _USE_MATH_DEFINES			/* added RX2011 */
#endif
#include <math.h>
#include "Euler.h"

#ifdef DEBUG_ON
// #define DEBUG 1
#endif

#define EXIT_WITH_HELP { for(i=0;help[i][0];i++) fprintf(stderr,"%s\n",help[i]); exit(1); }
#define MIN(A,B)	( (A)<(B)?(A):(B) )
#define MAX(A,B)	( (A)>(B)?(A):(B) )
#define OUTFILE		"eulerOut.txt"
int strFromTagFile(FILE *f,char *tagIn, char *value, long maxLen);


FILE	*fout;						/* file descriptor, either /dev/null, or stdout */

/* some good tests used command line switches of:
 * -f xyTOkq_Si.txt -a 0.04
 * -f xyTOkq_Al.txt -a 0.6
 */


int main (int argc, const char * argv[]) {
	int		err=0;							/* error flag, 0 is OK */
	double  keVmaxCalc=-1;					/* max energy when determining pattern */
	double  keVmaxTest=-1;					/* limit in energy for finding test points (Gtest), probably [2^(1/3)]*keVmaxCalc */
	char	fname[256]="xyTOkq_Si.txt";		/* or also test with "xyTOkq_Al.txt" */
	char	outfile[256]=OUTFILE;			/* name of output file, set a default here */
	int		hkl0[3]={0,0,1};				/* preferred hkl actually pointing to center of pattern */
	double  cone=72.;						/* acceptable cone angle (deg) from preferred hkl0 */
	double  angleTolerance=0.1;				/* difference in angles (deg) between G-vector pairs to be considered coincident*/
	int		i;
	long	maxData=250;					/* max number of spots to use, ie use the first maxData spots */
	int		quiet=0;						/* flag, if set, then suppresses most terminal output */
	char	tagFile[256]="";				/* name of a "tag values" file with all needed command line inputs */
	char	value[256];						/* value from tag file */
	char	*p;
	FILE	*f;								/* file descriptor for tagged input file */
	char	defaultFolder[256]=DEFAULT_FOLDER; /* name of output file, set a default here */

	static char *help[] = {"switches are:", "\t-k keV calc max", "\t-t keV test max", "\t-f filename with input peak data", 
							"\t-o filename for output results", "\t-h  h k l (preferred hkl, three agruments following -h)", 
							"\t-q suppress most terminal output (-Q forces output)", 
							"\t-c cone angle (deg)", "\t-a angle tolerance (deg)", "\t-n max num. of spots from data file to use",
							"\t-i tagged value file with inputs, available tags are:",
							"\t\t$EulerInputFile\talways include this tag to identify the file type",
							"\t\t$keVmaxCalc\tmaximum energy to calculate (keV)  [-k]",
							"\t\t$keVmaxTest\tmaximum energy to test (keV)  [-t]",
							"\t\t$inFile\t\tname of file with input peak positions  [-f]",
							"\t\t$outFile\toutput file name  [-o]",
							"\t\t$hkl\t\tpreferred hkl, 3 space separated numbers, hkl toward detector center,  [-h]",
							"\t\t$quiet\t\t1 or 0  [-q or -Q]",
							"\t\t$cone\t\tcone angle from hkl(deg)  [-c]",
							"\t\t$angleTolerance angular tolerance (deg)  [-a]",
							"\t\t$maxData\tmaximum number of peaks  [-n]",
							"\t\t$defaultFolder\tdefault folder to prepend to file names", "" };

/*	cone = 180/M_PI * 0.4*M_PI;				// only investigate hks that are within 72° of the (001)
 *	angleTolerance = 180/M_PI*0.025/(Detector height); // this corresponds to a 25µm pixel separation at 100mm away
 *	angleTolerance *= 40.;
 *	if (strstr(fname,"xyTOkq_Si")) angleTolerance *= 3.;	// 3 pixels wide, for good stuff like S
 *	else if (strstr(fname,"xyTOkq_Al")) angleTolerance *= 40.;// 20 pixels wide, for good stuff like Aluminum
 * for the Al, use angleTolerance = 0.6;
 * for the Si, use angleTolerance = 0.04;
 */

	if (argc<=1) EXIT_WITH_HELP;
	for (i=1;i<argc;i++) {
		if (!strncmp(argv[i],"-i",2)) {		/* get lots of values from the default file */
			if ((++i)>=argc) { fprintf(stderr,"-i not follwed by a argument with the input file name \n"); EXIT_WITH_HELP }
			if (strlen(argv[i])<1) { fprintf(stderr,"-f cannot read name of input file from argv[i]='%s'\n",argv[i]); EXIT_WITH_HELP }
			tagFile[0]='\0';
			if (argv[i][0]!='/') strcpy(tagFile,defaultFolder);
			strcat(tagFile,argv[i]);
			if (( f = fopen(tagFile, "r")) == NULL) { fprintf(stderr,"Can't open file '%s'\n",tagFile); EXIT_WITH_HELP; }
			if (strFromTagFile(f,"EulerInputFile",value,250)) { fprintf(stderr,"file '%s' is not an Euler input file\n",tagFile); EXIT_WITH_HELP }
			if (!strFromTagFile(f,"keVmaxCalc",value,250)) keVmaxCalc = strtod(value,NULL);
			if (!strFromTagFile(f,"keVmaxTest",value,250)) keVmaxTest = strtod(value,NULL);
			if (!strFromTagFile(f,"quiet",value,250)) quiet = strtod(value,NULL);
			if (!strFromTagFile(f,"cone",value,250)) cone = strtod(value,NULL);
			if (!strFromTagFile(f,"angleTolerance",value,250)) angleTolerance = strtod(value,NULL);
			if (!strFromTagFile(f,"maxData",value,250)) maxData = strtod(value,NULL);
			if (!strFromTagFile(f,"defaultFolder",value,250)) {
				if (strlen(value)<1) { fprintf(stderr,"$inFile cannot read name of default folder from '%s'\n",value); EXIT_WITH_HELP }
				strcpy(defaultFolder,value);
			}
			if (!strFromTagFile(f,"inFile",value,250)) {
				if (strlen(value)<1) { fprintf(stderr,"$inFile cannot read name of data file from '%s'\n",value); EXIT_WITH_HELP }
				fname[0]='\0';
				if (value[0]!='/') strcpy(fname,defaultFolder);
				strcat(fname,value);
			}
			if (!strFromTagFile(f,"outFile",value,250)) {
				if (strlen(value)<1) { fprintf(stderr,"-f cannot readd name of output file from '%s'\n",value); EXIT_WITH_HELP }
				outfile[0]='\0';
				if (value[0]!='/') strcpy(outfile,defaultFolder);
				strcat(outfile,value);
			}
			if (!strFromTagFile(f,"hkl",value,250)) {
				int h0,h1,h2;
				if (sscanf(value,"%d %d %d",&h0,&h1,&h2)!=3) { fprintf(stderr,"$hkl cannot interpret '%s' as a three numbers\n",value); EXIT_WITH_HELP }
				hkl0[0] = h0;
				hkl0[1] = h1;
				hkl0[2] = h2;
			}
			fclose(f);
			continue;
		}

		if (!strncmp(argv[i],"-f",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-f not follwed by a argument with the peak file name \n"); EXIT_WITH_HELP }
			if (strlen(argv[i])<1) { fprintf(stderr,"-f cannot read name of data file from argv[i]='%s'\n",argv[i]); EXIT_WITH_HELP }
			fname[0]='\0';
			if (argv[i][0]!='/') strcpy(fname,defaultFolder);
			strcat(fname,argv[i]);
			continue;
		}
		else if (!strncmp(argv[i],"-o",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-o not follwed by a argument with the output file name \n"); EXIT_WITH_HELP }
			if (strlen(argv[i])<1) { fprintf(stderr,"-o cannot read name of output file from argv[i]='%s'\n",argv[i]); EXIT_WITH_HELP }
			outfile[0]='\0';
			if (argv[i][0]!='/') strcpy(outfile,defaultFolder);
			strcat(outfile,argv[i]);
			continue;
		}
		else if (!strncmp(argv[i],"-k",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-k not follwed by a argument with the energy\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%lg",&keVmaxCalc)!=1) { fprintf(stderr,"-k cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-t",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-t not follwed by a argument with the energy\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%lg",&keVmaxTest)!=1) { fprintf(stderr,"-t cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-a",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-a not follwed by a argument with the angleTolerance\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%lg",&angleTolerance)!=1) { fprintf(stderr,"-a cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-c",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-c not follwed by a argument with the cone angle\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%lg",&cone)!=1) { fprintf(stderr,"-c cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-h",2)) {
			if ((i+3)>=argc) { fprintf(stderr,"-h not follwed by 3 arguments with the preferred central hkl\n"); EXIT_WITH_HELP }
			if (sscanf(argv[++i],"%d",&(hkl0[0]))!=1) { fprintf(stderr,"-h h cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			if (sscanf(argv[++i],"%d",&(hkl0[1]))!=1) { fprintf(stderr,"-h k cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			if (sscanf(argv[++i],"%d",&(hkl0[2]))!=1) { fprintf(stderr,"-h l cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-n",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-n not follwed by a argument with the max number of data points\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%ld",&maxData)!=1) { fprintf(stderr,"-n cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-q",2)) { quiet = 1; continue; }
		else if (!strncmp(argv[i],"-Q",2)) { quiet = 0; continue; }
		else EXIT_WITH_HELP;
	}

	if (!quiet) fout = stdout;						/* not quiet, send output to stdout */
	else {
		if ((fout=fopen("/dev/null", "w")) == NULL) { fprintf(stderr,"unable to open '/dev/null' for fout\n"); exit(1); }
	}

	keVmaxCalc = (keVmaxCalc<KEV_MIN) ? KEV_MAX : keVmaxCalc;
	if (keVmaxTest<KEV_MIN) {
		keVmaxTest = pow(2.,1./3.)*keVmaxCalc;		/* max keV for finding matching points, 2^.333 gives twice the volume */
		keVmaxTest = sqrt(2.)*keVmaxCalc;			/* max keV for finding matching points, sqrt(2) gives 2.8 times the volume */
	}
	keVmaxTest = fabs(keVmaxTest);
	keVmaxCalc = fabs(keVmaxCalc);

	/*	printf("(M_PI = %.15f - PI) = %g\n",M_PI,M_PI-3.141592653589793238462643); */

	if ((p=strstr(fname,"Peaks.txt"))) {
		strcpy(outfile,fname);
		p=strstr(outfile,"Peaks.txt");
		*p = '\0';									/* trim off the 'Peaks.txt' */
		strcat(p,"Index.txt");						/* and make it end with Index.txt */
	}
	else if ((p=strstr(fname,"xyTOkq_"))) {
		strcpy(outfile,defaultFolder);
		strcat(outfile,p+7);						/* append starting after the '_' */
		if ((p=strchr(outfile+strlen(defaultFolder),'.'))) p[0] = '\0';  /* terminate at the '.' before extension */
		strcat(outfile,".txt");						/* add '.txt' extension */
	}

	angleTolerance = fabs(angleTolerance*M_PI/180); /* convert deg->radian before calling testOrientFast() */
	cone = fabs(cone*M_PI/180);						/* convert deg->radian before calling testOrientFast() */
#ifdef DEBUG
	fprintf(fout,"using file='%s' to call testOrientFast()\n",fname);
#endif
	err = testOrientFast(fname,outfile,keVmaxCalc,keVmaxTest,angleTolerance,hkl0,cone,maxData);

	return err;
}


/*
 * void TestMakeVecList(long *Nm, double  (**vecList)[3] );
 *
 * 	long	Nm;
 *	int		i;
 *	double (*vecList)[3]=NULL;
 *
 *		vecList = NULL;
 *  	TestMakeVecList(&Nm, &vecList);
 *  	printf("returned allocated address of vecList is %lx\n",(unsigned long)*vecList);
 *  	for (i=0;i<Nm;i++) printf("%3d    %8.4f %8.4f %8.4f\n",i,vecList[i][0],vecList[i][1],vecList[i][2]);
 *      printf("End after TestMakeVecList!\n");
 */

