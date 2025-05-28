#include <time.h>
#include <math.h>
#include <ctype.h>
#include "peaksearch.h"
#include "WinViewImage.h"
#include "grid_operations.h"
#include "microHDF5.h"

/*#ifndef MAX_FILE_LENGTH
#define MAX_FILE_LENGTH 2047
#endif
*/

#define EXIT_WITH_HELP { for(i=0;help[i][0];i++) fprintf(stderr,"%s\n",help[i]); exit(1); }


double itype2saturation(int itype);


//	#define IDL_FILES 1

int main(int argc, char **argv){
//	if(argc<3){
//		printf("USAGE: %s InputImagefileName  OutputPeaksFileName\n",argv[0]);
//		exit(1);
//	}
	int		hdf5;					/* flag, file is hdf5 type */
	int		spe;					/* flags, file is spe type */
	char	inFile[MAX_FILE_LENGTH+1];
	char	outFile[MAX_FILE_LENGTH+1];
	char	maskFile[MAX_FILE_LENGTH+1];	/* only use pixels with mask==0 */
	struct HDF5_Header h5head;		/* HDF5 header information */
	WinViewImage* image=NULL;
	WinViewImage* mask=NULL;
	double	*buf=NULL;
	double	*bufMask=NULL;
	int		maskValue;				/* value of a mask pixel, use image pixel when this is 0 */
	double	threshold=NAN;			/* lower level threshold for accepting pixels as part of a peak */
	double	sum=0.0;				/* total of all pixels in image */
	double	average=0.0;			/* average value of a pixel */
	int		NpeakMax=-1;			/* only search the first NpeakMax peaks, this limits the search, only used by boxsearch */
	int		minSeparation=-1;		/* minimum separation between any two peaks (default is 2*boxsize) */
	double	saturation_level;		/* saturated pixel level */
	double	sumAboveThreshold=0.0;	/* sum of all pixels above threshold */
	size_t	numAboveThreshold=0;	/* number of pixels above threshold */
	double	min_size=0.5;			/* minimum spot size (dx or dy) FW */
	double	seconds;				/* execution timem NOT exposure (sec) */
	int		boxsize;				/* half width of box to use for each peak */
	float	maxRfactor;
	bool	smooth=false;			/* if true fit Lorentzian to smoothed image, otherwise use raw image */
	double	pixel;
	size_t	i;
	double	thresholdRatio = 4.0;	/* when threshold not given, use a threshold of thresholdRatio*(standard deviation) above background */
	struct ExtraOutput_Header exH;
	List* peaks=NULL;
	#ifdef USE_BOX
		NpeakMax = 50;				/* only search the first NpeakMax peaks, this limits the search */
	#endif

	clock_t tstart = clock();
	#ifdef USE_BOX
	static char *help[] = {"USAGE:  peaksearch [-b boxsize -R maxRfactor -m min_size -M max_peaks -s minSeparation -S -K maskFile -D distortionMap] InputImagefileName  OutputPeaksFileName",
		"switches are:", "\t-b box size (half width)", "\t-R maximum R factor", "\t-m min size of peak (pixels)", "\t-M max number of peaks to examine(default=50)",
		"\t-s minimum separation between two peaks (default=2*boxsize)", "\t-S use smoothed image for Lorentzian fit",
		"\t-p use -p L for Lorentzian (default), -p G for Gaussian", "\t-K mask file name (use pixels with mask==0)", "\t-D distortion map file name", ""};
	#else
	static char *help[] = {"USAGE:  peaksearch [-b boxsize -R maxRfactor -m min_size -M max_peaks -s minSeparation -t threshold -T thresholdRatio -p (L,G) -S -K maskFile -D distortionMap] InputImagefileName  OutputPeaksFileName",
		"switches are:", "\t-b box size (half width)", "\t-R maximum R factor", "\t-m min size of peak (pixels)", "\t-M max number of peaks to examine(default=50)",
		"\t-s minimum separation between two peaks (default=2*boxsize)", "\t-t user supplied threshold (optional, overrides -T)", "\t-T threshold ratio, set threshold to (ratio*[std dev] + avg) (optional)", 
		"\t-p use -p L for Lorentzian (default), -p G for Gaussian", "\t-S smooth the image", "\t-K mask_file_name (use pixels with mask==0)", "\t-D distortion map file name", ""};
//	static char *help[] = {"USAGE:  peaksearch [-b boxsize -R maxRfactor -m min_size -s minSeparation -K maskFile] InputImagefileName  OutputPeaksFileName",
//		"switches are:", "\t-b box size (half width)", "\t-R maximum R factor", "\t-m min size of peak (pixels)",
//		"\t-t user supplied threshold (optional)", "\t-p use -p L for Lorentzian (default), -p G for Gaussian", "\t-K mask_file_name (use pixels with mask==0)", ""};
	#endif

	Genfileinf *ginf=default_genfileinf();						/* this default was customized for spe files */

	if (argc<3) EXIT_WITH_HELP;									/* at least have to include the input and output files */
	inFile[0] = outFile[0] = maskFile[0] = '\0';
	for (i=1;i<argc;i++) {
		if (!strncmp(argv[i],"-b",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-b not follwed by an argument with the box size\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%d",&boxsize)!=1) { fprintf(stderr,"-b cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			if (boxsize<=0) { fprintf(stderr,"ERROR: boxsize = %d\n",boxsize); EXIT_WITH_HELP }
			ginf->boxsize = boxsize;
			ginf->maxCentToFit = boxsize;
//			ginf->maxwidth = boxsize;							/* maxwidth is max allowed hwhm of a fitted spot */
			ginf->maxwidth = boxsize*1.5;						/* maxwidth is max allowed hwhm of a fitted spot */
			continue;
		}
		else if (!strncmp(argv[i],"-R",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-R not follwed by an argument with the max R factor\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%g",&maxRfactor)!=1) { fprintf(stderr,"-R cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			if (maxRfactor<=0) { fprintf(stderr,"ERROR: maxRfactor = %g\n",maxRfactor); EXIT_WITH_HELP }
			ginf->maxRfactor = maxRfactor;
			continue;
		}
		else if (!strncmp(argv[i],"-m",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-m not follwed by an argument with the min peak size (pixels)\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%lg",&min_size)!=1) { fprintf(stderr,"-m cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			if (min_size<=0) { fprintf(stderr,"ERROR: min_size = %lg\n",min_size); EXIT_WITH_HELP }
//			ginf->minwidth = min_size*0.01;
			ginf->minwidth = min_size/4.0;						/* minwidth is minimum allowed hwhm of a fitted spot */
			continue;
		}
		else if (!strncmp(argv[i],"-s",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-s not follwed by an argument with the min separation between two peaks\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%d",&minSeparation)!=1) { fprintf(stderr,"-s cannot interpret argv[i]='%s' as an integer\n",argv[i]); EXIT_WITH_HELP }
			if (minSeparation<1) { fprintf(stderr,"ERROR: minSeparation = %d, it must be at least 1\n",minSeparation); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-p",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-p not follwed by an argument with the peak shape (L or G)\n"); EXIT_WITH_HELP }
			int		c = toupper((int)((argv[i])[0]));			/* upper case of first character */
			if		(c=='L') ginf->peakShape = 0;				/* Lorentzian */
			else if	(c=='G') ginf->peakShape = 1;				/* Gaussian */
			else	{ fprintf(stderr,"ERROR: -p %c, it must be only 'L' or 'G'\n",c); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-M",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-M not follwed by an argument with the max number of peaks to examine\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%d",&NpeakMax)!=1) { fprintf(stderr,"-M cannot interpret argv[i]='%s' as an integer\n",argv[i]); EXIT_WITH_HELP }
			if (NpeakMax<=0) { fprintf(stderr,"ERROR: NpeakMax = %d, it must be at least 1\n",NpeakMax); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-S",2)) {
			smooth = true;
			continue;
		}
		#ifndef USE_BOX
		else if (!strncmp(argv[i],"-t",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-t not follwed by an argument with the threshold\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%lg",&threshold)!=1) { fprintf(stderr,"-t cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			if (threshold!=threshold) { fprintf(stderr,"ERROR: threshold = %g\n",threshold); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-T",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-T not follwed by an argument with the thresholdRatio\n"); EXIT_WITH_HELP }
			if (sscanf(argv[i],"%lg",&thresholdRatio)!=1) { fprintf(stderr,"-T cannot interpret argv[i]='%s' as a number\n",argv[i]); EXIT_WITH_HELP }
			if (!(thresholdRatio>0)) { fprintf(stderr,"ERROR: thresholdRatio = %g\n",thresholdRatio); EXIT_WITH_HELP }
			continue;
		}
		#endif
		else if (!strncmp(argv[i],"-K",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-K not follwed by an argument with the name of the mask file\n"); EXIT_WITH_HELP }
			strncpy(maskFile,argv[i],MAX_FILE_LENGTH);			/* maximum file name length is MAX_FILE_LENGTH */
			if (maskFile[0]=='-') maskFile[0]='\0';			/* assume that string starting with switch is an error, not a mask file name */
			if (strlen(maskFile)<1) { fprintf(stderr,"-K no mask file found (name cannot start with a '-'\n"); EXIT_WITH_HELP }
			continue;
		}
		else if (!strncmp(argv[i],"-D",2)) {
			if ((++i)>=argc) { fprintf(stderr,"-D not follwed by an argument with the name of the distortion map file\n"); EXIT_WITH_HELP }
			strncpy(ginf->CCDFilename,argv[i],MAX_FILE_LENGTH);	/* maximum file name length is MAX_FILE_LENGTH */
			if ((ginf->CCDFilename)[0]=='-') (ginf->CCDFilename)[0]='\0';	/* assume that string starting with switch is an error, not a mask file name */
			if (strlen(ginf->CCDFilename)<1) { fprintf(stderr,"-D no distortion map file found (name cannot start with a '-'\n"); EXIT_WITH_HELP }
			continue;
		}
		else if (argv[i][0]=='-') { fprintf(stderr,"unknown switch '%s'\n",argv[i]); EXIT_WITH_HELP }

		else if (!inFile[0]) {
			strncpy(inFile,argv[i],MAX_FILE_LENGTH);			/* maximum file name length is MAX_FILE_LENGTH */
		}
		else if (!outFile[0]) {
			strncpy(outFile,argv[i],MAX_FILE_LENGTH);			/* maximum file name length is MAX_FILE_LENGTH */
		}
		else { fprintf(stderr,"unknown argument '%s'\n",argv[i]); EXIT_WITH_HELP }
	}
	if (!strncmp(inFile,outFile,MAX_FILE_LENGTH)) { fprintf(stderr,"input and output files are BOTH equal to '%s'\n",inFile); EXIT_WITH_HELP }

	if (minSeparation<1) minSeparation = 2*boxsize;				/* set default min separation between any two peaks */

	hdf5 = !(!strstr(inFile,".h5"));
	spe = strstr(inFile,".spe") || strstr(inFile,".SPE");
	if ((spe+hdf5) != 1) {
		fprintf(stderr,"ERROR: cannot figure out if '%s' is an hdf5 or spe file\n",inFile);
		exit(1);
	}

	if (spe) {
		image = winview_image_import(inFile);
		exH.depth = NumberByKey("depthSi",image->header->PVlist,0,0);	/* sample depth (micron) */
		exH.energy = NumberByKey("keV",image->header->PVlist,0,0);		/* monochromatgor energy (keV) */
		char stype[128];
		WinViewControllers(image->header->controllerType,stype);
		sprintf(exH.detector_ID,"Roper %s",stype);
		exH.CCDshutterIN = 1;											/* CCD shutter, 1=IN, 0=OUT */
		exH.monoMode[0] = exH.userName[0] = exH.title[0] = exH.sampleName[0] = exH.dateExposed[0] = exH.beamline[0] = '\0';
		if (maskFile[0]) mask = winview_image_import(maskFile);			/* there is a mask file, load it */
	}
	else if (hdf5) {
		H5Eset_auto2(H5E_DEFAULT,NULL,NULL);				/* turn off printing of HDF5 errors */
		if (readHDF5header(inFile, &h5head)) { fprintf(stderr,"ERROR: unable to read HDF5 header, check permissions?\n"); exit(1); }
		if (HDF5ReadROIdouble(inFile, "entry1/data/data", &buf, 0,h5head.xdim-1,0,h5head.ydim-1, &h5head)) { fprintf(stderr,"ERROR: unable to read HDF5 image\n"); exit(1); }
		image = winview_image_new_empty();
		image->data = grid_new((int)h5head.xdim,(int)h5head.ydim);
		image->type = HDF5_FILE;							/* 0=spe, 1=hdf5 */
		image->data->height		= (int)h5head.ydim;
		image->data->width		= (int)h5head.xdim;
		image->data->values		= buf;
		image->header			= winview_header_new();
		image->header->xdim		= h5head.xdim;				/* copy needed values from hdf5 header to winview header, not all are needed */
		image->header->ydim		= h5head.ydim;
		image->header->xDimDet	= h5head.xDimDet;
		image->header->yDimDet	= h5head.yDimDet;
		image->header->startx	= h5head.startx + 1;		/* WinView uses 1 based pixels, I insist upon zero based */
		image->header->endx		= h5head.endx + 1;
		image->header->groupx	= h5head.groupx;
		image->header->starty	= h5head.starty + 1;
		image->header->endy		= h5head.endy + 1;
		image->header->groupy	= h5head.groupy;
		image->header->itype	= h5head.itype;
		image->header->exposure	= h5head.exposure;
		image->header->xSample	= h5head.xSample;			/* sample position */
		image->header->ySample	= h5head.ySample;
		image->header->zSample	= h5head.zSample;
		image->header->CCDy		= NAN;						/* invalid for new detectors */
		exH.depth				= h5head.depth;				/* mono energy (keV) */
		exH.energy				= h5head.energy;			/* mono energy (keV) */
		exH.scanNum				= h5head.scanNum;
		exH.beamBad				= h5head.beamBad;
		exH.lightOn				= h5head.lightOn;
		exH.hutchTemperature	= h5head.hutchTemperature;
		exH.sampleDistance		= h5head.sampleDistance;
		exH.CCDshutterIN		= h5head.CCDshutter;		/* CCD shutter, 1=IN, 0=OUT */
		strncpy(exH.userName,h5head.userName,MAX_micro_STRING_LEN);
		strncpy(exH.title,h5head.title,MAX_micro_STRING_LEN);
		strncpy(exH.sampleName,h5head.sampleName,MAX_micro_STRING_LEN);
		strncpy(exH.monoMode,h5head.monoMode,MAX_micro_STRING_LEN);
		strncpy(exH.beamline,h5head.beamline,MAX_micro_STRING_LEN);
		strncpy(exH.dateExposed,h5head.fileTime,MAX_micro_STRING_LEN);
		strncpy(exH.detector_ID,h5head.detector_ID,MAX_micro_STRING_LEN);
		if (maskFile[0]) {									/* there is a mask file, load it */
			if (readHDF5header(maskFile, &h5head)) { fprintf(stderr,"ERROR: unable to read HDF5 header, check permissions?\n"); exit(1); }
			if (HDF5ReadROIdouble(maskFile, "entry1/data/data", &bufMask, 0,h5head.xdim-1,0,h5head.ydim-1, &h5head)) { fprintf(stderr,"ERROR: unable to read HDF5 mask image\n"); exit(1); }
			mask = winview_image_new_empty();
			mask->data = grid_new((int)h5head.xdim,(int)h5head.ydim);
			mask->type = HDF5_FILE;							/* 0=spe, 1=hdf5 */
			mask->data->height		= (int)h5head.ydim;
			mask->data->width		= (int)h5head.xdim;
			mask->data->values		= bufMask;
		}
	}
	if (mask) {												/* check that an existing mask has correct size */
		if ((image->data->height != mask->data->height) || (image->data->width != mask->data->width)) {
			fprintf(stderr,"ERROR: size of mask (%d x %d) does not match size of image (%d x %d)\n",mask->data->height,mask->data->width,image->data->height,image->data->width);
			exit(1);
		}
		strncpy(exH.maskFile,maskFile,MAX_micro_STRING_LEN);/* save for later output */
	}
	else (exH.maskFile)[0] = '\0';

	#ifdef DEBUG
		printf("image wid=%d, height=%d\n",image->data->width, image->data->height);
		if (mask) printf("mask wid=%d, height=%d\n",mask->data->width, mask->data->height);
	#endif
	/*
	for(int i=0; i<image->data->width;i++){
		printf("#%d ",i);
		for(int j=0;j<image->data->height;j++)
		printf("%f ",grid_get_value(image->data,i,j));
		printf("\n");
	}
	*/
	if (hdf5) {
		(ginf->CCDFilename)[0] = '\0';				/* do not use default distortion file for HDF5 files */
	}
	else if (spe && strlen(ginf->CCDFilename)<1) {
		strncpy(ginf->CCDFilename,"./CCD_distorMay03_corr.dat",39);
		(ginf->CCDFilename)[39] = '\0';				/* ensure nulltermination */
	}
	#ifdef DEBUG
		printf("for ginf, using values:\n");
		printf("min_size = %lg\n",min_size);
		print_genfileinf(ginf);
		printf("\n");
	#endif

	/* compute statistics of image, average and standard deviation, and the threshold */
	long N = (image->data->width) * (image->data->height);

	



#ifndef USE_BOX
	if (smooth) grid_smooth_gauss(image->data, 2);	/* smooth the input image before processing */
#endif
	
	
	
	
	if (threshold!=threshold) {						/* no threshold given, compute it */
		long Nused=0;								/* number of pixels used, we do not use masked pixels */
		double	aboveAverage=NAN;					/* threshold = aboveAverage + average, this should be an input to the program */
		double Xi=0.0,Xi2=0.0, sigma;				/* Xi = Sum(pixels), Xi2 = Sum(pixels^2), sigma = standard deviation */
		for (i=0;i<N;i++) {
			maskValue = mask ? (mask->data->values)[i] : 0;
			pixel = (image->data->values)[i];		/* pixel value, skipping zero pixels, means that <100% reconstructions to not increment Nused */
			if (!maskValue && pixel!=0.0) {			/* skip masked pixels, do not use pixels with mask>0, use pixels with mask==0, skip zero pixels too */
				Xi += pixel;
				Xi2 += pixel*pixel;
				Nused++;							/* accumumlate number of used pixels */
			}
		}
		average = Xi/(double)Nused;
		sigma = sqrt((Xi2 - 2.0*Xi*average + Nused*average*average)/(double)Nused);
		aboveAverage = thresholdRatio*sigma;
		aboveAverage = (aboveAverage == aboveAverage) ? aboveAverage : 5*average;
		if (average<0.0) {
			average = 0.0;
			#ifdef DEBUG
				printf("average = %g, resetting it to 0.0\n", average);
			#endif
		}
		threshold = average + aboveAverage;
		#ifdef DEBUG
			printf("for an average pixel value of %g, with standard deviation=%g,  using a threshold of (%g + %g) = %g\n",average,sigma,average,aboveAverage,threshold);
			printf("\n");
		#endif
	}
	else average = threshold - fabs(0.99*threshold);	/* set this value just below the threshold, needed to set masked out pixels */

	saturation_level = itype2saturation(image->header->itype);
	for (numAboveThreshold=sumAboveThreshold=sum=0.0, i=0;i<N;i++) {
		maskValue = mask ? (mask->data->values)[i] : 0;
		if (maskValue) {							/* set masked out pixels to average value */
			(image->data->values)[i] = average;
		}
		else {										/* accumumlate statistics only on used pixels */
			pixel = (image->data->values)[i];
			sum += pixel;
			if (pixel > threshold) {
				sumAboveThreshold += pixel;
				numAboveThreshold++;
			}
		}
	}
	exH.sumAboveThreshold = sumAboveThreshold;
	exH.numAboveThreshold = numAboveThreshold;
	exH.sum = sum;
	exH.NpeakMax = NpeakMax;

	#ifdef USE_BOX
		GridB* maskG = gridB_new(image->data->width, image->data->height);		/* make the maskG, this also sets all values to 0 */
		peaks = boxsearch(image->data, maskG, boxsize, NpeakMax, smooth, ginf);	/* list of peaks */
		list_delete_nodes((void *)maskG);
	#else
		/* if (smooth) grid_smooth_gauss(image->data, 2);						// smooth the input image before processing */
		/* if (smooth) grid_smooth_boxcar(image->data, 2); */
		/* if (smooth) grid_smooth_boxcar(image->data, 1); */
		/* if (smooth) grid_smooth_median(image->data, 1);						// median smooth, uses a 3x3 box to get rid of isolated noise spikes */
		List* blobs = blobsearch(image->data, threshold, (int)min_size, true);
		#ifdef DEBUG
			ListNode* blob = blobs->head;
			i = 0;
			printf("  X \t\t  Y  \t\tValue\t\t\tnumber of blobs: %d\n", blobs->size);
			while (blob != EMPTY_NODE && i++<30) {
				printf("%.1f  \t%.1f   \t%.0f\n", ((Point*)blob->value)->x, ((Point*)blob->value)->y,((Point*)blob->value)->value);
				blob = blob->next;
			}
		#endif

	// #warning "Added sorListPoints() to speed up program when threshold is too low"
	sorListPoints(blobs);		/* sort Points in blobs so that they are ordered from most to least intens */

//printf("\nstart processBlobs at %.2f seconds with %d blobs\n",((double)(clock() - tstart)) /((double)CLOCKS_PER_SEC),blobs->size);
		peaks = processBlobs(blobs,image,ginf,NpeakMax);	/* list of peaks */
//printf("\nfinish processBlobs at %.2f seconds\n",((double)(clock() - tstart)) /((double)CLOCKS_PER_SEC));
		list_delete_nodes(blobs);
	#endif

	#ifdef DEBUG
		ListNode* peak = peaks->head;
		i = 0;
		printf("\n fitX \t\t fitY  \t Intens \tdX		dY		chisq\t\tfitted %d peaks\n", peaks->size);
		while (peak != EMPTY_NODE && i++ < 30) {
			printf("% 6.1f  \t% 6.1f   \t% 7.0f	%.1f  \t%.1f   \t%.3f\n", \
				((Peak*)peak->value)->fitX, ((Peak*)peak->value)->fitY,((Peak*)peak->value)->fitIntens, \
				((Peak*)peak->value)->fitPeakWidthX, ((Peak*)peak->value)->fitPeakWidthY,((Peak*)peak->value)->chisq);
			peak = peak->next;
		}
	#endif

	peaks = removeNearbyPeaks(peaks, minSeparation);
	#ifdef DEBUG
		printf("\nremoved peaks that are too close, now number of peaks = %d\n",peaks->size);
		peak = peaks->head;
		i = 0;
		printf(" fitX \t\t fitY  \t Intens \tdX		dY		chisq\t\t%d acceptable peaks\n", peaks->size);
		while (peak != EMPTY_NODE && i++ < 30) {
			printf("% 6.1f  \t% 6.1f   \t% 7.0f	%.1f  \t%.1f   \t%.3f\n", \
				((Peak*)peak->value)->fitX, ((Peak*)peak->value)->fitY,((Peak*)peak->value)->fitIntens, \
				((Peak*)peak->value)->fitPeakWidthX, ((Peak*)peak->value)->fitPeakWidthY,((Peak*)peak->value)->chisq);
			peak = peak->next;
		}
		printf("\n");						/* the following 2-column part makes it easy to pate into Igor */
		peak = peaks->head;
		i = 0;
		while (peak != EMPTY_NODE && i++ < 50) {
			printf("%.1f	%.1f\n", ((Peak*)peak->value)->fitX, ((Peak*)peak->value)->fitY);
			peak = peak->next;
		}
	#endif

	seconds = ((double)(clock() - tstart)) /((double)CLOCKS_PER_SEC); 
	#ifdef DEBUG
		printf("\ntotal execution time for this process was %.1f seconds\n",seconds);
	#endif

	#if IDL_FILES
	savePeaksIDL(peaks,outFile);
	#else
	savePeaks(peaks,outFile,image->header,inFile,ginf,threshold,seconds,minSeparation,smooth,&exH,argv[0]);
	#endif

	delete_genfileinf(ginf);
	winview_image_delete(image);

#warning "list_delete_nodes() may also be wrong, does it first delete the allocated space inside, but probably does not matter in this program?"
	list_delete_nodes(peaks);

	return 0;
}



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
double itype2saturation(int itype)		/* returns largest value for an itype */
{
	switch (itype) {
		case 6:			/* signed int8 (1 byte) */
			return (double) ((1 << 7) - 1);			/* 2^7 - 1 = 127 */
		case 4:			/* string/char (1 byte) */
		case 7:			/* unsigned int8 (1 byte) */
			return (double) ((1 << 8) - 1);			/* 2^8 - 1 = 255 */
		case 2:			/* integer (2 byte) */
			return (double) ((1 << 15) - 1);		/* 2^15 - 1 = 32767 */
		case 3:			/* unsigned integer (2 byte) */
			return (double) ((1 << 16) - 1);		/* 2^15 - 1 = 65535 */
		case 1:			/* long integer (4 byte) */
			return (double) ((1 << 31) - 1UL);		/* 2^31 - 1 = 2147483647 */
		case 0:			/* float (4 byte) */
		case 5:			/* double (8 byte) */
			return INFINITY;
	}
	return NAN;
}

