/*
 Routines for reading in and looking at Roper (formerly Princeton Instruments) 
 '*.spe' files from a CCD.
 by Jon Tischler, Oak Ridge National Lab
 TischlerJZ@ornl.gov
*/

#include "WinViewRead.h"



/* #define DEBUG		/* debug flag, enable for messages */

long byteSwap2(void *j);
void byteSwapArray(size_t wordLen, size_t words, void *a);


/* takes a pointer to a buffer of a WinView header and returns values in the structure head */
/* this routine does not open or access any files, to open the file too, use WinViewReadHeader */
int WinViewParseHeader(
char	*buf,						/* buffer containing the first part of a WinView file */
WinViewHeader *head)			/* WinView header information */
{
	/* local variables */
	char	*pvs;						/* local value of PVlist[] */
	char	*ptr;						/* a pointer into buf */
	int		bkgApplied;					/* was a background correction already applied */
	float	expose;						/* exposure time (sec) */
	float	*pexpose;					/* pointer used to fine exposure time */
	long	j;

	if (!buf) return 1;					/* empty buffer */

#ifdef DEBUG
	printBufferHex(buf,1,2,21*2);
	printBufferHex(buf,1,2,328*2);
	printBufferHex(buf,1,2,108);
	printBufferHex(buf,1,2,150);
#endif

	head->xdim = byteSwap2(buf+2*21);	/* x,y dimensions of image (after any internal binning */
	head->ydim = byteSwap2(buf+328*2);
	head->itype = byteSwap2(buf+108);	/* type of numbers in image, eg floar, int, ... */
	bkgApplied = byteSwap2(buf+150);	/* set to 1 if background subtraction done */

	(head->bkgFile)[0]='\0';
	if (bkgApplied) {
		strncpy(head->bkgFile,(char *)(buf+2*1752),120);
		onlyNamePartOfFile(head->bkgFile,head->bkgFile);
	}

	head->xDimDet = byteSwap2(buf+6);			/* size of the actual chip (pixels) */
	head->yDimDet = byteSwap2(buf+18);
	head->ADCrate = byteSwap2(buf+190);			/* ADC rate */
	head->ADCtype = byteSwap2(buf+192);			/* ADC type */
	head->ADCresolution = byteSwap2(buf+194);	/* ADC resolution */
	head->geo_rotate = (buf[600] & 1);			/* geometric 1==rotate, 2==reverse, 4==flip */
	head->geo_reverse = (buf[600] & 2);
	head->geo_flip = (buf[600] & 4);
	head->controllerType = byteSwap2(buf+704);	/* flag indicating controller type  (see WinViewControllers) */

	/* for exposure time use float (4bytes starting at byte10) */
	pexpose = (float *)&(buf[10]);
	expose = *pexpose;
	#ifdef __BIG_ENDIAN__
	byteSwapArray(4,1,&expose);
	#endif
	head->exposure = expose;

	j = byteSwap2(buf+1510);					/* number of ROI's */
	if (j>1) return 4;							/* too many ROIi's in this file */

	head->startx = byteSwap2(buf+1512);			/* read the first ROI information */
	head->endx = byteSwap2(buf+1514);
	head->groupx = byteSwap2(buf+1516);
	head->starty = byteSwap2(buf+1518);
	head->endy = byteSwap2(buf+1520);
	head->groupy = byteSwap2(buf+1522);

	/* now deal with PV's passed in the image */
	/* read in each of the 5 lists, they start at 200, every 80 (200,280, 360, 440, 520) */
	pvs = head->PVlist;
	pvs[0] = '\0';
	for (j=200;j<521;j+=80) {					/* byte index into the file */
		ptr = strchr(&(buf[j]),' ');			/* points to first space */
		if (!ptr) continue;
		strncat(pvs,++ptr,EPICS_MAX_STRLEN);	/* concatenate everything after the space */
		strcat(pvs,";");
	}
	j = strlen(pvs);
	if(pvs[j-1]==';') pvs[j-1] = '\0';
	head->xSample = NumberByKey("X1",pvs,0,0);
	head->ySample = NumberByKey("Y1",pvs,0,0);
	head->zSample = NumberByKey("Z1",pvs,0,0);
	head->xWire = NumberByKey("X2",pvs,0,0);
	head->yWire = NumberByKey("Y2",pvs,0,0);
	head->zWire = NumberByKey("Z2",pvs,0,0);
	head->CCDy = 1000.*NumberByKey("CCDy",head->PVlist,0,0);	/* convert mm to micron */

	/* done parsing header information */
	return 0;
}


/* read a WinView .spe file's header information (not image) and returns values */
/* to read in the image, use WinViewReadROI */
int WinViewReadHeader(
FILE	*fid,						/* file id of input file */
WinViewHeader *head)			/* WinView header information */
{
	/* local variables */
	char	*buf;						/* char pointer to vbuf */
	fpos_t 	fpos;						/* initial file position */
	int		i;

	buf = NULL;
	buf = calloc(2000, 1);				/* I depend upon calloc to initialize to zero */
	if (!buf) exit(ENOMEM);				/* allocation error */

	fgetpos(fid, &fpos);				/* save current file position */
	fseek(fid,0,SEEK_SET);
	if (fread(buf,1,2000,fid)<100) { free(buf); return 2; }	/* read error */
	fsetpos(fid,&fpos);					/* reset to original file position */

	i = WinViewParseHeader(buf,head);
	if (i) return i;
	free(buf);
	/* done reading header information */

#ifdef DEBUG
	{
	char	stype[256];
	if (strlen(head->bkgFile)) printf(",             background file = '%s'\n",head->bkgFile);
	else printf("\n");
	printf("xdim = %ld      ydim = %ld      ", head->xdim,head->ydim);
	printf("total length = %ld x %ld  = %ld pixels\n", head->xdim,head->ydim,(head->xdim)*(head->ydim));
	printf("number type is  '%s'\n",WinViewFileTypeString(head->itype,stype));
	}
#endif

	return 0;
}


/* read a WinView .spe file image part.  To get header information, first call WinViewReadHeader */
/* the image is in vbuf, it is ordered with x moving fastest */
int WinViewReadROI(
FILE	*fid,						/* file id of input file */
void	**vbuf,						/* pointer to image */
size_t	xlo,						/* reads region (xlo,ylo) to (xhi,yhi) */
size_t	xhi,						/* if xhi or yhi are <0, then it uses xdim or ydim */
size_t	ylo,						/* note max yhi is ydim-1, and ditto for x */
size_t	yhi,
WinViewHeader *head)			/* WinView header information (if NULL, read header) */
{
	/* local variables */
	char	*buf;						/* char pointer to image locally */
	char	*ystart;					/* points to start of each y-value */
	char	*next;
	size_t	xdim;						/* x,y dimensions of image in file */
	size_t	ydim;
	size_t	pixels;						/* number of pixels, = xdim*ydim */
	long	i;
	size_t	offset;
	size_t	nx;							/* xhi - xlo + 1 */
	size_t	ny;							/* yhi - ylo + 1 */
	size_t	xbytes;
	size_t	ilen;						/* number of bytes per pixel */
	fpos_t 	fpos;						/* initial file position */
	int		itype;						/* WinView file type */
	WinViewHeader headLocal;		/* WinView header information, used for local read */

	if (!head) {
		i = WinViewReadHeader(fid,&headLocal);	/* read header information */
		xdim = headLocal.xdim;
		ydim = headLocal.ydim;
		itype = headLocal.itype;
		if (i) return i;
	}
	else{
		xdim = head->xdim;
		ydim = head->ydim;
		itype = head->itype;
	}
	if (xlo<0 || ylo<0 || xlo>xhi || ylo>yhi || xhi>xdim-1 || yhi>ydim-1) {
		/* no image to read, invalid range */
		*vbuf = NULL;
		return 2;						/* nothing read in */
	}
	if (!(ilen=WinView_itype2len(itype))) return 3;	/* get length of word from WinView itype */
	pixels = xdim * ydim;				/* total number of pixels in the image */

	xhi = (xhi<0 || xhi>(xdim-1)) ? xdim-1 : xhi;	/* xhi is now actual to use */
	yhi = (yhi<0 || yhi>(ydim-1)) ? ydim-1 : yhi;
	xlo = (xlo>xhi) ? xhi : xlo;
	ylo = (ylo>yhi) ? yhi : ylo;
	xlo = (xlo<0) ? 0 : xlo;
	ylo = (ylo<0) ? 0 : ylo;
	nx = xhi - xlo + 1;
	ny = yhi - ylo + 1;

	/* now set up to read the image itself */
	pixels = ny*xdim;				/* num of pixels to read in, will reduce it later if nx<xdim */
#ifdef DEBUG
	printf("in WinViewReadROI:\n");
	printf("    about to try to allocate image space of %lu Kbytes\n",pixels*ilen/1024);
#endif
	buf = calloc(pixels,ilen);
	if (!buf) exit(ENOMEM);					/* allocation error */
#ifdef DEBUG
	printf("    image buffer = buf = %p,   xdim = %ld, ydim = %ld\n",buf,xdim,ydim);
	printf("    [xlo,xhi]=[%ld,%ld] nx=%ld,  [ylo,yh]=[%ld,%ld] ny=%ld,  read %ld pixels\n",xlo,xhi,nx,ylo,yhi,ny,pixels);
#endif

	offset = HEADER_LENGTH + ilen*ylo*xdim;	/* offset (bytes) to start of roi, HEADER_LENGTH is to start of image */
	fgetpos(fid,&fpos);					/* save current file position */
	if (fseek(fid,offset,SEEK_SET)) { free(buf); *vbuf=NULL; return 6; }	/* seek error */
	i = fread(buf,ilen,pixels,fid);
#ifdef DEBUG
	printf("    tried to read %ld pixels (each is %lu bytes) at offset of %lu, it read %ld items\n",pixels,ilen,offset,i);
#endif
	if (i<pixels) { free(buf); *vbuf=NULL; return 4; }		/* read error */
	fsetpos(fid,&fpos);					/* reset to original file position */

	/* compress up in the x direction */
	if (nx<xdim) {
		xbytes = ilen * xdim;			/* number of bytes in each row before compression */
#ifdef DEBUG
	printf("        xbytes = %ld\n",xbytes);
#endif
		ystart = buf + xbytes;			/* points to start next y-value */
		next = buf + nx*ilen;			/* points to the next place in reduced array */
		for (i=0;i<ny;i++) {			/* loop over each y-value of the array */
			memmove(next, ystart+xlo*ilen, nx*ilen);
#ifdef DEBUG
	printf("        memmove(%p,%p,%lu) == memmove(%d+buf,%d+buf,%lu)\n",next, ystart+xlo*ilen, nx*ilen,next-buf, ystart+xlo*ilen-buf, nx*ilen);
#endif
			ystart += xbytes;			/* points to start next y-value */
			next += nx*ilen;			/* points to the next place in reduced array */
		}
		buf = (char *)realloc(buf,nx*ny*ilen);	/* reduce allocation */
	}
	pixels = nx*ny;

	#ifdef __BIG_ENDIAN__
	byteSwapArray(ilen,pixels,buf);
	#endif
	*vbuf = buf;
	return 0;
}



/* write an ROI into a WinView file, does not write the header */
int WinViewWriteROI(
FILE	*fid,						/* file id of input file */
char	*vbuf,						/* the image ROI to write */
int		itype,						/* WinView image itype (not length) */
size_t	xdim,						/* xdim of full image in file */
size_t	xlo,						/* write region (xlo,ylo) to (xhi,yhi) */
size_t	xhi,						/* if xhi or yhi are <0, then it uses xdim or ydim */
size_t	ylo,						/* note max yhi is ydim-1, and ditto for x */
size_t	yhi)
{
	long	fpos0;					/* original position in the file */
	size_t	dx;						/* number of x pixels in ROI, xhi-xlo+1 */
	size_t	ilen;					/* number of bytes per pixel */
	size_t	length;					/* current length of file */
	size_t	offset;					/* offset to start of ROI */
	size_t	pixels;					/* number of pixels in ROI */
	char 	*buf=NULL;				/* temp buffer for padding file */
	long	i;						/* result of fwrite() */
	long	y;						/* index of each y in slow writing part */
	long	more;					/* more length needed for the slow writing */
	int		err=0;					/* error value */

	fpos0 = ftell(fid);				/* store original position */
	ilen = WinView_itype2len(itype);
	dx = xhi-xlo+1;							/* number of x values in ROI */
	if (dx>xdim || xhi>=xdim) {fprintf(stderr,"asked to write [xlo,xhi]=[%ld,%ld] with xdim=%ld\n",xlo,xhi,xdim); return 1;}
	pixels = (xhi-xlo+1)*(yhi-ylo+1);

	fseek(fid,0,SEEK_END);					/* goto end of file */
	length = ftell(fid);					/* current length of file */
	offset = HEADER_LENGTH+ilen*xdim*ylo;	/* start of ROI */

	/* is it long enough?  needs to be at least offset bytes long */
	if (length < offset ) {
		buf = calloc(offset,sizeof(char));
		if (!buf) {fprintf(stderr,"could not allocate temp buffer %ld bytes in WinViewWriteROI\n",offset); exit(ENOMEM);}
		i = fwrite(buf,sizeof(char),offset,fid);
		free(buf);
		if (i!=offset) { fprintf(stderr,"error writing buffer in WinViewWriteROI, only wrote %ld of %ld bytes",i,offset); return 1; }
	}

	#ifdef __BIG_ENDIAN__
	byteSwapArray(ilen,pixels,vbuf);		/* swap on way in */
	#endif

	if (dx==xdim) {							/* an easy write xlo=0 and xhi=xdim-1 */
		fseek(fid,offset,SEEK_SET);			/* goto start of ROI */
		i = fwrite(vbuf,ilen,pixels,fid);
		if (i!=pixels) { fprintf(stderr,"WinViewWriteROI, only wrote %ld of %ld pixels\n",i,pixels); err=1; goto wayOut;}
	}
	else {									/* must write each line separately */
		more = (offset+(yhi-ylo+1)*xdim*ilen) - length;
		if (more>0) {						/* first extend the file more if needed */
			buf = calloc(more,sizeof(char));
			if (!buf) {fprintf(stderr,"could not allocate temp buffer %ld bytes in WinViewWriteROI\n",more); err=1; exit(ENOMEM);}
			i = fwrite(buf,sizeof(char),more,fid);
			free(buf);
			if (i!=more) { fprintf(stderr,"error writing more in WinViewWriteROI, only wrote %ld of %ld bytes",i,more); err=1; goto wayOut;}
		}
		offset += xlo;						/* start of ROI */
		for (y=ylo;y<=yhi;y++) {			/* for each of the y lines */
			fseek(fid,offset,SEEK_SET);		/* goto start of ROI */
			i = fwrite(vbuf,ilen,dx,fid);
			if (i!=dx) { fprintf(stderr,"error writing line of ROI in WinViewWriteROI, only wrote %ld of %ld pixels",i,dx); err=1; goto wayOut;}
			offset += xdim*ilen;
			vbuf += dx*ilen;
		}
		free(buf);
	}

	wayOut:
	#ifdef __BIG_ENDIAN__
	byteSwapArray(ilen,pixels,vbuf);		/* swap back */
	#endif
	fseek(fid,fpos0,SEEK_END);				/* goto original point of file */
	return err;
}



long byteSwap2(					/* convert 2-byte unsigned swapped to long */
void	*v)						/* pointer to word to be swapped */
{
	short unsigned int *i2;
	long	i4;
	#ifdef __BIG_ENDIAN__		/*	byteSwapArray(2,1,v); */
	char	c;
	char	b[4];				/* work space */
	memcpy(b,v,4);
	c = b[0];
	b[0] = b[1];
	b[1] = c;
	i2 = (short unsigned int *)b;
	#else
	i2 = (short unsigned int *)v;
	#endif
	i4 = *i2;					/* convert unsigned 2-byte int to signed long */
	return i4;
}
/*long byteSwap2(					/* convert 2-byte unsigned swapped to long */
/*void	*j)						/* pointer to word to be swapped */
/*{
/*	short unsigned int *i2;
/*	long	i4;
/*
/*	#ifdef __BIG_ENDIAN__
/*	byteSwapArray(2,1,j);
/*	#endif
/*	i2 = (short unsigned int *)j;
	/*	i4 = *i2;					/* convert unsigned 2-byte int to signed long */
/*	return i4;
/*}
 */

void byteSwapArray(				/* swap bytes of pointer v (n bytes long) */
size_t	wordLen,				/* length of each word (in bytes) to swap */
size_t	words,					/* number of words to swap, starting at v */
void	*v)						/* pointer to first byte of v to start swapping */
{
	int		i;
	size_t	j;
	char	c;
	char	*b;
	int		halfLen;

	if (words<=0 || wordLen<=1) return;				/* do nothing */
	halfLen = wordLen/2;
	b = v;
	for (j=0;j<words;b+=wordLen,j++) {				/* for each word in array b */
		for (i=0;i<halfLen;i++) {					/* swap wordLen bytes starting at b */
			c = b[i];
			b[i] = b[wordLen-1-i];
			b[wordLen-1-i] = c;
		}
	}
	return;
}


#ifdef DEBUG
#define NUM_PER_LINE	20
void printBufferHex(		/* print n values of buffer starting at offset */
void	*buffer,
size_t	bufSizeOf,			/* size of each element in buffer */
size_t	n,					/* number of words (of size bufSizeOf) to show */
size_t	offset)				/* offset into buffer (in units of bufSizeOf) */
{
	unsigned char	*byte;			/* byte pointer to buffer */
	long	j;						/* index into byte for the start of line */
	long	k;						/* index into byte for the last value in the line to print */
	long	i;						/* index into byte for each item in the line */

	byte = (unsigned char *)buffer;	/* a char pointer to buffer */
	j = offset*bufSizeOf;
	n = bufSizeOf*(n+offset);		/* index into byte of the very last byte to print */

	k = ((j+NUM_PER_LINE-1)>(n-1)) ? (n-1) : (j+NUM_PER_LINE-1);	/* how far to print on this line */
	while (j<n) {
		printf("buffer[%03ld,%03ld] = ",j/bufSizeOf,k/bufSizeOf);	/* label range printed */
		for (i=j;i<=k;i++) printf("%02X, ",byte[i]);				/* for each byte on this line */
		printf("\n");
		j = k+1;
		k = ((j+NUM_PER_LINE-1)>(n-1)) ? n-1 : j+NUM_PER_LINE-1;
	}
}
#endif


char *onlyNamePartOfFile(		/* retrun only the name part of a file */
char	*full,					/* input string with full path name */
char	*name)					/* output string to recieve the name part */
{
	char	*p0;				/* start of name */
	char	*p1;				/* end of name */
	char	*pp;				/* temp pointer */
	size_t	len;				/* length of full */
	char	nameTemp[256];		/* temp for name */

	len = strlen(full);
	if (!len) { name[0]='\0'; return name; }

	p0 = strrchr(full,'\\');		/* windows */
	pp = strrchr(full,':');			/* mac */
	p0 = (p0>pp) ? p0 : pp;
	pp = strrchr(full,'/');			/* unix */
	p0 = (p0>pp) ? p0 : pp;			/* p0 is now the first character in name */
	if (p0==NULL) { name[0]='\0'; return name; }

	p1 = strstr(full,".SPE");		/* now remove .spe if present */
	pp = strstr(full,".spe");
	p1 = (p1>pp) ? p1 : pp;
	if ((full+len-4) != p1) p1 = full+len-1;

	if (!p0 || !p1 || p0>p1 || ((p1-p0+1)>len)) { name[0]='\0'; return name; }
	len = p1-p0+1;
	strncpy(nameTemp,p0,len);
	return strcpy(name,nameTemp);
}




char *WinViewControllers(	/* returns a descriptive string of the CCD controller */
int		itype,				/* WinView file type */
char	*stype)				/* to recieve descriptive string make at least 51 long */
{
	char	*name[]={"new120 (TYPE II)","old120 (TYPE I)","ST130","ST121","ST138","DC131 (Pentamax)","ST133 (MicroMAX/SpectroMAX)","ST135 (GPIB)","VICCD","ST116 (GPIB)","OMA3 (GPIB)","OMA4"};
	if (itype<0 || itype>10) stype[0]='\0';
	else strcpy(stype,name[itype]);
	return stype;
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
char *WinViewFileTypeString(/* puts a descriptive string into stype */
int		itype,				/* WinView file type */
char	*stype)				/* to recieve descriptive string make at least 30 long */
{
	char	*name[]={"float (4 byte)","long integer (4 byte)","integer (2 byte)","unsigned integer (2 byte)",\
					"string/char (1 byte)","double (8 byte)","signed int8 (1 byte)","unsigned int8 (1 byte)"};
	if (itype<0 || itype>7) stype[0]='\0';
	else strncpy(stype,name[itype],30);
	return stype;
}


int	WinView_itype2len(		/* convert WinView file type to number of bytes/pixel, 0 is for error*/
int		itype)		/* WinView itype */
{
	int		ilen;

	switch (itype) {
		case 4:			/* string/char (1 byte) */
		case 6:			/* signed int8 (1 byte) */
		case 7:			/* unsigned int8 (1 byte) */
			ilen=1;
			break;
		case 2:			/* integer (2 byte) */
		case 3:			/* unsigned integer (2 byte) */
			ilen=2;
			break;
		case 0:			/* float (4 byte) */
		case 1:			/* long integer (4 byte) */
			ilen=4;
			break;
		case 5:			/* double (8 byte) */
			ilen=8;
			break;
		default:
			ilen = 0;	/* error */
	}
	return ilen;
}





#define PRINT_SIZE(TYPE,NAME,WANT) printf("sizeof(%s) = %ld, should be %d\n",NAME,sizeof(TYPE),WANT);
#define COMPARE_SIZE(TYPE,NAME,WANT) { if (sizeof(TYPE)!=WANT) { bad=1; PRINT_SIZE(TYPE,NAME,WANT) } }

int	checkTypeSizes(void)			/* check the sizes of the number types */
{
	int		bad=0;

#ifdef DEBUG
	PRINT_SIZE(char,"char",1)
	PRINT_SIZE(int,"int",4)
	PRINT_SIZE(unsigned int,"unsigned int",4)
	PRINT_SIZE(short int,"short int",2)
	PRINT_SIZE(short unsigned int,"short unsigned int",2)
	PRINT_SIZE(short,"short",2)
	PRINT_SIZE(long,"long",4)
	PRINT_SIZE(unsigned long,"unsigned long",4)
	PRINT_SIZE(float,"float",4)
	PRINT_SIZE(double,"double",8)
	PRINT_SIZE(size_t,"size_t",4)
#endif

	COMPARE_SIZE(char,"char",1)
	COMPARE_SIZE(int,"int",4)
	COMPARE_SIZE(unsigned int,"unsigned int",4)
	COMPARE_SIZE(short int,"short int",2)
	COMPARE_SIZE(short unsigned int,"short unsigned int",2)
	COMPARE_SIZE(short,"short",2)
	COMPARE_SIZE(long,"long",4)
	COMPARE_SIZE(unsigned long,"",4)
	COMPARE_SIZE(float,"float",4)
	COMPARE_SIZE(double,"double",8)
	COMPARE_SIZE(size_t,"size_t",4)
	return bad;
}




/*
Returns a number from a keyed list.  If the key does not exist in the list, then NaN is returned.  
Each item in the list is of the form key=value.  Where the equal sign is acutally the character 
given by keySepChar.  If keySepChar is less then or equal to 0, then an equal sign is assumed. 
Each key=value pair is separated by the listSepChar character, so the list is a string that 
looks like:   key1=val1;key2=val2;...keyN=valN.  The key string is limited to 200 characters 
long (limitation is from StringByKey()).  An example of how it might be used is:

	printf("Y1 = %g\n",NumberByKey("Y1",PVlist,0,0));

if PVlist = "X1=123;Y1=567.200;Z1=678.9", the result will be:
	
 >	Y1 = 567.2
*/
double NumberByKey(			/* return a number from a keyed list "key1=val1;key2=val2;...keyN=valN"*/
char	*key,				/* key string to match */
char	*list,				/* list */
char	keySepChar,			/* separator character between key and value, <= means use "=" */
char	listSepChar)		/* separator character between list elements, <=0 means use ";" */
{
		double	value;				/* the result */
		char	*ptr;				/* pointer into list, points to start of value */

		ptr = StringByKey(key,list,keySepChar,listSepChar,30);
		if (ptr) {
			value = atof(ptr);
			free(ptr);
		}
		else value=NAN;				/* if NAN not defined, you can use sqrt(-1.0) */
		return value;
}


/*
Returns a long int from a keyed list.  The returned value is rounede, not truncated.  If the key 
does not exist in the list, then the integer version of NaN is returned (-2147483648 ?).  
Each item in the list is of the form key=value.  Where the equal sign is acutally the character 
given by keySepChar.  If keySepChar is less then or equal to 0, then an equal sign is assumed. 
Each key=value pair is separated by the listSepChar character, so the list is a string that 
looks like:   key1=val1;key2=val2;...keyN=valN.  The key string is limited to 200 characters 
long (limitation is from StringByKey()).  An example of how it might be used is:

	printf("Y1 = %ld\n",IntByKey("Y1",PVlist,0,0));

if PVlist = "X1=123;Y1=567.800;Z1=678.9", the result will be:
	
 >	Y1 = 568
*/
long IntByKey(				/* return an integer from a keyed list "key1=val1;key2=val2;...keyN=valN"*/
char	*key,				/* key string to match */
char	*list,				/* list */
char	keySepChar,			/* separator character between key and value, <= means use "=" */
char	listSepChar)		/* separator character between list elements, <=0 means use ";" */
{
		return (int)floor(NumberByKey(key,list,keySepChar,listSepChar)+0.5);
}


/*
Return a pointer to the value in a keyed list.  This routine allocates the space for the result.  So do not 
forget to free the space when you are done!.  Each item in the list is of the form key=value.  Where the equal 
sign is acutally the character given by keySepChar.  If keySepChar is less then or equal to 0, then an equal 
sign is assumed. Each key=value pair is separated by the listSepChar character, so the list is a string that
looks like:   key1=val1;key2=val2;...keyN=valN.  The key string is limited to 200 characters long (limitation 
is from size of search),  and the returned value is limited to maxLen-1 characters (maxLen if you count 
terminating null).  An example of how it might be used is:

	char *temp;
	temp = StringByKey("Z1",PVlist,0,0,30);
	if (temp) {
		printf(" as a string, Z1 = '%s'\n",temp);
		free(temp);
	}
	else printf(" as a string, Z1 is NULL\n");
*/
char* StringByKey(			/* return a pointer to a string from a keyed list "key1=val1;key2=val2;...keyN=valN"*/
char	*key,				/* key string to match */
char	*list,				/* list */
char	keySepChar,			/* separator character between key and value, <= means use "=" */
char	listSepChar,		/* separator character between list elements, <=0 means use ";" */
int		maxLen)				/* maximum length allowed for the result */
{
	char	search[256];		/* search string */
	long	n;					/* length of search string */
	char	*ptr;				/* pointer into list, points to start of value */
	char	*ptr2;				/* pointer to end of value */
	char	*result;			/* answer, allocate space for it too */
	long	nr;					/* length of result, not > maxLen */

	if (maxLen<1) return NULL;
	if (strlen(key)>250) return NULL;		/* key string is too long */
	listSepChar  = (listSepChar<=0) ? ';' : listSepChar;
	keySepChar  = (keySepChar<=0) ? '=' : keySepChar;
	result = NULL;

	search[0] = listSepChar;				/* search = ";key=" */
	search[1] = '\0';
	strcat(search,key);
	n = strlen(search);
	search[n++] = keySepChar;
	search[n] = '\0';

	if ((ptr=strstr(list,search))) ptr += n;			/* in the middle of the list, found ";key=" */
	else if (list==(ptr=strstr(list,search+1))) ptr += (n-1);	/* at the beginning of the list, found "key=" */
	else return NULL;

	ptr2 = strchr(ptr,listSepChar);						/* points to list separator after value */
	if (!ptr2) nr = strlen(ptr);						/* assume value goes to end of string */
	else nr = ptr2-ptr;
	if (nr < 1) return NULL;							/* nothing there */
	if ((nr+1)>maxLen) return NULL;						/* value too long */

	result = (char*)calloc((size_t)nr+1,1);				/* space for value and terminating NULL */
	if (!result) exit(ENOMEM);							/* allocation error */
	strncpy(result,ptr,(size_t)nr);
	return result;
}







