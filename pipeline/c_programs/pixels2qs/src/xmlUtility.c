/*
 *  xmlUtility.c
 *  xmlUtility
 *
 *  Created by Jon Tischler on 7/6/11.
 *  Copyright 2011 Oak Ridge National Laboratory. All rights reserved.
 *
 */

/* Reading Generic XML
 *
 *	XML support		(occurance optionally allows selecting the the occuranceth 
 *					instance of tag), note vectors usually delimited by a space
 *
 *	XMLNodeList(buf)									returns a list with all top level nodes in buf
 *	XMLtagContents(tag,buf,[occurance])					returns the contents of tag
 *	XMLtagContents2ListDouble(tag,buf,occurance,delimiters,** vec)	returns the contents of tag as a list,
 *	XMLtagContents2ListFloat(tag,buf,occurance,delimiters,** vec)	  useful for vectors in the contents
 *	XMLtagContents2ListInt(tag,buf,occurance,delimiters,** vec)
 *	XMLattibutes2KeyList(tag,buf)						return a string list with all of the attribute value pairs for tag
 *	XMLremoveComments(str)								remove all xml comments from str
 */

#include "xmlUtility.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


/* local functions */
char *startOfxmltag(char* tag, char* buf, int occurance);
char* TrimFrontBackWhiteSpace(char *str);
char* TrimTrailingWhiteSpace(char *str);
char* TrimLeadingWhiteSpace(char *str);



int isXMLfile(char *buf)
{
	char	*p;
	p = strstr(buf,"<?xml");
	if (buf!=p) return 0;
	p = strstr(buf,"?>");
	return p>buf;
}




/* returns an array of char*, each char* is a node names at top most level in buf */
/* the result is allocated space here, remember to deallocate it yourself when done */
/* the last node is a NULL, to tell you where the end is */
long	XMLNodeList(
char	*buf,
char	***nodes)					/* used to hold array of node names */
{
	char*	i0;
	char*	i1;
	char*	i2;
	char*	c;
	char	name[256];				/* an xml name */
	size_t	len;
	size_t	nAlloc;					/* number of nodes allocated */
	size_t	N=0;					/* number of nodes used */

	nAlloc = 100;
	*nodes = calloc(nAlloc,sizeof(char*));

	i0 = buf;
	while(i0) {
		i0 = strchr(i0,'<');		/* find start of a tag */
		if (!i0) break;

		if (strstr(i0,"<!--")==i0) {/* this is a comment, skip it */
			i0 = strstr(i0,"-->");
			continue;				/* goto top of loop and find next tag */
		}
		i0++;						/* skip the '<' */

		i1 = strchr(i0,' ');		/* find end of tag name using i1 & i2, end will be in i1 */
		i2 = strchr(i0,'>');
		i1 = !i1 ? i2 : i1;			/* if i1==NULL, use i2 */
		i2 = !i2 ? i1 : i2;			/* if i2==NULL, use i1 */
		i1 = i1<i2 ? i1 : i2;		/* use min of i1 & i2 */
		if (!i1) break;

		len = i1-i0;
		len = len>250 ? 250 : len;	/* make sure len is not too long */
		strncpy(name,i0,len);		/* put name into name */
		name[len] = '\0';

		if (N>=nAlloc) {			/* need to allocate more space */
			nAlloc += 100;
			*nodes = realloc(*nodes,nAlloc*sizeof(char*));
		}
		(*nodes)[N] = calloc(strlen(name)+1,sizeof(char));
		strcpy((*nodes)[N],name);
		N++;

		memmove(name+2,name,len);	/* change name to the closer */
		name[0] = '<';
		name[1] = '/';
		name[len+2] = '>';
		name[len+3] = '\0';
		c = strstr(i1,name);		/* position of closer */

		if (!c)	i0 = strchr(i1,'>');/* no '</name>', just a simple node */
		else 	i0 = c + strlen(name);	/* first character after '</name>' */
	}

	nAlloc = N+1;					/* allocate one extra string */
	*nodes = realloc(*nodes,nAlloc*sizeof(char*));
	(*nodes)[N] = NULL;				/* set last one to NULL, signifying end */
	return N;
}



/* returns contents of xml tag, this allocates the space, and does NOT free it */
char*XMLtagContents(
char*	tag,						/* name of tag */
char*	buf,
int		occurance)					/* use 0 for first occurance, 1 for second, ... */
{
	char*	i0;
	char*	i1;
	char*	closer=NULL;
	size_t	len;
	char*	str;

	i0 = startOfxmltag(tag,buf,occurance);
	if (!i0) return NULL;


	i0 = strchr(i0,'>');			/* character after '>' in intro */
	if (!i0) return NULL;
	i0++;							/* start of contents */

	closer = calloc(strlen(tag)+4,sizeof(char));
	closer[0] = '<';
	closer[1] = '/';
	strcpy(closer+2,tag);
	strcat(closer,">");
	i1 = strstr(i0,closer);			/* position of closer */
	CHECK_FREE(closer);
	if (!i1) return NULL;			/* could not find closer */

	len = i1 - i0;					/* length of body */
	str = calloc(len+1,sizeof(char));/* allocate space */
	strncpy(str,i0,len);			/*   and copy */
	str[len] = '\0';
	return str;
}


/* reads a tag contents and converts it to a double vector */
long XMLtagContents2ListDouble(
char*	tag,
char*	buf,						/* buffer containing xml */
int		occurance,					/* use 0 for first occurance, 1 for second, ... */
char*	delimiters,					/* possible delimiters, default is space, tab, cr, or nl = " \t\r\n" */
double** vec)
{
	char*	delimDefault=" \t\r\n";	/* default white-space dlimiters */
	char*	delimLocal;
	char*	str;
	char*	c;
	size_t	nAlloc;					/* number allocated */
	size_t	N;						/* number processed */
	size_t	wsize=sizeof(double);
	char* endptr;

	if (*vec) {
		fprintf(stderr,"ERROR -- in XMLtagContents2ListDouble, *vec not NULL on entry\n");
		return 0;
	}
	if (!delimiters || strlen(delimiters)<1)	delimLocal = delimDefault;
	else										delimLocal = delimiters;

	str = XMLtagContents(tag,buf,occurance);

	nAlloc = 100;
	*vec = calloc(nAlloc,wsize);
	N = 0;
	c = strtok(str,delimLocal);		/* set up strtok */
	c = strtok(NULL,delimLocal);	/* get the first one */
	while(c) {
		if (N>=nAlloc) {			/* need to allocate more space */
			nAlloc += 100;
			*vec = realloc(*vec,nAlloc*wsize);
		}
		(*vec)[N] = strtod(c,&endptr);
		if (strlen(endptr)) (*vec)[N] = NAN;
		c = strtok(NULL,delimLocal);/* next one */
		N++;
	}
	*vec = realloc(*vec,N*wsize);	/* trim to final size */
	return N;
}

/* reads a tag contents and converts it to a float vector */
long XMLtagContents2ListFloat(
char*	tag,
char*	buf,						/* buffer containing xml */
int		occurance,					/* use 0 for first occurance, 1 for second, ... */
char*	delimiters,					/* possible delimiters, default is space, tab, cr, or nl = " \t\r\n" */
float**	vec)
{
	char*	delimDefault=" \t\r\n";	/* default white-space dlimiters */
	char*	delimLocal;
	char*	str;
	char*	c;
	size_t	nAlloc;					/* number allocated */
	size_t	N;						/* number processed */
	size_t	wsize=sizeof(float);
	char* endptr;

	if (*vec) {
		fprintf(stderr,"ERROR -- in XMLtagContents2ListFloat, *vec not NULL on entry\n");
		return 0;
	}
	if (!delimiters || strlen(delimiters)<1)	delimLocal = delimDefault;
	else										delimLocal = delimiters;

	str = XMLtagContents(tag,buf,occurance);

	nAlloc = 100;
	*vec = calloc(nAlloc,wsize);
	N = 0;
	c = strtok(str,delimLocal);		/* set up strtok */
	c = strtok(NULL,delimLocal);	/* get the first one */
	while(c) {
		if (N>=nAlloc) {			/* need to allocate more space */
			nAlloc += 100;
			*vec = realloc(*vec,nAlloc*wsize);
		}
		(*vec)[N] = strtof(c,&endptr);
		if (strlen(endptr)) (*vec)[N] = NAN;
		c = strtok(NULL,delimLocal);/* next one */
		N++;
	}
	*vec = realloc(*vec,N*wsize);	/* trim to final size */
	return N;
}

/* reads a tag contents and converts it to a int vector */
long XMLtagContents2ListInt(
char*	tag,
char*	buf,						/* buffer containing xml */
int		occurance,					/* use 0 for first occurance, 1 for second, ... */
char*	delimiters,					/* possible delimiters, default is space, tab, cr, or nl = " \t\r\n" */
int**	vec)
{
	char*	delimDefault=" \t\r\n";	/* default white-space dlimiters */
	char*	delimLocal;
	char*	str;
	char*	c;
	size_t	nAlloc;					/* number allocated */
	size_t	N;						/* number processed */
	size_t	wsize=sizeof(int);
	char* endptr;

	if (*vec) {
		fprintf(stderr,"ERROR -- in XMLtagContents2ListInt, *vec not NULL on entry\n");
		return 0;
	}
	if (!delimiters || strlen(delimiters)<1)	delimLocal = delimDefault;
	else										delimLocal = delimiters;

	str = XMLtagContents(tag,buf,occurance);

	nAlloc = 100;
	*vec = calloc(nAlloc,wsize);
	N = 0;
	c = strtok(str,delimLocal);		/* set up strtok */
	c = strtok(NULL,delimLocal);	/* get the first one */
	while(c) {
		if (N>=nAlloc) {			/* need to allocate more space */
			nAlloc += 100;
			*vec = realloc(*vec,nAlloc*wsize);
		}
		(*vec)[N] = (int)strtol(c,&endptr,0);
		if (strlen(endptr)) {		/* cannot deal with numerical errors, no way to flag them */
			CHECK_FREE(*vec);
			return 0;
		};
		c = strtok(NULL,delimLocal);/* next one */
		N++;
	}
	*vec = realloc(*vec,N*wsize);	/* trim to final size */
	return N;
}



/* return a list with all of the attribute value pairs for tag */
/* the list looks like: "key1=value1;key2=value2;key3=value3;..." */
/* space for the list is allocated here, you must free it when done */
long XMLattibutes2KeyList(
char	*tag,						/* name of tag to find */
char	*buf,						/* buffer containing xml */
int		occurance,					/* use 0 for first occurance, 1 for second, ... */
char	**keyVals)					/* the result, space is allocated here, you must free it */
{
	char*	i0;
	char*	i1;
	char*	work=NULL;
	char*	value=NULL;
	char*	key=NULL;
	char*	c;
	size_t	len;
	size_t	i;
	long	N=0;					/* number of keys collected */
	
	i0 = startOfxmltag(tag,buf,occurance);
	if (!i0) return 0;
	i0 += strlen(tag) + 1;			/* space before start of attributes, or closing '>' */
	if (*i0 == '>') return 0;		/* no attributes, just return */
	i0++;							/* advance to start of attributes */
	i1 = strchr(i0,'>');			/* end of attributes */
	if (!i1) return 0;				/* could not find closing '>' */
	i1--;

	len = i1 - i0 + 1;
	if (len<1) return 0;

	work = calloc(len+1,sizeof(char));
	strncpy(work,i0,len);
	work[len] = '\0';
	while((c=strchr(work,'\t'))) *c = ' ';	/* replace tab, CR, NL with a single space */
	while((c=strchr(work,'\r'))) *c = ' ';
	while((c=strchr(work,'\n'))) *c = ' ';
	while((c=strchr(work,';'))) *c = '_';	/* no semi-colons, that is the seperator */
	TrimFrontBackWhiteSpace(work);

	/* parse work into key=value pairs */
	*keyVals = calloc(len+1,sizeof(char));
	value = calloc(len,sizeof(char));
	key = calloc(len,sizeof(char));
	(*keyVals)[0] = '\0';			/* should not be necessary */

	i0 = work - 1;
	while(i0) {
		i0++;
		i1 = strchr(i0,'=');
		strncpy(key,i0,i=(i1-i0));
		key[i] = '\0';
		key = TrimFrontBackWhiteSpace(key);

		i0 = strchr(i1,'\"');		/* first double quote around value */
		if (!i0) break;
		i0++;						/* start of value */

		i1 = strchr(i0,'\"');		/* second double quote around value */
		if (i1<i0) break;
		strncpy(value,i0,i=(i1-i0));
		value[i] = '\0';
		if (strlen(key)>0) {
			strcat(*keyVals,key);
			if (strlen(value)) {
				strcat(*keyVals,"=");
				strcat(*keyVals,value);
			}
			strcat(*keyVals,";");
			N++;
		}
		i0 = strchr(i1+1,' ');		/* next space separator, set up for next key="val" pair */
	}
	while(i0>0 && strlen(key) && strlen(value))
	*keyVals = realloc(*keyVals,strlen(*keyVals)*sizeof(char));
	CHECK_FREE(work);
	CHECK_FREE(value);
	CHECK_FREE(key);
	return N;
}



/* remove all xml comments from str, this changes the contents of str, and allocates no space */
char *XMLremoveComments(
char*	str)
{
	static char* start="<!--";
	static char* tail="-->";
	char*	i0;
	char*	i1;
	size_t	mlen;					/* length to move */
	size_t	tailLen;

	tailLen = strlen(tail);
	i0 = strstr(str,start);			/* start of a comment */
	i1 = strstr(str,tail);			/* end of a comment */
	while(i0 && i1) {
		i1 += tailLen;				/* points to first character after the tail */
		mlen = strlen(i1)+1;		/* length of stuff to move, the +1 includes trailing 0 */
		memmove(i0,i1,mlen);		/* remove comment */
		i0 = strstr(str,start);		/* start of next comment */
		i1 = strstr(str,tail);		/* end of next comment */
	}
	return str;
}



int XMLtagExists(					/* returns true if start of tag was found in buf */
char*	tag,						/* name of tag */
char*	buf)
{
	size_t	tlen;
	char	*t;
	
	if (!buf || !tag) return 0;		/* nothing to do */
	tlen = strlen(tag);				/* length of tag */
	t = calloc(tlen+3,sizeof(char));/* room for local copy of tag + (leading <) + (> or ' ') + (terminating null) */
	if (!t) return 0;				/* unable to allocate memory, probably tag is too long */

	strncpy(&(t[1]),tag,tlen);		/* local copy of tag, copied into t[1] */
	t[0] = '<';
	t[tlen+2] = '\0';

	t[tlen+1] = ' ';
	if (strstr(buf,t)) return 1;	/* find start of tag of form '<tag more stuff...>' */

	t[tlen+1] = '>';
	if (strstr(buf,t)) return 1;	/* find start of tag of form '<tag>' */

	return 0;						/* failed to find the tag */
}



/* returns a pointer into buf pointing to the start of '<tag', this allocates no space for user to free */
char *startOfxmltag(
char*	tag,						/* name of tag */
char*	buf,
int		occurance)					/* use 0 for first occurance, 1 for second, ... */
{
	char*	i0;
	char*	i1;
	int		i;
	size_t	len;
	size_t	iend;
	char*	xml=NULL;				/* xml tag with leading '<' and terminator ' ' or '>' */

	len = strlen(tag);
	xml = calloc(len+3,sizeof(char));
	strcpy(xml+1,tag);
	xml[0] = '<';
	iend = len+1;					/* index to either ' ' or '>' */
	xml[iend+1] = '\0';
	i0 = buf;

	for (i=0;i<=occurance;i+=1) {
		xml[iend] = '>';
		i1 = strstr(i0,xml);		/* find start of an xml without attributes */
		xml[iend] = ' ';
		i0 = strstr(i0,xml);		/* find start of an xml with attributes */

		if (!i0 && !i1) break;		/* unable to find start of '<tag ' or '<tag>' */
		i0 = !i0 ? i1 : i0;			/* if i0==NULL, use i1 */
		i1 = !i1 ? i0 : i1;			/* if i1==NULL, use i0 */
		i0 = i0<i1 ? i0 : i1;		/* use min of i0 & i1 */
		i0 += (i<occurance) ? len : 0;	/* for more, move starting point forward */
	}
	CHECK_FREE(xml);
	return i0;
}



/* trim both leading & trailing space, white space is space or less */
char *TrimFrontBackWhiteSpace(
char *str)
{
	TrimLeadingWhiteSpace(str);
	TrimTrailingWhiteSpace(str);
	return str;
}


/* trim leading white space, white space is space or less */
char *TrimLeadingWhiteSpace(
char *str)
{
	size_t	len;
	char*	c;

	for (c=str; (*c && *c<=' '); c++) ;
	if (c!=str) {					/* something to do */
		len = strlen(c);
		memmove(str,c,len);			/* slide down and overwrite leading spaces */
	}
	return str;
}


/* trim trailing space, white space is space or less */
char* TrimTrailingWhiteSpace(
char *str)
{
	size_t	len;
	char*	c;
	len = strlen(str);
	for (c=(str+len-1); (*c<=' ' && c>=str); c--) *c = '\0';	
	return str;
}
