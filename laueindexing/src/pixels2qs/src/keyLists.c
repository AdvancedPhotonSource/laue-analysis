/*
 *  keyLists.c
 *
 *  Created by Jon Tischler on 9/2/12.
 *  Copyright 2012 ANL. All rights reserved.
 *
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "keyLists.h"


/* #define CHECK_FREE(A)   { if(A) free(A); (A)=NULL; } */


/*******************************************************************************************
 ***************************  Utility Section, all are External  ***************************
 *******************************************************************************************/

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
		return (long)floor(NumberByKey(key,list,keySepChar,listSepChar)+0.5);
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

/*******************************************************************************************/
/*******************************************************************************************/
