/*
 *  keyLists.h
 *  
 *
 *  Created by Jon Tischler on 9/2/12.
 *  Copyright 2012 ANL. All rights reserved.
 *
 */

#ifndef _KEY_LISTS_
#define _KEY_LISTS_


/* #define MAX_DETECTOR_ID_LEN 255		// max length of string with detector ID */
#define MAX_micro_STRING_LEN 1023		/* max length of a string easily read from a data */

#ifndef MAX
#define MAX(X,Y) ( ((X)<(Y)) ? (Y) : (X) )
#endif
#ifndef MIN
#define MIN(X,Y) ( ((X)>(Y)) ? (Y) : (X) )
#endif




char *getFileTypeString(int itype, char *stype);


double NumberByKey(char *key, char *list, char keySepStr, char listSepStr);
long IntByKey(char *key, char *list, char keySepChar, char listSepChar);
char* StringByKey(char *key, char *list, char keySepChar, char listSepChar, int maxLen);


#endif	/* _KEY_LISTS_ */
