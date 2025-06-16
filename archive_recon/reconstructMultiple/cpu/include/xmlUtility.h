/*
 *  xmlUtility.h
 *  xmlUtility
 *
 *  Created by Jon Tischler on 7/6/11.
 *  Copyright 2011 Oak Ridge National Laboratory. All rights reserved.
 *
 */

#ifndef _xmlUtility
#define _xmlUtility


#ifndef CHECK_FREE
#define CHECK_FREE(A)   { if(A) free(A); (A)=NULL;}
#endif


int isXMLfile(char *buf);
long XMLNodeList(char *buf, char ***nodes);
char *XMLremoveComments(char* str);
long XMLattibutes2KeyList(char *tag, char *buf, int occurance, char **keyVals);
char *XMLtagContents(char *tag, char* buf, int occurance);
long XMLtagContents2ListDouble(char* tag, char* buf, int occurance, char* delimiters, double** vec);
long XMLtagContents2ListFloat(char* tag, char* buf, int occurance, char* delimiters, float** vec);
long XMLtagContents2ListInt(char* tag, char* buf, int occurance, char* delimiters, int** vec);
int XMLtagExists(char* tag, char* buf);
//	char ***XMLNodeList(char *buf);
//	double *XMLtagContents2ListDouble(char* tag, char* buf, int occurance, char* delimiters);
//	float *XMLtagContents2ListFloat(char* tag, char* buf, int occurance, char* delimiters);
//	int *XMLtagContents2ListInt(char* tag, char* buf, int occurance, char* delimiters);

#endif
