/**********************************************************

	Data Structure and Routines for a CCD table
       

/**********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#ifndef _CCDTABLE_H_
#define _CCETABLE_H_

typedef struct {

  int nx;
  int ny;
  int cornerx0;
  int cornery0;
  int cornerx1;
  int cornery1;
  float *xymap; //managed as nx*ny*4 array
	
} CCDTable;

//CCDTable* ccdTable_new(int nx,int ny,int cornerx0,int cornery0,int cornerx1,int cornery1);
//CCDTable* ccdTable_new_empty();

CCDTable *  loadCCDTable (char * filename); 
float ccdTable_getValue(CCDTable* ct,int x,int y,int i);
void ccdTable_setValue(CCDTable* ct, float value,int x,int y,int i);

void	ccdTable_delete(CCDTable* ct);

void print_ccdTable(CCDTable* ct);

#endif
