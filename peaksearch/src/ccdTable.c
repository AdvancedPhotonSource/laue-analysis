/**********************************************************

	Data Structure and Routines for a CCD table
       

/**********************************************************/
#include "ccdTable.h"


CCDTable * loadCCDTable ( char * filename){

  CCDTable *ct = malloc(sizeof(CCDTable));
  
  FILE *input=fopen(filename,"r");
  if(input==NULL){
    printf("Error: Can not open file %s to read\n", filename);
    exit(1);
  }
  char line[256], delim[]=" ";

  fgets(line,256,input);

  ct->nx=atoi(strtok(line,delim));
  ct->ny=atoi(strtok(NULL,delim));
  ct->cornerx0=atoi(strtok(NULL,delim));
  ct->cornery0=atoi(strtok(NULL,delim));
  ct->cornerx1=atoi(strtok(NULL,delim));
  ct->cornery1=atoi(strtok(NULL,delim));
 
  ct->xymap=malloc(sizeof(float)*(ct->nx)*(ct->ny)*4);
  if(!ct->xymap) exit(ENOMEM);

	int i;
  for(i=0;i<(ct->nx)*(ct->ny)*4;i+=4){
    if(fgets(line,256,input)==NULL){
	printf("Error in reading file %s.\n",filename);
	exit(1);
      }
    ct->xymap[i]=atof(strtok(line,delim));
    ct->xymap[i+1]=atof(strtok(NULL,delim));
    ct->xymap[i+2]=atof(strtok(NULL,delim));
    ct->xymap[i+3]=atof(strtok(NULL,delim));
 
  }
  fclose(input);

  return ct;
 

}
float ccdTable_getValue(CCDTable *ct,int x,int y, int i){
  int location;
  float value;
  location=i+y*4+x*(ct->ny)*4;
  value= ct->xymap[location];
  return value;
}
void ccdTable_setValue(CCDTable *ct,float value, int x,int y,int i){
  ct->xymap[i+y*4+x*4*(ct->ny)]=value;
}

void ccdTable_delete(CCDTable* ct){
  
  free(ct->xymap);
  
  free(ct);

} 

void print_ccdTable(CCDTable* ct){

  printf("%d %d %d %d %d %d\n",ct->nx,ct->ny,ct->cornerx0,ct->cornery0,ct->cornerx1,ct->cornery1);

	int i, j;
  for(i=0;i<ct->nx;i++)
    for(j=0;j<ct->ny;j++)
     printf("%10.5f%10.5f%10.5f%10.5f\n",ccdTable_getValue(ct,i,j,0),
      ccdTable_getValue(ct,i,j,1),ccdTable_getValue(ct,i,j,2),ccdTable_getValue(ct,i,j,3));
  printf("Done print ccdtable.\n");
	   
}



