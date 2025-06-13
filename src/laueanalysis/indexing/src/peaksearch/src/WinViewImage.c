/**********************************************************

	Data Structure and Routines for working with
	WinViewImage data types. These are a combination
	of a Grid data type for the data, and a
	WinViewHeader data type for the header information

/**********************************************************/

#include "WinViewImage.h"

WinViewImage* winview_image_new_empty(void){

	WinViewImage* i = malloc(sizeof(WinViewImage));
	if (!i) exit(ENOMEM);		/* Not enough space. */
	i->data = NULL;				/* these have yet to be allocated */
	i->header = NULL;			/* these have yet to be allocated */
	return i;
}

WinViewImage* winview_image_new(int height, int width){

	WinViewImage* i = winview_image_new_empty();
	i->header = winview_header_new();
	i->data = grid_new(height, width);
	return i;

}

WinViewImage* winview_image_copy(WinViewImage* i) {

	WinViewImage* i2 = winview_image_new_empty();
	
	/* copy the values from the header */
	i2->header = winview_header_copy(i->header);
	
	/* now get a copy of the image data */
	i2->data = grid_new_copy(i->data);
	
	return i2;

}

WinViewImage* winview_image_import(char* filename) {
	/* imports an spe file as a Grid */
	

	WinViewImage* image = winview_image_new_empty();
	image->header = winview_header_new();

	
	FILE* fp = fopen(filename, "r");
	if (!fp) exit(ENOENT);		/* No such file or directory. */
	

	int result;

	/* read the header, and place the values in the struct pointed to by image->header */	
	result = WinViewReadHeader(fp, image->header);
	image->type = SPE_FILE;

	/* now that we have the header, we know the size of the image that we want to import */
	/* so let's allocate the required memory */
	image->data = grid_new(image->header->ydim, image->header->xdim);

	/* read the data contents of the file into a void pointer buffer */
	void* vbuf = NULL;
	result = WinViewReadROI(fp, &vbuf, 0, image->header->xdim-1, 0, image->header->ydim-1, NULL);		/* -1 means the whole dimension */
	
	/* vbuf now points to an array of elements of type head->itype */
	
	/* cover every pixel in the image data */
	int x, y;
	for (x = 0; x < image->header->xdim; x++){
		for (y = 0; y < image->header->ydim; y++){
			grid_set_value( image->data, x, y, spe_value_to_double(vbuf, y*image->header->xdim+x, image->header->itype) );
		}
	}
	
	return image;
	
}

void winview_image_delete(WinViewImage* i){
	if (i->header) free(i->header);
	if (i->data) {
		if (i->data->values) free(i->data->values);
		free(i->data);
	}
	free(i);
}




double spe_value_to_double(void* buffer, int offset, int itype){

	double result;
	
	/* == example of how to get a value out of the buffer using the char datatype ==
	
	cast the buffer from void pointer to pointer into a character pointer to pointer
	(char*)buffer
				
	then take the offset'th element, now that it has a datatype (char) to define size
	((char*)buffer)[offset]
	
	then cast that value to double
	(double)(((char*)buffer)[offset])
	
	*/

	switch (itype) {
		case 4:			/* string/char (1 byte) */
			result = (double)(((char*)buffer)[offset]);
			break;
		case 6:			/* signed int8 (1 byte) */
			result = (double)(((int8_t*)buffer)[offset]);
			break;
		case 7:			/* unsigned int8 (1 byte) */
			result = (double)(((uint8_t*)buffer)[offset]);
			break;
		case 2:			/* short integer (2 byte) */
			result = (double)(((int16_t*)buffer)[offset]);
			break;
		case 3:			/* unsigned integer (2 byte) */			
			result = (double)(((uint16_t*)buffer)[offset]);
			break;
		case 0:			/* float (4 byte) */
			result = (double)(((float*)buffer)[offset]);
			break;
		case 1:			/* long integer (4 byte) */
			result = (double)(((int32_t*)buffer)[offset]);
			break;
		case 5:			/* double (8 byte) */
			result = (double)(((double*)buffer)[offset]);
			break;
		default:
			result = 0.0;
	}
	
	return result;

}
