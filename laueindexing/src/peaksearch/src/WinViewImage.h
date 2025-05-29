/**********************************************************

	Data Structure and Routines for working with
	WinViewImage data types. These are a combination
	of a Grid data type for the data, and a
	WinViewHeader data type for the header information

/**********************************************************/

#include <stdint.h>
#include <errno.h>

#include "grid.h"
#include "WinViewHeader.h"
#include "WinViewRead.h"

#ifndef _WINVIEW_IMAGE_H_
#define _WINVIEW_IMAGE_H_

#define SPE_FILE 0		/* used for the 'type' in WinViewImage */
#define HDF5_FILE 1

typedef struct {

	Grid* data;
	WinViewHeader* header;
	int	type;				/* 0=spe, 1=hdf5 */
	/* Additional entries? */

} WinViewImage;

WinViewImage*	winview_image_new_empty(void);
WinViewImage*	winview_image_new(int height, int width);
WinViewImage*	winview_image_copy(WinViewImage* i);
WinViewImage*	winview_image_import(char* filename);
void			winview_image_delete(WinViewImage* i);

double			spe_value_to_double(void* buffer, int offset, int itype);

#endif
