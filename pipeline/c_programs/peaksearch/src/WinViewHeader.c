/**********************************************************

	Data Structure and Routines for working with
	WinViewHeader data types. This is a collection of
	values describing an SPE file's properties

/**********************************************************/

#include "WinViewHeader.h"

WinViewHeader* winview_header_new(void){

	WinViewHeader* h = malloc(sizeof(WinViewHeader));
	if (!h) exit(ENOMEM);		/* Not enough space. */
	return h;
	
}

WinViewHeader* winview_header_copy(WinViewHeader* h){

	WinViewHeader* h2 = winview_header_new();
	memcpy(h2, h, sizeof(WinViewHeader));
	return h2;

}

void winview_header_delete(WinViewHeader* h){

	free(h);

}
