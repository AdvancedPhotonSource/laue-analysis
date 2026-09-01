/* Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
   Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE */
/**********************************************************

	Data Structure and Routines for a doubly linked list.

/**********************************************************/

#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>

#ifndef _LIST_H_
#define _LIST_H_

#define EMPTY_NODE NULL

struct ListNodeStruct {
	void* value;
	struct ListNodeStruct* prev;
	struct ListNodeStruct* next;
};

typedef struct ListNodeStruct ListNode;

typedef struct {
	ListNode* head;
	ListNode* tail;
	int size;
} List;

typedef void (*ListValueDestructor)(void* value);


List*	list_new(void);
int	list_append(List* l, void* value);
int	list_remove(List* l, ListNode* n);
void	list_concat(List* l, List* source);
void	list_delete(List* l);
/* Delete the nodes and container, leaving borrowed values untouched. */
void	list_delete_nodes(List* l);
/* Delete owned values with destroy_value, then delete the nodes and container. */
void	list_delete_with_values(List* l, ListValueDestructor destroy_value);

#endif
