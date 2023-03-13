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


List*	list_new(void);
void	list_append(List* l, void* value);
void	list_remove(List* l, ListNode* n);
void	list_concat(List* l, List* source);
void	list_delete(List* l);
void	list_delete_nodes(List* l);

#endif
