/**********************************************************

	Data Structure and Routines for a doubly linked list.

/**********************************************************/
#include <stdio.h>
#include "list.h"

List* list_new(void){

	List* l = malloc(sizeof(List));
	if (!l) exit(ENOMEM);		/* Not enough space. */
	l->size = 0;
	l->head = EMPTY_NODE;
	l->tail = EMPTY_NODE;

	return l;
}

/*list_append takes a pointer to the data */
/*we include a pointer to the list so that we can update length*/
void list_append(List* l, void* value){

	/*create our new node*/
	ListNode* n = malloc(sizeof(ListNode));
	if (!n) exit(ENOMEM);		/* Not enough space. */
	n->value = value;			/* assign the data to it */
	
	n->next = EMPTY_NODE;		/* this element will be at end, so there is no next element */
	n->prev = l->tail;			/* the previous is the existing tail */
	
	if(l->head == EMPTY_NODE) {	/* this list has no nodes in it yet... */
		l->head = l->tail = n;	/* set both the list's head and tail to point to this node */
	} else {
		/*set the tail's 'next' property to point to this node, now, then set list's 'tail' property to point to n, too */
		l->tail->next = n;		/* change original tail to point to new element */
		l->tail = n;			/* set tail to be new element */
	}
	
	l->size++;					/* increment the size of the list */
}

/*list_remove removes the given node from the list, and links the ones before and after it*/
/*we include a pointer to the list so that we can update length, and can handle cases where we're removing the head node*/
void list_remove(List* l, ListNode* n){
	if (n==NULL) {
		fprintf(stderr,"ERROR -- tried to remove null node in list_remove()");
		exit(1);
	};

	/* if this node is both the front and back of the list, then we're going to end up with the empty list */
	if (n == l->head && n == l->tail){
		l->head = EMPTY_NODE;
		l->tail = EMPTY_NODE;
	} 
	/* if this is the head of the list, but not the tail, adjust the head */
	else if (n == l->head){
		l->head = n->next;
		l->head->prev = EMPTY_NODE;				/* the new head has no previous */
	}
	/* if this is the tail of the list, but not the head, adjust the tail */
	else if (n == l->tail){
		l->tail = n->prev;
		l->tail->next = EMPTY_NODE;				/* the new tail has no next */
	}
	/* this is neither the head, nor the tail, so stitch up with its two neighbours */
	else {
		n->prev->next = n->next;
		n->next->prev = n->prev;
	}
	
	/* reclaim some memory, n is no longer needed */
	free(n);
	n = NULL;
	l->size -= (l->size>0) ? 1 : 0;				/* reset length, but never less than 0 */
}

void list_concat(List* dest, List* source){

	if (source->head == EMPTY_NODE) return;		/* the source is empty */

	if (dest->tail == EMPTY_NODE){
		/* the destination is empty, simply take source's list */
		dest->head = source->head;
		dest->tail = source->tail;
	} else {
		/* the destination isn't empty, merge the two lists together */
		dest->tail->next = source->head;	
		source->head->prev = dest->tail;
		dest->tail = source->tail;
	}
	
	dest->size += source->size;
}

void list_delete(List* l) {
	if (l) free(l);
	l = NULL;
}

void list_delete_nodes(List* l) {
	if (!l) return;						/* if NULL, do nothing */
	ListNode* node;
	ListNode* next;

	node = l->head;
	while(node != EMPTY_NODE) {			/* keep going as long as we have valid nodes */
		next = node->next;				/* get the next node before we free this one*/
		free(node);						/*free this node*/
		node = next;
	}

	list_delete(l);
}
//
//void list_delete_nodes(List* l){
//	if (!l) return;						/* if NULL, do nothing */
//	ListNode* node;
//	ListNode* next;
//
//	node = l->head;
//
//	/*keep going until we reach the end of the list*/
//	while(true) {
//		/*get the next node before we free this one*/
//		next = node->next;
//		
//		/*free the node*/
//		if (node) free(node);
//		node = NULL;
//		
//		/*break if the next node is the empty list, otherwise assign next to node*/
//		if (next == EMPTY_NODE) break;
//		node = next;
//	}
//
//	list_delete(l);
//}

