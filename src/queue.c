#include "./queue.h"
#include "./utils.h"

struct queue_tag
{

};

Queue
queue_init(void)
{
	Queue q = palloc(sizeof(*q));

	return q;
}

void
queue_fini(Queue q)
{
	pfree(q);
}

void
queue_put(Queue q, State state)
{
}

State
queue_pop(Queue q)
{
	State state;

	return state;
}

void
queue_dump(Queue q)
{
}
