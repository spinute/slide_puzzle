#include "./pq.h"
#include "./utils.h"

#include <assert.h>

/*
 * NOTE:
 * This priority queue is implemented doubly reallocated array.
 * It will only extend and will not shrink, for now.
 * It may be improved by using array of layers of iteratively widened array
 */
struct pq_tag
{
	size_t n_elems;
	size_t depth;
	size_t capa;
	size_t max_depth;
	State *array;
};

static inline size_t
calc_init_depth(size_t capa_hint)
{
	size_t depth = 1;
	assert(hint > 0);

	while (1 << depth < hint)
		depth += 1;
	return depth;
}

PQ
pq_init(size_t init_capa_hint)
{
	PQ pq = palloc(sizeof(*pq));

	pq->n_elems = pq->depth = 0;
	pq->max_depth = calc_init_depth(init_capa_hint);
	pq->capa = (1 << pq->max_depth) - 1;

	assert(pq->capa <= SIZE_MAX / sizeof(State));
	pq->array = palloc(sizeof(State) * pq->capa);

	return pq;
}

void
pq_fini(PQ pq)
{
	for()
		/* free all the states inside pq */
	pfree()
	pfree(pq);
}

static inline bool
pq_is_full(PQ pq)
{
	assert(pq->n_elems <= pq->capa);
	return pq->n_elems == pq->capa;
	/* NOTE: Actually, there is one empty slot, but that is regard as full */
}

static inline size_t
pq_up(size_t i)
{
	/* NOTE: By using 1-origin, it may be written more simply, i >> 1 */
	return (i >> 1) - 1;
}

static inline size_t
pq_left(size_t i)
{
	return (i << 1) + 1;
}

static inline size_t
pq_right(size_t i)
{
	return (i << 1) + 2;
}

void
pq_put(PQ pq, State state)
{
}

State
pq_pop(PQ pq)
{
}
