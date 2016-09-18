#pragma once

#include "./state.h"
#include <stdint.h>

typedef struct pq_tag *PQ;

PQ pq_init(size_t init_capa_hint);
void pq_fini(PQ pq);

void pq_put(PQ pq, State state);
State pq_pop(PQ pq);
