#pragma once

#include "./state.h"

#include <stddef.h>

typedef struct stack_tag *Stack;

Stack stack_init(size_t init_capa_hint);
void stack_fini(Stack stack);
void stack_put(Stack stack, State state);
State stack_pop(Stack stack); /* return NULL if empty*/
