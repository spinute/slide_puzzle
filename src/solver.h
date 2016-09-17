#pragma once

#include "state.h"

void solver_dfs(State init_state,
                State goal_state); /* have no guarantee to finish since no
                                      duplicate detection, for now */
void solver_stack_dfs(State init_state, State goal_state);
void solver_bfs(State init_state, State goal_state);
