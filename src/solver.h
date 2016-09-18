#pragma once

#include "state.h"

void solver_dfs(State init_state, State goal_state);
void solver_stack_dfs(State init_state, State goal_state);
void solver_bfs(State init_state, State goal_state);

