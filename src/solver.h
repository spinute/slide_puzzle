#pragma once

#include "state.h"

void solver_iddfs(State init_state, State goal_state);
bool solver_dls(State init_state, State goal_state, int depth_limit);

void solver_dfs(State init_state, State goal_state);
void solver_bfs(State init_state, State goal_state);

void solver_recursive_dfs_without_closed(State init_state, State goal_state);
void solver_dfs_without_closed(State init_state, State goal_state);
void solver_bfs_without_closed(State init_state, State goal_state);
