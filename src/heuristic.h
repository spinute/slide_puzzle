#pragma once

#include "./state.h"

// Heuristics below are found at https://heuristicswiki.wikispaces.com/N+-+Puzzle

int heuristic_manhattan_distance(State from, State to);
int heuristic_linear_conflict(State from, State to);
int heuristic_pattern_database(State from, State to);
int heuristic_misplaced_tiles(State from, State to);
int heuristic_nillson(State from, State to);
int heuristic_n_max_swap(State from, State to);
int heuristic_xy(State from, State to);
int heuristic_tiles_out_of_row_col(State from, State to);
