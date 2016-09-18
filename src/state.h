#pragma once

#include <stdbool.h>

#define N_DIR 4
typedef enum direction_tag {
    LEFT  = 0,
    DOWN  = 1,
    RIGHT = 2,
    UP    = 3,
} Direction;

typedef unsigned char state_panel;

#define STATE_EMPTY 0
#define WIDTH 3

/*
 * v_list
 *  [0] [1] [2]
 *  [3] [4] [5]
 *  [6] [7] [8]
 */

typedef struct state_tag *State;

State state_init(state_panel v_list[WIDTH * WIDTH], int depth);
void state_fini(State state);
State state_copy(State src);

int state_get_depth(State state);
bool state_movable(State state, Direction dir);
void state_move(State state, Direction dir);
bool state_pos_equal(State s1, State s2);
void state_dump(State state);

// Heuristics below are found at
// https://heuristicswiki.wikispaces.com/N+-+Puzzle
int heuristic_manhattan_distance(State from, State to);
int heuristic_linear_conflict(State from, State to);
int heuristic_pattern_database(State from, State to);
int heuristic_misplaced_tiles(State from, State to);
int heuristic_nillson(State from, State to);
int heuristic_n_max_swap(State from, State to);
int heuristic_xy(State from, State to);
int heuristic_tiles_out_of_row_col(State from, State to);
