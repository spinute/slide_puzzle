#pragma once

#include <stdbool.h>
#include <stddef.h>

#define N_DIR 4
typedef enum direction_tag {
    LEFT  = 0,
    DOWN  = 1,
    RIGHT = 2,
    UP    = 3,
} Direction;

typedef unsigned char state_panel;

#define STATE_EMPTY 0
#define STATE_WIDTH 4
#define STATE_N STATE_WIDTH *STATE_WIDTH

/*
 * v_list is corresponds to the state of the puzzle such as described below
 *  [ 0] [ 1] [ 2] [ 3]
 *  [ 4] [ 5] [ 6] [ 7]
 *  [ 8] [ 9] [10] [11]
 *  [12] [13] [14] [15]
 */

typedef struct state_tag *State;

State state_init(state_panel v_list[STATE_N], int depth);
void state_fini(State state);
State state_copy(State src);

bool state_is_goal(State s);
int state_get_hvalue(State s);
bool state_movable(State state, Direction dir);
void state_move(State state, Direction dir);
bool state_pos_equal(State s1, State s2);
Direction state_get_parent(State state);
size_t state_hash(State state);
void state_dump(State state);

/*
 * Heuristics below are found at
 * https://heuristicswiki.wikispaces.com/N+-+Puzzle
 */

/* obsolete */
/*
typedef enum {
    HeuristicManhattanDistance,
    HeuristicTilesOutOfRowCol,
    HeuristicMisplacedTiles,

    HeuristicNotSet,
} Heuristic;

int calc_h_value(Heuristic heuristic, State from, State to);
*/
