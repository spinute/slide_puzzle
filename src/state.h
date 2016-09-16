#pragma once

#include <stdbool.h>

#define N_DIR 4
typedef enum direction_tag {
    Left  = 0,
    Down  = 1,
    Right = 2,
    Up    = 3,
} Direction;

/*
 *  position by idx_t
 *  [0,0] [1,0] [2,0]
 *  [0,1] [1,1] [2,1]
 *  [0,2] [1,2] [2,2]
 */

#define WIDTH 3

typedef unsigned char idx_t;
typedef unsigned char value;

#define VALUE_EMPTY 0

typedef struct state_tag
{
    int   depth;
    value pos[WIDTH][WIDTH];
    idx_t i, j; /* pos of empty */
} * State;

void state_init(State state, value v_list[WIDTH * WIDTH], int depth);
void state_copy(State src, State dst);
bool state_movable(State state, Direction dir);
void state_move(State state, Direction dir);
bool state_pos_equal(State s1, State s2);
void state_dump(State state);
