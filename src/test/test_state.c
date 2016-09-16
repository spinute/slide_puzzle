#include "../state.h"
#include "./test.h"

TEST_GROUP(state);

static struct state_tag s;

TEST_SETUP(state)
{
	/*
	 * 1 2 3
	 * 4 _ 8
	 * 7 6 5
	 */
	value v_list[WIDTH*WIDTH] = {1,2,3,4,VALUE_EMPTY,8,7,6,5};

	state_init(&s, v_list, 0);
}

TEST_TEAR_DOWN(state)
{
}

TEST(state, initalization)
{
	TEST_ASSERT(s.depth == 0);
	TEST_ASSERT(s.pos[0][0] == 1);
	TEST_ASSERT(s.pos[1][0] == 2);
	TEST_ASSERT(s.pos[2][0] == 3);
	TEST_ASSERT(s.pos[0][1] == 4);
	TEST_ASSERT(s.pos[1][1] == VALUE_EMPTY);
	TEST_ASSERT(s.pos[2][1] == 8);
	TEST_ASSERT(s.pos[0][2] == 7);
	TEST_ASSERT(s.pos[1][2] == 6);
	TEST_ASSERT(s.pos[2][2] == 5);
	TEST_ASSERT(s.i == 1);
	TEST_ASSERT(s.j == 1);
}

TEST(state, copied_state_should_be_the_same)
{
	struct state_tag t;

	state_copy(&s, &t);
	TEST_ASSERT(s.depth == t.depth);
	for (idx_t i = 0; i < WIDTH; ++i)
		for (idx_t j = 0; j < WIDTH; ++j)
			TEST_ASSERT(s.pos[i][j] == t.pos[i][j]);
	TEST_ASSERT(s.i == t.i);
	TEST_ASSERT(s.j == t.j);
}
TEST(state, move_left)
{
	state_move(&s, Left);
	TEST_ASSERT(s.depth == 1);
	TEST_ASSERT(s.pos[0][0] == 1);
	TEST_ASSERT(s.pos[1][0] == 2);
	TEST_ASSERT(s.pos[2][0] == 3);
	TEST_ASSERT(s.pos[0][1] == VALUE_EMPTY);
	TEST_ASSERT(s.pos[1][1] == 4);
	TEST_ASSERT(s.pos[2][1] == 8);
	TEST_ASSERT(s.pos[0][2] == 7);
	TEST_ASSERT(s.pos[1][2] == 6);
	TEST_ASSERT(s.pos[2][2] == 5);
	TEST_ASSERT(s.i == 0);
	TEST_ASSERT(s.j == 1);
}
TEST(state, move_right)
{
	state_move(&s, Right);
	TEST_ASSERT(s.depth == 1);
	TEST_ASSERT(s.pos[0][0] == 1);
	TEST_ASSERT(s.pos[1][0] == 2);
	TEST_ASSERT(s.pos[2][0] == 3);
	TEST_ASSERT(s.pos[0][1] == 4);
	TEST_ASSERT(s.pos[1][1] == 8);
	TEST_ASSERT(s.pos[2][1] == VALUE_EMPTY);
	TEST_ASSERT(s.pos[0][2] == 7);
	TEST_ASSERT(s.pos[1][2] == 6);
	TEST_ASSERT(s.pos[2][2] == 5);
	TEST_ASSERT(s.i == 2);
	TEST_ASSERT(s.j == 1);
}
TEST(state, move_down)
{
	state_move(&s, Down);
	TEST_ASSERT(s.depth == 1);
	TEST_ASSERT(s.pos[0][0] == 1);
	TEST_ASSERT(s.pos[1][0] == 2);
	TEST_ASSERT(s.pos[2][0] == 3);
	TEST_ASSERT(s.pos[0][1] == 4);
	TEST_ASSERT(s.pos[1][1] == 6);
	TEST_ASSERT(s.pos[2][1] == 8);
	TEST_ASSERT(s.pos[0][2] == 7);
	TEST_ASSERT(s.pos[1][2] == VALUE_EMPTY);
	TEST_ASSERT(s.pos[2][2] == 5);
	TEST_ASSERT(s.i == 1);
	TEST_ASSERT(s.j == 2);
}
#include <stdio.h>
TEST(state, move_up)
{
	state_move(&s, Up);
	TEST_ASSERT(s.depth == 1);
	TEST_ASSERT(s.pos[0][0] == 1);
	TEST_ASSERT(s.pos[1][0] == VALUE_EMPTY);
	TEST_ASSERT(s.pos[2][0] == 3);
	TEST_ASSERT(s.pos[0][1] == 4);
	TEST_ASSERT(s.pos[1][1] == 2);
	TEST_ASSERT(s.pos[2][1] == 8);
	TEST_ASSERT(s.pos[0][2] == 7);
	TEST_ASSERT(s.pos[1][2] == 6);
	TEST_ASSERT(s.pos[2][2] == 5);
	TEST_ASSERT(s.i == 1);
	TEST_ASSERT(s.j == 0);
}

TEST_GROUP_RUNNER(state)
{
	RUN_TEST_CASE(state, initalization);
    RUN_TEST_CASE(state, copied_state_should_be_the_same);
	RUN_TEST_CASE(state, move_right);
	RUN_TEST_CASE(state, move_down);
	RUN_TEST_CASE(state, move_left);
	RUN_TEST_CASE(state, move_up);
}
