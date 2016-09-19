#include "../pq.h"
#include "../state.h"
#include "./test.h"

TEST_GROUP(pq);

static PQ    q;
static State s, g;

TEST_SETUP(pq)
{
    state_panel s_list[STATE_WIDTH * STATE_WIDTH]
		= {1, 2, 3, 4, STATE_EMPTY, 8, 7, 6, 5};
    state_panel g_list[STATE_WIDTH * STATE_WIDTH]
		= {1, 2, 3, 4, STATE_EMPTY, 8, 7, 6, 5};

    q = pq_init(3);
    s = state_init(s_list, 0);
    g = state_init(g_list, 0);
}

TEST_TEAR_DOWN(pq)
{
    state_fini(s);
    state_fini(g);
    pq_fini(q);
}

TEST(pq, initialization)
{
}

TEST(pq, put)
{
	pq_put(q, s, heuristic_manhattan_distance(s, g));
}
TEST(pq, empty_pop_should_be_null)
{
	State poped;
	poped = pq_pop(q);
	TEST_ASSERT_NULL(poped);
	poped = pq_pop(q);
	TEST_ASSERT_NULL(poped);
}
TEST(pq, pop_is_put)
{
	State poped;

	pq_put(q, s, heuristic_manhattan_distance(s, g));
	poped = pq_pop(q);
	state_pos_equal(s, poped);

	state_fini(poped);
}
TEST(pq, put_many)
{
	for (int i = 0; i < 1000; i++)
		pq_put(q, s, 15 * 17 * 19 * i % 13);
}
TEST(pq, put_many_and_pop_the_half)
{
	for (int i = 0; i < 100; i++)
		pq_put(q, s, 5 * 7 * 31 * i % 19);

	for (int i = 0; i < 50; i++)
	{
		State ret = pq_pop(q);
		TEST_ASSERT_NOT_NULL(ret);
		state_fini(ret);
	}

}

TEST_GROUP_RUNNER(pq)
{
    RUN_TEST_CASE(pq, initialization);
    RUN_TEST_CASE(pq, put);
    RUN_TEST_CASE(pq, empty_pop_should_be_null);
    RUN_TEST_CASE(pq, pop_is_put);
	RUN_TEST_CASE(pq, put_many);
	RUN_TEST_CASE(pq, put_many_and_pop_the_half);
}
