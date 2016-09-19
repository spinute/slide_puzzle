#include "../pq.h"
#include "../state.h"
#include "./test.h"

TEST_GROUP(pq);

static PQ    pq;
static State key;

TEST_SETUP(pq)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH]
		= {1, 2, 3, 4, STATE_EMPTY, 8, 7, 6, 5};

    pq  = pq_init(3);
    key = state_init(v_list, 0);
}

TEST_TEAR_DOWN(pq)
{
    state_fini(key);
    pq_fini(pq);
}

TEST(pq, initialization)
{
}


TEST_GROUP_RUNNER(pq)
{
    RUN_TEST_CASE(pq, initialization);
}
