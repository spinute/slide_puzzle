#include "../stack.h"
#include "./test.h"

TEST_GROUP(stack);

Stack s;

TEST_SETUP(stack)
{
    s = stack_init(1234);
}

TEST_TEAR_DOWN(stack)
{
    stack_fini(s);
}

TEST(stack, initialization)
{
    TEST_ASSERT_NULL(stack_pop(s));
}

TEST_GROUP_RUNNER(stack)
{
    RUN_TEST_CASE(stack, initialization);
}
