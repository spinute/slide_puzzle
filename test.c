#include <stdio.h>
#define STATE_N 16
#define N_CORE 32

typedef unsigned char uchar;

int main(void)
{
	uchar  s_list[STATE_N * N_CORE];
	int i;

	for(i = 0; i < 16; i++)
		s_list[i] = i;

	for(i = 0; i < 16; i++)
		printf("%d ", (int)s_list[i]);
	puts("");

	for (i = 16; i < 32*16; ++i)
		s_list[i] = s_list[i%STATE_N];
	for (i = 0; i < 32*16; ++i)
		printf("%d ", (int)s_list[i]);
	puts("");

	return 0;
}
