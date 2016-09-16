#pragma once

#include <stdio.h>
#include <stdarg.h>

#define elog(...) fprintf(stderr, __VA_ARGS__)

void *palloc(size_t size);
void pfree(void *ptr);
