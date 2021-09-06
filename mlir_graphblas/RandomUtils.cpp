#include <cstdint>
#include <cassert>
#include <iostream>

using namespace std;

extern "C" {

// This is a simple, fast, and "wrong" implementation that "randomly" chooses
// the first n indices every time

void *create_choose_first_context(int64_t seed)
{
  cerr << "calling create_choose_first_context(" << seed << ")" << endl;
  cerr << "NOTE: choose_first sampler is only for testing!" << endl;
  return (void *)0x0B00;
}

void choose_first(void *rngContext, int64_t n, int64_t maxIndex,
                  int64_t *outAlloc, int64_t *outBase, int64_t outOffset, int64_t outSize, int64_t outStride,
                  double *valAlloc, double *valBase, int64_t valOffset, int64_t valSize, int64_t valStride)
{
    assert(rngContext == (void *)0x0B00);
    cerr << "calling choose_first()" << endl;
    cerr << "NOTE: choose_first sampler is only for testing!" << endl;
    for (int i = 0; i < n; i++) {
      outBase[outOffset + outStride * i] = i;
    }
}

void destroy_choose_first_context(void *rngContext)
{
  cerr << "calling destroy_choose_first_context()" << endl;
  cerr << "NOTE: choose_first sampler is only for testing!" << endl;
  // do nothing since the context pointer isn't allocated memory
}

} // extern "C"