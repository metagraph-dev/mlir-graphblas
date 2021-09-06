#include <cstdint>
#include <cassert>
#include <iostream>

using namespace std;

extern "C" {

// This is a simple, fast, and "wrong" implementation that "randomly" chooses
// the first n indices every time
void choose_first(int64_t rngContext, int64_t n, int64_t maxIndex,
                  int64_t *outAlloc, int64_t *outBase, int64_t outOffset, int64_t outSize, int64_t outStride,
                  double *valAlloc, double *valBase, int64_t valOffset, int64_t valSize, int64_t valStride)
{
    assert(rngContext == 0x0B00);  // only checked when built in debug mode
    cerr << "calling choose_first()" << endl;
    cerr << "NOTE: choose_first sampler is only for testing!" << endl;
    for (int i = 0; i < n; i++) {
      outBase[outOffset + outStride * i] = i;
    }
}

} // extern "C"