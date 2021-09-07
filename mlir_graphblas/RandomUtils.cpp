#include <cstdint>
#include <cassert>
#include <iostream>
#include <set>
#include <random>

using namespace std;

std::default_random_engine globalGenerator;

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

// A uniform sampler using a temporary set
void choose_uniform(int64_t rngContext, int64_t n, int64_t maxIndex,
                  int64_t *outAlloc, int64_t *outBase, int64_t outOffset,
                  int64_t outSize, int64_t outStride, double *valAlloc,
                  double *valBase, int64_t valOffset, int64_t valSize,
                  int64_t valStride) {

  std::set<int64_t> selected;
  std::uniform_int_distribution<int64_t> choose_int(0, maxIndex - 1);

  while (selected.size() < (size_t) n) {
    int64_t choice = choose_int(globalGenerator);
    if (selected.count(choice) == 0)
      selected.insert(choice);
  }

  // sets are stored in sorted order
  int i = 0;
  for (int64_t element : selected) {
    outBase[outOffset + outStride * i] = element;
    i++;
  }
}

} // extern "C"