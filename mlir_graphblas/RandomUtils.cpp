#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>

using namespace std;

extern "C" {

// This is a simple, fast, and "wrong" implementation that "randomly" chooses
// the first n indices every time
void choose_first(int64_t rngContext, int64_t n, int64_t maxIndex,
                  int64_t *outAlloc, int64_t *outBase, int64_t outOffset,
                  int64_t outSize, int64_t outStride, double *valAlloc,
                  double *valBase, int64_t valOffset, int64_t valSize,
                  int64_t valStride) {
  assert(rngContext == 0x0B00); // only checked when built in debug mode
  cerr << "calling choose_first()" << endl;
  cerr << "NOTE: choose_first sampler is only for testing!" << endl;
  for (int i = 0; i < n; i++) {
    outBase[outOffset + outStride * i] = i;
  }
}

// A uniform sampler using a temporary set
void *create_choose_uniform_context(uint64_t seed) {
  auto generator = new std::default_random_engine(seed);
  return (void *)generator;
}

void choose_uniform(void *rngContext, int64_t n, int64_t maxIndex,
                    int64_t *outAlloc, int64_t *outBase, int64_t outOffset,
                    int64_t outSize, int64_t outStride, double *valAlloc,
                    double *valBase, int64_t valOffset, int64_t valSize,
                    int64_t valStride) {

  std::set<int64_t> selected;
  std::uniform_int_distribution<int64_t> choose_int(0, maxIndex - 1);

  auto generator = (std::default_random_engine *)rngContext;

  while (selected.size() < (size_t)n) {
    int64_t choice = choose_int(*generator);
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

void destroy_choose_uniform_context(void *rngContext) {
  auto generator = (std::default_random_engine *)rngContext;
  delete generator;
}

// A weighted sampler using a temporary set
void *create_choose_weighted_context(uint64_t seed) {
  auto generator = new std::default_random_engine(seed);
  return (void *)generator;
}

void choose_weighted(void *rngContext, int64_t n, int64_t maxIndex,
                     int64_t *outAlloc, int64_t *outBase, int64_t outOffset,
                     int64_t outSize, int64_t outStride, double *valAlloc,
                     double *valBase, int64_t valOffset, int64_t valSize,
                     int64_t valStride) {

  std::set<int64_t> selected;

  auto generator = (std::default_random_engine *)rngContext;

  // compute cumulative distribution
  std::vector<double> cumulative(maxIndex);
  double acc = 0.0;
  for (int64_t i = 0; i < maxIndex; i++) {
    acc += valBase[valOffset + i * valStride];
    cumulative[i] = acc;
  }

  std::uniform_real_distribution<double> choose_double(
      0, cumulative[maxIndex - 1]);

  while (selected.size() < (size_t)n) {
    double r = choose_double(*generator);

    // find smallest element in cumulative distribution greater than r
    int64_t choice = std::distance(
        cumulative.begin(),
        std::upper_bound(cumulative.begin(), cumulative.end(), r));

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

void destroy_choose_weighted_context(void *rngContext) {
  auto generator = (std::default_random_engine *)rngContext;
  delete generator;
}

} // extern "C"
