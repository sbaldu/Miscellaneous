
#ifndef Color_cuh
#define Color_cuh

#include <cstdint>
#include "Numeric.cuh"

using PixelValue = uint8_t;

namespace cuda {

  struct Color {
    PixelValue red;
    PixelValue green;
    PixelValue blue;
  };

  __device__ PixelValue max(const Color& c) { return max(c.red, c.green, c.blue); }
  __device__ PixelValue min(const Color& c) { return min(c.red, c.green, c.blue); }

};  // namespace cuda

#endif
