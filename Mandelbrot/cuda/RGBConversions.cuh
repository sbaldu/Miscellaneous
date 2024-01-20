
#ifndef RGBConversions_cuh
#define RGBConversions_cuh

#include <cstdint>
#include "Color.cuh"
#include "Numeric.cuh"

using PixelValue = uint8_t;

namespace cuda {

  struct Lightness {
    __device__ PixelValue operator()(const Color& c) const {
      return (PixelValue)((min(c) + max(c)) / 2.0f);
    }
  };

  struct Average {
    __device__ PixelValue operator()(const Color& c) const {
      return (PixelValue)((c.red + c.green + c.blue) / 3.0f);
    }
  };

  struct Luminosity {
    __device__ PixelValue operator()(const Color& c) const {
      return (PixelValue)(0.3f * c.red + 0.59f * c.green + 0.11f * c.blue);
    }
  };

};  // namespace cuda

#endif
