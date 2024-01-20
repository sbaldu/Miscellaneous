
#include <SFML/Graphics.hpp>
#include <string>

#include "Color.cuh"
#include "Complex.cuh"
#include "Numeric.cuh"
#include "RGBConversions.cuh"

using PixelValue = uint8_t;
using Bytes = std::size_t;
using Color = cuda::Color;
using Complex = cuda::Complex<double>;

__device__ int mandelbrot(const Complex& c) {
  int i{};
  auto z{c};
  for (; i != 256 && cuda::norm(z) < 4.; ++i) {
    z = z * z + c;
  }

  return i;
}

__device__ Color to_color(int k) {
  return k < 256 ? Color{static_cast<PixelValue>(10 * k), 0, 0} : Color{0,0,0};
}

// maybe this could be accelerated further using shared memory, but I think that it would
// be an overkill in this case
template <typename T, typename Method = cuda::Luminosity>
__global__ void build_image(T* image_pixels,
                            std::size_t width,
                            std::size_t height,
                            Complex top_left,
                            double d_x,
                            double d_y) {
  unsigned int thx{threadIdx.x + blockIdx.x * blockDim.x};
  unsigned int thy{threadIdx.y + blockIdx.y * blockDim.y};

  if (thx < width && thy < height) {
    int k{mandelbrot(top_left + Complex{d_x * thx, -d_y * thy})};
    image_pixels[thy * width + thx] = Method{}(to_color(k));
  }
}

int main(int argc, char** argv) {
  const uint16_t width{800}, height{800};

  const Complex top_left{-2.2, 1.5};
  const Complex bottom_right{0.8, -1.5};
  const auto diff{bottom_right - top_left};

  const double d_x{diff.real() / width};
  const double d_y{diff.imag() / height};

  // reminder: sf::Image contains a Vector2u which contains the size of the image and an
  // std::vector<Uint8> containing the information of the pixels
  sf::Image image;
  image.create(width, height);

  const std::size_t image_size{width * height};
  const Bytes image_bytes{image_size * sizeof(PixelValue)};

  // create a buffer on the GPU representing the pixels
  PixelValue* d_pixels;
  cudaMalloc(&d_pixels, image_bytes);
  cudaMemcpy(d_pixels, image.getPixelsPtr(), image_bytes, cudaMemcpyHostToDevice);

  // construct working division
  std::size_t block_size;
  if (argc == 1) {
	block_size = 256;
  } else {
	block_size = std::stoul(argv[1]);
  }
  const std::size_t grid_size{
      static_cast<std::size_t>(std::ceil(image_size / static_cast<float>(block_size)))};

  build_image<<<grid_size, block_size>>>(d_pixels, width, height, top_left, d_x, d_y);

  cudaMemcpy(const_cast<PixelValue*>(image.getPixelsPtr()),
             d_pixels,
             image_bytes,
             cudaMemcpyDeviceToHost);

  const std::string filename{"mandelbrot_" + std::to_string(block_size) + ".png"};
  image.saveToFile(filename);
}
