
#include <SFML/Graphics.hpp>
#include <chrono>
#include <complex>
#include <cstdint>
#include <iostream>
#include <oneapi/tbb.h>
#include <string>

template <typename T>
struct ImageSize_t {
  T width;
  T height;
};

using Complex = std::complex<double>;
using ImageSize = ImageSize_t<uint16_t>;

int mandelbrot(const Complex& c) {
  int i{};
  auto z{c};
  for (; i != 256 && norm(z) < 4.; ++i) {
    z = z * z + c;
  }

  return i;
}

auto to_color(int k) {
  return k < 256 ? sf::Color{static_cast<sf::Uint8>(10 * k), 0, 0} : sf::Color::Black;
}

template <typename Partitioner>
void build_image(sf::Image& image,
                 const ImageSize& size,
                 const Complex& top_left,
                 std::size_t grainsize,
                 const Partitioner& partitioner,
                 double d_x,
                 double d_y) {
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<size_t>(
          0, size.height, grainsize, 0, size.width, grainsize),
      [&](oneapi::tbb::blocked_range2d<size_t> r) {
        for (auto row_it{r.rows().begin()}; row_it != r.rows().end(); ++row_it) {
          for (auto col_it{r.cols().begin()}; col_it != r.cols().end(); ++col_it) {
            auto k = mandelbrot(top_left + Complex{d_x * col_it, d_y * row_it});
            image.setPixel(col_it, row_it, to_color(k));
          }
        }
      },
      partitioner);
}

int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << "Error: no grainsize specified" << '\n';
    std::cout << "Usage: " << argv[0] << " grainsize_x grainsize_y" << '\n';
    return 1;
  }
  const uint16_t width{800}, height{800};

  const Complex top_left{-2.2, 1.5};
  const Complex bottom_right{0.8, -1.5};
  const auto diff{bottom_right - top_left};

  const auto d_x{diff.real() / width};
  const auto d_y{diff.imag() / height};

  sf::Image image;
  image.create(width, height);

  // read grainsize from command line
  const std::size_t grainsize{std::stoul(argv[1])};

  auto start{std::chrono::high_resolution_clock::now()};
  build_image(image,
              {width, height},
              top_left,
              grainsize,
              oneapi::tbb::simple_partitioner{},
              d_x,
              d_y);
  auto finish{std::chrono::high_resolution_clock::now()};
  std::cout
      << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
      << '\n';

  const std::string filename{"mandelbrot_" + std::to_string(grainsize) + ".png"};
  image.saveToFile(filename);
}
