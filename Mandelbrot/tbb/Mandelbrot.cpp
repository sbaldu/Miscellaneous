
#include <SFML/Graphics.hpp>
#include <complex>
#include <oneapi/tbb.h>
#include <string>

using Complex = std::complex<double>;

template <typename T>
T mandelbrot(const Complex& c) {
  T i{};
  auto z{c};
  for (; i != 256 && norm(z) < 4.; ++i) {
    z = z * z + c;
  }

  return i;
}

auto to_color(int k) {
  return k < 256 ? sf::Color{static_cast<sf::Uint8>(10 * k), 0, 0} : sf::Color::Black;
}

int main(int argc, char **argv) {
  const uint16_t width{800}, height{800};

  const Complex top_left{-2.2, 1.5};
  const Complex bottom_right{0.8, -1.5};
  const auto diff{bottom_right - top_left};

  const auto d_x{diff.real() / width};
  const auto d_y{diff.imag() / height};

  sf::Image image;
  image.create(width, height);

  // read grainsize from command line
  const std::size_t grainsize_x{std::stoul(argv[1])};
  const std::size_t grainsize_y{std::stoul(argv[2])};

  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range2d<size_t>(0, height, grainsize_y, 0, width, grainsize_x),
      [&](oneapi::tbb::blocked_range2d<size_t> r) {
        for (auto row_it{r.rows().begin()}; row_it != r.rows().end(); ++row_it) {
          for (auto col_it{r.cols().begin()}; col_it != r.cols().end(); ++col_it) {
            auto k = mandelbrot<int>(top_left + Complex{d_x * col_it, d_y * row_it});
            image.setPixel(col_it, row_it, to_color(k));
          }
        }
      },
      oneapi::tbb::simple_partitioner());

  const std::string filename{"mandelbrot_" + std::to_string(grainsize_x) + "_" +
                             std::to_string(grainsize_y) + ".png"};
  image.saveToFile(filename);
}
