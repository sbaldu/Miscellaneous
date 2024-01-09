
#include <SFML/Graphics.hpp>
#include <complex>
#include <cstdint>

using Complex = std::complex<double>;

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

int main() {
  const uint16_t width{800};
  const uint16_t height{800};

  const Complex top_left{-2.2, 1.5};
  const Complex lower_right{0.8, -1.5};
  const auto diff{lower_right - top_left};

  const auto d_x = diff.real() / width;
  const auto d_y = diff.imag() / height;

  sf::Image image;
  image.create(width, height);

  for (int row{}; row != height; ++row) {
    for (int column{}; column != width; ++column) {
      auto k = mandelbrot(top_left + Complex{d_x * column, d_y * row});
      image.setPixel(column, row, to_color(k));
    }
  }

  image.saveToFile("mandelbrot_serial.png");
}
