
#ifndef Complex_cuh
#define Complex_cuh

namespace cuda {

  // redefine std::complex for using it in device code
  template <typename T>
  class Complex {
  private:
    T m_real;
    T m_imag;

  public:
    __host__ __device__ Complex() : m_real{}, m_imag{} {}
    __host__ __device__ Complex(T real, T imag) : m_real{real}, m_imag{imag} {}

    __host__ __device__ T real() const { return m_real; }
    __host__ __device__ T imag() const { return m_imag; }

    __device__ friend Complex operator+(const Complex& lhs, const Complex& rhs) {
      return {lhs.m_real + rhs.m_real, lhs.m_imag + rhs.m_imag};
    }
    __host__ __device__ friend Complex operator-(const Complex& lhs, const Complex& rhs) {
      return {lhs.m_real - rhs.m_real, lhs.m_imag - rhs.m_imag};
    }
    __device__ friend Complex operator*(const Complex& lhs, const Complex& rhs) {
      return {lhs.m_real * rhs.m_real - lhs.m_imag * rhs.m_imag,
              lhs.m_real * rhs.m_imag + lhs.m_imag * rhs.m_real};
    }
  };

  template <typename T>
  __device__ T norm(const cuda::Complex<T>& c) {
    return c.real() * c.real() + c.imag() * c.imag();
  }
};  // namespace cuda

#endif
