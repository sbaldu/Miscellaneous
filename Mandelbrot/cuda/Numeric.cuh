
#ifndef Numeric_cuh
#define Numeric_cuh

namespace cuda {

  template <typename T1, typename T2>
  __device__ T1 max(T1 arg1, T2 arg2) {
    if (arg1 >= arg2) {
      return arg1;
    } else {
      return arg2;
    }
  }

  template <typename T1, typename T2, typename... Tn>
  __device__ T1 max(T1 arg1, T2 arg2, Tn... args) {
    if (arg1 >= arg2) {
      return max(arg1, args...);
    } else {
      return max(arg2, args...);
    }
  }

  template <typename T1, typename T2>
  __device__ T1 min(T1 arg1, T2 arg2) {
    if (arg1 <= arg2) {
      return arg1;
    } else {
      return arg2;
    }
  }

  template <typename T1, typename T2, typename... Tn>
  __device__ T1 min(T1 arg1, T2 arg2, Tn... args) {
    if (arg1 <= arg2) {
      return min(arg1, args...);
    } else {
      return min(arg2, args...);
    }
  }

};  // namespace cuda

#endif
