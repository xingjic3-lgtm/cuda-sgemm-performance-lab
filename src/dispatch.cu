#include "dispatch.cuh"
#include <array>

namespace {

// Phase-three measurements only cover these six square shapes. Entries outside
// this table use a correctness-safe general configuration instead of assuming
// that the numerically nearest measured size has the same fastest kernel.
constexpr std::array<K10DispatchEntry, 6> K10_DISPATCH_TABLE{{
    {128, 128, 128, K10ConfigId::Config29},
    {256, 256, 256, K10ConfigId::Config11},
    {512, 512, 512, K10ConfigId::Config29},
    {1024, 1024, 1024, K10ConfigId::Config38},
    {2048, 2048, 2048, K10ConfigId::Config5},
    {4096, 4096, 4096, K10ConfigId::Config21},
}};

} // namespace

K10ConfigId selectK10Config(int M, int N, int K) {
  for (const K10DispatchEntry &entry : K10_DISPATCH_TABLE) {
    if (M == entry.m && N == entry.n && K == entry.k) {
      return entry.config;
    }
  }
  return K10ConfigId::GeneralFallback;
}

const char *k10ConfigName(K10ConfigId config) {
  switch (config) {
  case K10ConfigId::Config29:
    return "config29";
  case K10ConfigId::Config11:
    return "config11";
  case K10ConfigId::Config38:
    return "config38";
  case K10ConfigId::Config5:
    return "config5";
  case K10ConfigId::Config21:
    return "config21";
  case K10ConfigId::GeneralFallback:
    return "general-fallback";
  }
  return "unknown";
}
