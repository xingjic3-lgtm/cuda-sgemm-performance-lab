#pragma once

enum class K10ConfigId {
  Config29,
  Config11,
  Config38,
  Config5,
  Config21,
  GeneralFallback,
};

struct K10DispatchEntry {
  int m;
  int n;
  int k;
  K10ConfigId config;
};

K10ConfigId selectK10Config(int M, int N, int K);
const char *k10ConfigName(K10ConfigId config);
