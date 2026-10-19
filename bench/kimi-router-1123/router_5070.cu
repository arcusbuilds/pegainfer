// pegainfer#1123: old (8-round reduction) vs new (rank count) Kimi-K2 router select kernel.
// Bit-identity of topk_idx/topk_weight on identical device inputs + selection-kernel timing.
// Standalone kit for an RTX 5070 Ti (sm_120): both kernel bodies are inlined verbatim below, no repo needed.
// Build: nvcc -O3 -arch=sm_120 router_5070.cu -o router_5070   (CUDA 12.8+)
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- OLD kernel, verbatim from pegainfer e013adc:pegainfer-kernels/csrc/kimi_k2/kimi_router.cu ----
namespace oldk {
constexpr int kRouterSelectThreads = 512;
constexpr int kKimiExperts = 384;
constexpr int kKimiTopk = 8;

__device__ __forceinline__ bool better_router_choice(float value, int expert, float best_value,
                                                     int best_expert) {
  return value > best_value || (value == best_value && expert < best_expert);
}

__global__ void router_scores_topk_normalize_kernel(
    const float *__restrict__ logits,
    const float *__restrict__ e_score_correction_bias,
    float *__restrict__ topk_weight,
    int *__restrict__ topk_idx,
    int active_tokens,
    int padded_tokens,
    int n_experts,
    int topk,
    float route_scale) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= padded_tokens) return;

  extern __shared__ char shared[];
  float *scores = reinterpret_cast<float *>(shared);
  float *choice_scores = scores + blockDim.x;
  float *reduce_values = choice_scores + blockDim.x;
  int *reduce_indices = reinterpret_cast<int *>(reduce_values + blockDim.x);
  float *selected_scores = reinterpret_cast<float *>(reduce_indices + blockDim.x);

  const int expert = tid;
  if (expert < n_experts) {
    float score = 1.0f / (1.0f + expf(-logits[token * n_experts + expert]));
    scores[tid] = score;
    choice_scores[tid] = score + e_score_correction_bias[expert];
  } else {
    scores[tid] = 0.0f;
    choice_scores[tid] = -CUDART_INF_F;
  }
  if (token >= active_tokens) return;
  __syncthreads();

  float selected_sum = 0.0f;
  for (int route = 0; route < topk; ++route) {
    reduce_values[tid] = choice_scores[tid];
    reduce_indices[tid] = expert < n_experts ? expert : n_experts;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other_value = reduce_values[tid + stride];
        const int other_idx = reduce_indices[tid + stride];
        if (better_router_choice(other_value, other_idx, reduce_values[tid], reduce_indices[tid])) {
          reduce_values[tid] = other_value;
          reduce_indices[tid] = other_idx;
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      const int best_idx = reduce_indices[0];
      const float route_score = best_idx < n_experts ? scores[best_idx] : 0.0f;
      selected_scores[route] = route_score;
      topk_idx[token * topk + route] = best_idx;
      topk_weight[token * topk + route] = route_score;
      selected_sum += route_score;
      if (best_idx < n_experts) {
        choice_scores[best_idx] = -CUDART_INF_F;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float scale = selected_sum > 0.0f ? route_scale / selected_sum : 0.0f;
    for (int route = 0; route < topk; ++route) {
      topk_weight[token * topk + route] = selected_scores[route] * scale;
    }
  }
}

}  // namespace oldk
// ---- NEW kernel, verbatim from branch perf/kimi-router-rank-count (61845d3) ----
namespace newk {
constexpr int kRouterSelectThreads = 384;
constexpr int kKimiExperts = 384;
constexpr int kKimiTopk = 8;

__device__ __forceinline__ bool better_router_choice(float value, int expert, float best_value,
                                                     int best_expert) {
  return value > best_value || (value == best_value && expert < best_expert);
}

__global__ void router_scores_topk_normalize_kernel(
    const float *__restrict__ logits,
    const float *__restrict__ e_score_correction_bias,
    float *__restrict__ topk_weight,
    int *__restrict__ topk_idx,
    int active_tokens,
    int padded_tokens,
    int n_experts,
    int topk,
    float route_scale) {
  int token = blockIdx.x;
  int tid = threadIdx.x;
  if (token >= padded_tokens) return;

  extern __shared__ char shared[];
  float *scores = reinterpret_cast<float *>(shared);
  float *choice_scores = scores + blockDim.x;
  float *selected_scores = reinterpret_cast<float *>(choice_scores + blockDim.x);

  const int expert = tid;
  if (expert < n_experts) {
    float score = 1.0f / (1.0f + expf(-logits[token * n_experts + expert]));
    scores[tid] = score;
    choice_scores[tid] = score + e_score_correction_bias[expert];
  } else {
    scores[tid] = 0.0f;
    choice_scores[tid] = -CUDART_INF_F;
  }
  if (token >= active_tokens) return;
  __syncthreads();

  if (expert < n_experts) {
    const float cv = choice_scores[tid];
    int rank = 0;
    for (int j = 0; j < n_experts; ++j) {
      if (better_router_choice(choice_scores[j], j, cv, expert)) ++rank;
    }
    if (rank < topk) {
      topk_idx[token * topk + rank] = expert;
      selected_scores[rank] = scores[expert];
    }
  }
  __syncthreads();

  if (tid == 0) {
    float selected_sum = 0.0f;
    for (int route = 0; route < topk; ++route) selected_sum += selected_scores[route];
    const float scale = selected_sum > 0.0f ? route_scale / selected_sum : 0.0f;
    for (int route = 0; route < topk; ++route) {
      topk_weight[token * topk + route] = selected_scores[route] * scale;
    }
  }
}

}  // namespace newk

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));       \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

constexpr int E = 384, K = 8;
constexpr float kRouteScale = 1.0f;  // KIMI_K2_ROUTER_WEIGHT_SCALE
constexpr int32_t kIdxSentinel = 0x7EADBEEF;
constexpr uint32_t kWeightSentinel = 0x7FC0DEADu;  // NaN payload the kernel never produces

// Same smem formulas as each version's launcher.
const size_t kOldSmem = size_t(oldk::kRouterSelectThreads) * (3 * sizeof(float) + sizeof(int)) +
                        size_t(K) * sizeof(float);
const size_t kNewSmem =
    size_t(newk::kRouterSelectThreads) * (2 * sizeof(float)) + size_t(K) * sizeof(float);

struct Lcg {
  uint64_t s;
  uint32_t next() {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(s >> 32);
  }
  float unit() { return (next() / float(UINT32_MAX)) * 2.0f - 1.0f; }  // [-1, 1]
  int below(int n) { return int(next() % uint32_t(n)); }
};

struct Case {
  std::string name;
  int active, padded;
  std::vector<float> logits, bias;
};

void launch(bool old_kernel, const float *logits, const float *bias, float *w, int *idx,
            int active, int padded, cudaStream_t s) {
  if (old_kernel) {
    oldk::router_scores_topk_normalize_kernel<<<padded, oldk::kRouterSelectThreads, kOldSmem, s>>>(
        logits, bias, w, idx, active, padded, E, K, kRouteScale);
  } else {
    newk::router_scores_topk_normalize_kernel<<<padded, newk::kRouterSelectThreads, kNewSmem, s>>>(
        logits, bias, w, idx, active, padded, E, K, kRouteScale);
  }
}

struct Out {
  std::vector<int32_t> idx;
  std::vector<float> w;
};

Out run(bool old_kernel, const Case &c, const float *d_logits, const float *d_bias) {
  const size_t n = size_t(c.padded) * K;
  std::vector<int32_t> si(n, kIdxSentinel);
  std::vector<uint32_t> sw(n, kWeightSentinel);
  int *d_idx;
  float *d_w;
  CK(cudaMalloc(&d_idx, n * 4));
  CK(cudaMalloc(&d_w, n * 4));
  CK(cudaMemcpy(d_idx, si.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_w, sw.data(), n * 4, cudaMemcpyHostToDevice));
  launch(old_kernel, d_logits, d_bias, d_w, d_idx, c.active, c.padded, 0);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  Out o{std::vector<int32_t>(n), std::vector<float>(n)};
  CK(cudaMemcpy(o.idx.data(), d_idx, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.w.data(), d_w, n * 4, cudaMemcpyDeviceToHost));
  CK(cudaFree(d_idx));
  CK(cudaFree(d_w));
  return o;
}

// Rows < active: 8 distinct in-range experts, no sentinel left. Rows >= active: untouched.
bool sane(const Out &o, const Case &c, const char *who) {
  for (int t = 0; t < c.padded; ++t) {
    for (int r = 0; r < K; ++r) {
      const size_t i = size_t(t) * K + r;
      uint32_t wb;
      memcpy(&wb, &o.w[i], 4);
      const bool sentinel = o.idx[i] == kIdxSentinel && wb == kWeightSentinel;
      if (t >= c.active && !sentinel) {
        printf("  %s %s: filler row %d slot %d was written\n", c.name.c_str(), who, t, r);
        return false;
      }
      if (t < c.active && (o.idx[i] < 0 || o.idx[i] >= E || wb == kWeightSentinel)) {
        printf("  %s %s: row %d slot %d not written / out of range\n", c.name.c_str(), who, t, r);
        return false;
      }
      for (int q = 0; t < c.active && q < r; ++q) {
        if (o.idx[size_t(t) * K + q] == o.idx[i]) {
          printf("  %s %s: row %d duplicate expert %d\n", c.name.c_str(), who, t, o.idx[i]);
          return false;
        }
      }
    }
  }
  return true;
}

long g_cases = 0, g_rows = 0, g_fail = 0;

void check(const Case &c, bool /*unused*/ = false) {
  float *d_logits, *d_bias;
  CK(cudaMalloc(&d_logits, c.logits.size() * 4));
  CK(cudaMalloc(&d_bias, c.bias.size() * 4));
  CK(cudaMemcpy(d_logits, c.logits.data(), c.logits.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bias, c.bias.data(), c.bias.size() * 4, cudaMemcpyHostToDevice));
  Out a = run(true, c, d_logits, d_bias);
  Out b = run(false, c, d_logits, d_bias);
  CK(cudaFree(d_logits));
  CK(cudaFree(d_bias));
  ++g_cases;
  g_rows += c.active;
  bool ok = sane(a, c, "old") && sane(b, c, "new");
  // Whole padded buffers, so filler rows are compared too.
  if (memcmp(a.idx.data(), b.idx.data(), a.idx.size() * 4) != 0 ||
      memcmp(a.w.data(), b.w.data(), a.w.size() * 4) != 0) {
    ok = false;
    for (size_t i = 0; i < a.idx.size(); ++i) {
      uint32_t wa, wb;
      memcpy(&wa, &a.w[i], 4);
      memcpy(&wb, &b.w[i], 4);
      if (a.idx[i] != b.idx[i] || wa != wb) {
        printf("  MISMATCH %s active=%d padded=%d row %zu slot %zu: old (%d, %08x %.9g) new (%d, %08x %.9g)\n",
               c.name.c_str(), c.active, c.padded, i / K, i % K, a.idx[i], wa, a.w[i], b.idx[i], wb,
               b.w[i]);
        break;
      }
    }
  }
  if (!ok) ++g_fail;
}

Case make(const std::string &name, int active, int padded) {
  return Case{name, active, padded, std::vector<float>(size_t(padded) * E), std::vector<float>(E)};
}

// Continuous logits: ties only by chance.
Case random_case(uint64_t seed, int active, int padded, float logit_range, float bias_scale) {
  Lcg r{seed};
  Case c = make("random_s" + std::to_string(seed) + "_a" + std::to_string(active) + "_p" +
                    std::to_string(padded),
                active, padded);
  for (auto &b : c.bias) b = r.unit() * bias_scale;
  for (auto &x : c.logits) x = r.unit() * logit_range;
  return c;
}

// Few distinct (logit, bias) pairs: ~32 experts share each choice value, so the top value
// group always straddles the 8th/9th place and the lower-index rule decides the picks.
Case quantized_tie_case(uint64_t seed, int active, int padded) {
  Lcg r{seed};
  Case c = make("qtie_s" + std::to_string(seed) + "_a" + std::to_string(active) + "_p" +
                    std::to_string(padded),
                active, padded);
  const float lv[4] = {-1.0f, 0.0f, 1.0f, 2.0f}, bv[3] = {0.0f, 0.25f, 0.5f};
  for (auto &b : c.bias) b = bv[r.below(3)];
  for (auto &x : c.logits) x = lv[r.below(4)];
  return c;
}

// Exact choice ties with DIFFERENT un-biased scores: sigmoid(30) == 1.0f with bias 0, and
// sigmoid(0) == 0.5f with bias 0.5, both give choice == 1.0f exactly. n_hi experts per row
// (random 5..20, so the tie group can straddle place 8/9) sit at choice 1.0; the rest are
// strictly below. Which tied experts win changes the weights, so weight bits test the order.
Case score_tie_case(uint64_t seed, int active, int padded) {
  Lcg r{seed};
  Case c = make("stie_s" + std::to_string(seed) + "_a" + std::to_string(active) + "_p" +
                    std::to_string(padded),
                active, padded);
  for (int e = 0; e < E; ++e) c.bias[e] = (r.below(2) == 0) ? 0.0f : 0.5f;
  for (int t = 0; t < padded; ++t) {
    float *row = &c.logits[size_t(t) * E];
    for (int e = 0; e < E; ++e) {
      // choice < 1.0: bias 0 -> sigmoid(<= 6) < 1; bias 0.5 -> sigmoid(< 0) < 0.5.
      row[e] = c.bias[e] == 0.0f ? r.unit() * 6.0f : -0.01f - (r.unit() + 1.0f) * 3.0f;
    }
    const int n_hi = 5 + r.below(16);
    for (int k = 0; k < n_hi; ++k) {
      const int e = r.below(E);
      row[e] = c.bias[e] == 0.0f ? 30.0f : 0.0f;
    }
  }
  return c;
}

// --------------------------------------------------------------------------------------
float median(std::vector<float> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void time_bs(int bs) {
  constexpr int kLaunches = 100, kWarmup = 300, kTrials = 50;
  Case c = random_case(0x1123 + bs, bs, bs, 6.0f, 0.01f);
  float *d_logits, *d_bias, *d_w;
  int *d_idx;
  CK(cudaMalloc(&d_logits, c.logits.size() * 4));
  CK(cudaMalloc(&d_bias, E * 4));
  CK(cudaMalloc(&d_w, size_t(bs) * K * 4));
  CK(cudaMalloc(&d_idx, size_t(bs) * K * 4));
  CK(cudaMemcpy(d_logits, c.logits.data(), c.logits.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_bias, c.bias.data(), E * 4, cudaMemcpyHostToDevice));
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  // CUDA graph of kLaunches back-to-back launches: measures GPU time, not host launch rate
  // (decode runs under CUDA graphs too).
  cudaGraphExec_t exec[2];
  for (int k = 0; k < 2; ++k) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < kLaunches; ++i) launch(k == 0, d_logits, d_bias, d_w, d_idx, bs, bs, s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&exec[k], g, 0));
    CK(cudaGraphDestroy(g));
  }
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int i = 0; i < kWarmup; ++i)
    for (int k = 0; k < 2; ++k) CK(cudaGraphLaunch(exec[k], s));
  CK(cudaStreamSynchronize(s));
  std::vector<float> us[2];
  for (int i = 0; i < kTrials; ++i) {
    for (int k = 0; k < 2; ++k) {  // interleave old/new so clock drift hits both
      CK(cudaEventRecord(e0, s));
      CK(cudaGraphLaunch(exec[k], s));
      CK(cudaEventRecord(e1, s));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      us[k].push_back(ms * 1000.0f / kLaunches);
    }
  }
  const float mo = median(us[0]), mn = median(us[1]);
  printf("| %2d | %6.2f | %6.2f | %5.2fx |\n", bs, mo, mn, mo / mn);
  for (int k = 0; k < 2; ++k) CK(cudaGraphExecDestroy(exec[k]));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  CK(cudaStreamDestroy(s));
  CK(cudaFree(d_logits));
  CK(cudaFree(d_bias));
  CK(cudaFree(d_w));
  CK(cudaFree(d_idx));
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  int rt = 0, drv = 0;
  CK(cudaRuntimeGetVersion(&rt));
  CK(cudaDriverGetVersion(&drv));
  printf("GPU: %s (sm_%d%d), %d SMs\nCUDA runtime %d.%d, driver supports %d.%d\n", p.name, p.major,
         p.minor, p.multiProcessorCount, rt / 1000, (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10);
  printf("old: %d threads, %zu B smem; new: %d threads, %zu B smem\n", oldk::kRouterSelectThreads, kOldSmem, newk::kRouterSelectThreads,
         kNewSmem);

  // 1. All experts tie: expect idx 0..7 and weight exactly 0.125 (checked here and by memcmp).
  {
    Case c = make("all_zero", 4, 4);
    check(c, true);
    float *d_l, *d_b;
    CK(cudaMalloc(&d_l, c.logits.size() * 4));
    CK(cudaMalloc(&d_b, E * 4));
    CK(cudaMemcpy(d_l, c.logits.data(), c.logits.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_b, c.bias.data(), E * 4, cudaMemcpyHostToDevice));
    for (int k = 0; k < 2; ++k) {
      Out o = run(k == 0, c, d_l, d_b);
      for (int i = 0; i < 4 * K; ++i) {
        if (o.idx[i] != i % K || o.w[i] != 0.125f) {
          printf("  all_zero %s: slot %d = (%d, %.9g)\n", k ? "new" : "old", i, o.idx[i], o.w[i]);
          ++g_fail;
          break;
        }
      }
    }
    CK(cudaFree(d_l));
    CK(cudaFree(d_b));
    printf("all-tie rows:          %s\n", g_fail ? "FAIL" : "OK");
  }

  const int sizes[] = {1, 2, 3, 5, 7, 8, 13, 31, 32, 33, 64, 257};
  const std::pair<int, int> padded_pairs[] = {{1, 2},  {1, 8},   {3, 5},   {5, 7},  {7, 8},
                                              {8, 16}, {13, 16}, {31, 32}, {32, 64}, {17, 4096}};

  // 2. Random continuous inputs.
  long f0 = g_fail;
  for (uint64_t seed = 1; seed <= 64; ++seed) {
    for (int bs : sizes) {
      check(random_case(seed * 1000 + bs, bs, bs, 6.0f, 0.01f), seed == 1 && bs == 8);
      check(random_case(seed * 7919 + bs, bs, bs, 2.0f, 0.5f));  // bias reorders more
    }
  }
  check(random_case(4096, 4096, 4096, 6.0f, 0.01f));
  printf("random:                %s\n", g_fail == f0 ? "OK" : "FAIL");

  // 3. Forced exact ties.
  f0 = g_fail;
  for (uint64_t seed = 1; seed <= 64; ++seed) {
    for (int bs : sizes) {
      check(quantized_tie_case(seed * 31 + bs, bs, bs), seed == 1 && bs == 8);
      check(score_tie_case(seed * 37 + bs, bs, bs), seed == 1 && bs == 8);
    }
  }
  printf("exact ties:            %s\n", g_fail == f0 ? "OK" : "FAIL");

  // 4. padded_tokens > active_tokens: filler rows must stay at the sentinel in both kernels.
  f0 = g_fail;
  for (uint64_t seed = 1; seed <= 16; ++seed) {
    for (const auto &ap : padded_pairs) {
      const int a = ap.first, pd = ap.second;
      const bool d = seed == 1 && a == 3;
      check(random_case(seed * 4243 + pd, a, pd, 6.0f, 0.01f), d);
      check(quantized_tie_case(seed * 4253 + pd, a, pd), d);
      check(score_tie_case(seed * 4259 + pd, a, pd), d);
    }
  }
  printf("padded > active:       %s\n", g_fail == f0 ? "OK" : "FAIL");

  printf("TOTAL: %ld cases, %ld active token rows, %ld failing cases\n\n", g_cases, g_rows, g_fail);

  printf("select kernel only, CUDA graph of 100 launches, median of 50 interleaved trials\n");
  printf("| bs | old (us) | new (us) | speedup |\n|---|---|---|---|\n");
  for (int bs : {1, 8, 16, 24, 32, 48, 64, 69, 70, 71, 72, 96, 128, 140, 141}) time_bs(bs);
  printf("\nRESULT: %s (%ld cases, %ld failing)\n", g_fail ? "FAIL" : "PASS", g_cases, g_fail);
  return g_fail ? 1 : 0;
}
