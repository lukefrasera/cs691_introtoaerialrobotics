#include <stdio.h>
#include <random>
#include <cmath>
#include <stdint.h>
#include <vector>
#include <fstream>

typedef struct KalmanParam_s {
  float est_variance;
  float proc_variance;
} KalmanParam_t;

typedef struct KalmanOutput_s {
  std::vector<float> estimate;
  std::vector<float> error;
} KalmanOutput_t;

KalmanOutput_t FilterData(std::vector<float> samples, KalmanParam_t params, uint32_t iters) {
  std::vector<float> xhat(samples.size());
  std::vector<float> P(samples.size());
  std::vector<float> xhatminus(samples.size());
  std::vector<float> Pminus(samples.size());
  std::vector<float> K(samples.size());

  for (int i = 0; i < samples.size(); ++i) {
    xhat[i]      = 0;
    P[i]         = 0;
    xhatminus[i] = 0;
    Pminus[i]    = 0;
    K[i]         = 0;
  }
  xhat[0] = 0.0; // Initial value estimate
  P[0]    = 1.0; // Initial error estimate

  for (int i = 1; i < samples.size(); ++i) {
    // time update
    xhatminus[i] = xhat[i-1];
    Pminus[i] = P[i-1] + params.proc_variance;

    // memsuement update
    K[i] = Pminus[i] / (Pminus[i] + params.est_variance);
    xhat[i] = xhatminus[i] + K[i] * (samples[i] - xhatminus[i]);
    P[i] = (1 - K[i]) * Pminus[i];
  }
  KalmanOutput_t result = {xhat, Pminus};
  return result;
}

int main(int argc, char *argv[]) {
  // Generate Data
  KalmanParam_t params = {std::pow(0.1, 2), 1e-5};
  int n_iter = 50;
  float truth_value = -0.37727;
  float measure_std = 0.1;
  std::default_random_engine gen;
  std::normal_distribution<float> distrib(truth_value, measure_std);

  std::vector<float> samples(n_iter);
  for (int i = 0; i < samples.size(); ++i) {
    samples[i] = distrib(gen);
  }

  KalmanOutput_t result = FilterData(samples, params, n_iter);

  std::ofstream fout;
  fout.open("result.txt");
  fout << samples[0];
  for (int i = 1; i < samples.size(); ++i) {
    fout << ", " << samples[i];
  }
  fout << std::endl;

  fout << result.estimate[0];
  for (int i = 1; i < result.estimate.size(); ++i)
    fout << ", " << result.estimate[i];
  fout << std::endl;

  fout << result.error[0];
  for (int i = 1; i < result.error.size(); ++i)
    fout << ", " << result.error[i];
  fout << std::endl;

  return 0;
}