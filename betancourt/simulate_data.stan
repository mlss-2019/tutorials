data {
  int N;
}

transformed data {
  real mu = 1;
  real<lower=0> sigma = 0.75;
}

generated quantities {
  // Simulate data from observational model
  real y[N];
  for (n in 1:N) y[n] = normal_rng(mu, sigma);
}
