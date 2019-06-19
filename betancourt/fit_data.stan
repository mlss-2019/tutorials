data {
  int<lower=1> N; // Number of observations
  real y[N];      // Observations
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  // Prior model
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  
  // Observational model
  y ~ normal(mu, sigma);
}
