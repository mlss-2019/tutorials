transformed data {
  int<lower=1> N = 25;
}

parameters {
  vector[N] x;
}

model {
  x ~ normal(0, 1);
}
