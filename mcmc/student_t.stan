data {
  real<lower=0> nu;
}

transformed data {
  int<lower=1> N = 10;
}

parameters {
  vector[N] x;
}

model {
  x ~ student_t(nu, 0, 1);
}
