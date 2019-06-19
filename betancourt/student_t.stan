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

generated quantities {
  real V = 0;
  for (n in 1:N)
    V = V - student_t_lpdf(x[n] | nu, 0, 1);
}
