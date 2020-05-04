data {
    int<lower=1> N;
    int<lower=1> K;
    int<lower=1> M;
    matrix[N,K] X;
    matrix[N,M] X_corr;
    vector[N] y;
}
parameters {
    real<lower=0> nug;
    real<lower=0> sig_sq;
    vector<lower=0>[M] d1;
    vector<lower=0>[M] d2;
    vector[K] b;
}
model {
    matrix[N,N] Sigma;
    vector[N] mu;
    matrix[N,K] Mu;
    vector[M] d;

    for(m in 1:M){
        d1[m] ~ gamma(1,20);
        d2[m] ~ gamma(10,10);
        d[m] = .5*(d1[m] + d2[m]);
    }
    for (i in 1:(N-1)) {
    for (j in (i+1):N) {
        vector[M] summand;
        for(m in 1:M){
            summand[m] = -pow(X_corr[i,m] - X_corr[j,m],2)/d[m];
        }
        Sigma[i,j] = exp(sum(summand));
        Sigma[j,i] = Sigma[i,j];
    }
}
    for (i in 1:N){
        for(k in 1:K){
            Mu[i,k] = X[i,k]*b[k];
        }
        mu[i]=sum(Mu[i,1:K]);
    }
    for (i in 1:N) Sigma[i,i] = 1 + nug; // + jitter

    sig_sq ~ inv_gamma(1,1);

    nug ~ exponential(1);

    b ~ normal(0,3);
    y ~ multi_normal(mu,sig_sq*Sigma);
}

