import pystan

#random linear treatment effects with known variance
schools_code = """
data { //this block defines the data that the user inputs
    int<lower=0> J; // number of schools
    vector[J] y; // estimated treatment effects
    vector<lower=0>[J] sigma; // s.e. of effect estimates
}
parameters { //this block defines the parameters we want to trace
    real mu;
    real<lower=0> tau;
    vector[J] eta;
}
transformed parameters { //this block is used for parameters that are defined through other parameters or data
    vector[J] theta;
    theta = mu + tau * eta;
}
model { //this is where we set up the distributions
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
}
"""

schools_dat = {'J': 8,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

sm = pystan.StanModel(model_code=schools_code)
fit = sm.sampling(data=schools_dat, iter=1000, chains=4, seed = 1954989094)

fit
fit.plot()
fit.extract()
fit.extract('theta')

#from a file
sm = pystan.StanModel(file = 'KocPython2020/in-classMaterial/day11/8schools.stan')

#useful additional arguments: n_jobs and control
fit2 = sm.sampling(data=schools_dat, iter=1000, chains=4, seed = 1954989094, n_jobs=1, control = {'adapt_delta':.9, 'stepsize':20, 'max_treedepth':20})

#control has the following possible parameters (the above three are the most important)

#    adapt_engaged : bool
#    adapt_gamma : float, positive, default 0.05
#    adapt_delta : float, between 0 and 1, default 0.8
#    adapt_kappa : float, between default 0.75
#    adapt_t0 : float, positive, default 10
#    adapt_init_buffer : int, positive, defaults to 75
#    adapt_term_buffer : int, positive, defaults to 50
#    adapt_window : int, positive, defaults to 25
#    stepsize: float, positive
#    stepsize_jitter: float, between 0 and 1
#    metric : str, {“unit_e”, “diag_e”, “dense_e”}
#    max_treedepth : int, positive


"""
Example of running emcee to fit the parameters of a straight line.
"""

from __future__ import print_function, division

import os
import sys
import numpy as np

import matplotlib as mpl

# import PyStan
import pystan

os.chdir('KocPython2020/in-classMaterial/day11')

# import model and data
from createdata import *

# Create model code
line_code = """
data {{
    int<lower=0> N;      // number of data points
    real y[N];           // observed data points
    real x[N];           // abscissa points
    real<lower=0> sigma; // standard deviation
}}
parameters {{
    // parameters for the fit
    real m;
    real c;
}}
transformed parameters {{
    real theta[N];
    for (j in 1:N)
    theta[j] = m * x[j] + c; // straight line model
}}
model {{
    m ~ normal({mmu}, {msigma});     // prior on m (gradient)
    c ~ uniform({clower}, {cupper}); // prior on c (y-intercept)
    y ~ normal(theta, sigma);        // likelihood of the data given the model
}}
"""

# set the data and the abscissa
linear_data = {'N': M,          # number of data points
               'y': data,       # observed data (converted from numpy array to a list)
               'x': x,          # abscissa points (converted from numpy array to a list)
               'sigma': sigma}  # standard deviation

Nsamples = 1000 # set the number of iterations of the sampler
chains = 4      # set the number of chains to run with

# dictionary for inputs into line_code (this type of setting up priors is unnecessary, but thought you should know it exists)
linedict = {}
linedict['mmu'] = 0.0    # mean of Gaussian prior distribution for m
linedict['msigma'] = 10  # standard deviation of Gaussian prior distribution for m
linedict['clower'] = -10 # lower bound on uniform prior distribution for c
linedict['cupper'] = 10  # upper bound on uniform prior distribution for c

sm = pystan.StanModel(model_code=line_code.format(**linedict)); # compile model
fit3 = sm.sampling(data=linear_data, iter=Nsamples, chains=chains); # perform sampling

la = fit3.extract(permuted=True)  # return a dictionary of arrays

# extract the samples
postsamples = np.vstack((la['m'], la['c'])).T

# plot posterior samples (if corner.py is installed)
try:
    import corner # import corner.py
except ImportError:
    sys.exit(1)

print('Number of posterior samples is {}'.format(postsamples.shape[0]))

fig = corner.corner(postsamples, labels=[r"$m$", r"$c$"], truths=[m, c])


#TODO: Write a logit model that takes one continuous explanatory variable for a binary outcome. Then run it on the voting turnout data in turnout.csv (the outcome is whether or not they voted, vote, and the explanatory variable is income)


#TODO: Change the model to be a probit model and compare the results to the logit. Why would the estimates be different?


#TODO: Generalize the logit model to take K predictors. Include age, educate, and income as predictors.




#hierarchical modeling

#Hierarchical or multilevel modeling is a generalization of regression modeling.

#Multilevel models are regression models in which the constituent model parameters are given probability models. This implies that model parameters are allowed to vary by group.

#Observational units are often naturally clustered. Clustering induces dependence between observations, despite random sampling of clusters and random sampling within clusters.

#A hierarchical model is a particular multilevel model where parameters are nested within one another.

#Some multilevel structures are not hierarchical.

#    e.g. "country" and "year" are not nested, but may represent separate, but overlapping, clusters of parameters

#We will motivate this topic using an environmental epidemiology example.

#Example: Radon contamination (Gelman and Hill 2006)

#Radon is a radioactive gas that enters homes through contact points with the ground. It is a carcinogen that is the primary cause of lung cancer in non-smokers. Radon levels vary greatly from household to household.

#radon

#The EPA did a study of radon levels in 80,000 houses. Two important predictors:

#    measurement in basement or first floor (radon higher in basements)
#    county uranium level (positive correlation with radon levels)

#We will focus on modeling radon levels in Minnesota.

#The hierarchy in this example is households within county.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import radon data
srrs2 = pd.read_csv('srrs2.dat')
srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2.assign(fips=srrs2.stfips*1000 + srrs2.cntyfips)[srrs2.state=='MN']
# Next, obtain the county-level predictor, uranium, by combining two variables.
cty = pd.read_csv('cty.dat')
cty_mn = cty[cty.st=='MN'].copy()
cty_mn[ 'fips'] = 1000*cty_mn.stfips + cty_mn.ctfips
# Use the merge method to combine home- and county-level information in a single DataFrame.
srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
u = np.log(srrs_mn.Uppm)
n = len(srrs_mn)

srrs_mn.head()

# We also need a lookup table (dict) for each unique county, for indexing.
srrs_mn.county = srrs_mn.county.str.strip()
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)

# Finally, create local copies of variables.
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))
county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
floor_measure = srrs_mn.floor.values
# Distribution of radon levels in MN (log scale):
srrs_mn.activity.apply(lambda x: np.log(x+0.1)).hist(bins=25)

#Conventional approaches

#The two conventional alternatives to modeling radon exposure represent the two extremes of the bias-variance tradeoff:

#Complete pooling:

#Treat all counties the same, and estimate a single radon level.
#yi=α+βxi+ϵi

#No pooling:

#Model radon in each county independently.
#yi=αj[i]+βxi+ϵi

#where j=1,…,85

#The errors ϵi may represent measurement error, temporal within-house variation, or variation among houses.

#To specify this model in Stan, we begin by constructing the data block, which includes vectors of log-radon measurements (y) and floor measurement covariates (x), as well as the number of samples (N).

pooled_data = """
data {
  int<lower=0> N; 
  vector[N] x;
  vector[N] y;
}
"""

#Next we initialize our parameters, which in this case are the linear model coefficients and the normal scale parameter. Notice that sigma is constrained to be positive

pooled_parameters = """
parameters {
  vector[2] beta;
  real<lower=0> sigma;
} 
"""

#Finally, we model the log-radon measurements as a normal sample with a mean that is a function of the floor measurement.

pooled_model = """
model {
  y ~ normal(beta[1] + beta[2] * x, sigma);
}
"""

#We then pass the code, data, and parameters to the stan function. The sampling requires specifying how many iterations we want, and how many parallel chains to sample. Here, we will sample 2 chains of length 1000.

pooled_data_dict = {'N': len(log_radon),
               'x': floor_measure,
               'y': log_radon}
#notice we did not compile the model first --- either is fine
pooled_fit = pystan.stan(model_code=pooled_data + pooled_parameters + pooled_model, data=pooled_data_dict, iter=1000, chains=2)
                         
#The sample can be extracted for plotting and summarization.

pooled_sample = pooled_fit.extract(permuted=True)

b0, m0 = pooled_sample['beta'].T.mean(1)

plt.scatter(srrs_mn.floor, np.log(srrs_mn.activity+0.1))
xvals = np.linspace(-0.2, 1.2)
plt.plot(xvals, m0*xvals+b0, 'r--')

#At the other end of the extreme, we can fit separate (independent) means for each county. The only things that are shared in this model are the coefficient for the basement measurement effect, and the standard deviation of the error.

unpooled_model = """data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[85] a;
  real beta;
  real<lower=0,upper=100> sigma;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = beta * x[i] + a[county[i]];
}
model {
  y ~ normal(y_hat, sigma);
}"""

unpooled_data = {'N': len(log_radon),
               'county': county+1, # Stan counts starting at 1
               'x': floor_measure,
               'y': log_radon}

unpooled_fit = pystan.stan(model_code=unpooled_model, data=unpooled_data, iter=1000, chains=2)

unpooled_estimates = pd.Series(unpooled_fit['a'].mean(0), index=mn_counties)
unpooled_se = pd.Series(unpooled_fit['a'].std(0), index=mn_counties)

#We can plot the ordered estimates to identify counties with high radon levels:

order = unpooled_estimates.sort_values().index

plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
    plt.plot([i,i], [m-se, m+se], 'b-')
plt.xlim(-1,86); plt.ylim(-1,4)
plt.ylabel('Radon estimate');plt.xlabel('Ordered county');

#Here are visual comparisons between the pooled and unpooled estimates for a subset of counties representing a range of sample sizes.

sample_counties = ('LAC QUI PARLE', 'AITKIN', 'KOOCHICHING', 
                    'DOUGLAS', 'CLAY', 'STEARNS', 'RAMSEY', 'ST LOUIS')

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
axes = axes.ravel()
m = unpooled_fit['beta'].mean(0)
for i,c in enumerate(sample_counties):
    y = srrs_mn.log_radon[srrs_mn.county==c]
    x = srrs_mn.floor[srrs_mn.county==c]
    axes[i].scatter(x + np.random.randn(len(x))*0.01, y, alpha=0.4)
    
    # No pooling model
    b = unpooled_estimates[c]
    
    # Plot both models and data
    xvals = np.linspace(-0.2, 1.2)
    axes[i].plot(xvals, m*xvals+b)
    axes[i].plot(xvals, m0*xvals+b0, 'r--')
    axes[i].set_xticks([0,1])
    axes[i].set_xticklabels(['basement', 'floor'])
    axes[i].set_ylim(-1, 3)
    axes[i].set_title(c)
    if not i%2:
        axes[i].set_ylabel('log radon level')
        
#Neither of these models are satisfactory:
#    if we are trying to identify high-radon counties, pooling is useless
#    we do not trust extreme unpooled estimates produced by models using few observations

#Multilevel and hierarchical models

#When we pool our data, we imply that they are sampled from the same model. This ignores any variation among sampling units (other than sampling variance)

#When we analyze data unpooled, we imply that they are sampled independently from separate models. At the opposite extreme from the pooled case, this approach claims that differences between sampling units are to large to combine them

#In a hierarchical model, parameters are viewed as a sample from a population distribution of parameters. Thus, we view them as being neither entirely different or exactly the same. This is parital pooling.

#We can use PyStan to easily specify multilevel models, and fit them using Hamiltonian Monte Carlo.

#    Estimates for counties with smaller sample sizes will shrink towards the state-wide average.
#    Estimates for counties with larger sample sizes will be closer to the unpooled county estimates.

partial_pooling = """
data {
  int<lower=0> N; 
  int<lower=1,upper=85> county[N];
  vector[N] y;
} 
parameters {
  vector[85] a;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = a[county[i]];
}
model {
  mu_a ~ normal(0, 1);
  a ~ normal (10 * mu_a, sigma_a);

  y ~ normal(y_hat, sigma_y);
}"""

#Notice now we have two standard deviations, one describing the residual error of the observations, and another the variability of the county means around the average.

partial_pool_data = {'N': len(log_radon),
               'county': county+1, # Stan counts starting at 1
               'y': log_radon}

partial_pool_fit = pystan.stan(model_code=partial_pooling, data=partial_pool_data, iter=1000, chains=2)

sample_trace = partial_pool_fit['a']

fig, axes = plt.subplots(1, 2, figsize=(14,6), sharex=True, sharey=True)
samples, counties = sample_trace.shape
jitter = np.random.normal(scale=0.1, size=counties)

n_county = srrs_mn.groupby('county')['idnum'].count()
unpooled_means = srrs_mn.groupby('county')['log_radon'].mean()
unpooled_sd = srrs_mn.groupby('county')['log_radon'].std()
unpooled = pd.DataFrame({'n':n_county, 'm':unpooled_means, 'sd':unpooled_sd})
unpooled['se'] = unpooled.sd/np.sqrt(unpooled.n)

axes[0].plot(unpooled.n + jitter, unpooled.m, 'b.')
for j, row in zip(jitter, unpooled.iterrows()):
    name, dat = row
    axes[0].plot([dat.n+j,dat.n+j], [dat.m-dat.se, dat.m+dat.se], 'b-')
axes[0].set_xscale('log')
axes[0].hlines(sample_trace.mean(), 0.9, 100, linestyles='--')

        
samples, counties = sample_trace.shape
means = sample_trace.mean(axis=0)
sd = sample_trace.std(axis=0)
axes[1].scatter(n_county.values + jitter, means)
axes[1].set_xscale('log')
axes[1].set_xlim(1,100)
axes[1].set_ylim(0, 3)
axes[1].hlines(sample_trace.mean(), 0.9, 100, linestyles='--')
for j,n,m,s in zip(jitter, n_county.values, means, sd):
    axes[1].plot([n+j]*2, [m-s, m+s], 'b-')

#Notice the difference between the unpooled and partially-pooled estimates, particularly at smaller sample sizes. The former are both more extreme and more imprecise.

#Varying intercept model

#This model allows intercepts to vary across county, according to a random effect.
#yi=αj[i]+βxi+ϵi

#where
#ϵi∼N(0,σ2y)

#and the intercept random effect:
#αj[i]∼N(μα,σ2α)

#As with the the “no-pooling” model, we set a separate intercept for each county, but rather than fitting separate least squares regression models for each county, multilevel modeling shares strength among counties, allowing for more reasonable inference in counties with little data.

varying_intercept = """
data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J] a;
  real b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {

  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = a[county[i]] + x[i] * b;
}
model {
  sigma_a ~ uniform(0, 100);
  a ~ normal (mu_a, sigma_a);

  b ~ normal (0, 1);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
"""

varying_intercept_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'x': floor_measure,
                          'y': log_radon}

varying_intercept_fit = pystan.stan(model_code=varying_intercept, data=varying_intercept_data, iter=1000, chains=2)

a_sample = pd.DataFrame(varying_intercept_fit['a'])

import seaborn as sns
sns.set(style="ticks", palette="muted", color_codes=True)

# Plot the orbital period with horizontal boxes
plt.figure(figsize=(16, 6))
sns.boxplot(data=a_sample, whis=np.inf, color="c")

varying_intercept_fit.plot(pars=['sigma_a', 'b']);

varying_intercept_fit['b'].mean()

xvals = np.arange(2)
bp = varying_intercept_fit['a'].mean(axis=0)
mp = varying_intercept_fit['b'].mean()
for bi in bp:
    plt.plot(xvals, mp*xvals + bi, 'bo-', alpha=0.4)
plt.xlim(-0.1,1.1);

#It is easy to show that the partial pooling model provides more objectively reasonable estimates than either the pooled or unpooled models, at least for counties with small sample sizes.

fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)
axes = axes.ravel()
for i,c in enumerate(sample_counties):
    
    # Plot county data
    y = srrs_mn.log_radon[srrs_mn.county==c]
    x = srrs_mn.floor[srrs_mn.county==c]
    axes[i].scatter(x + np.random.randn(len(x))*0.01, y, alpha=0.4)
    
    # No pooling model
    m,b = unpooled_estimates[['floor', c]]
    
    xvals = np.linspace(-0.2, 1.2)
    # Unpooled estimate
    axes[i].plot(xvals, m*xvals+b)
    # Pooled estimate
    axes[i].plot(xvals, m0*xvals+b0, 'r--')
    # Partial pooling esimate
    axes[i].plot(xvals, mp*xvals+bp[county_lookup[c]], 'k:')
    axes[i].set_xticks([0,1])
    axes[i].set_xticklabels(['basement', 'floor'])
    axes[i].set_ylim(-1, 3)
    axes[i].set_title(c)
    if not i%2:
        axes[i].set_ylabel('log radon level')


#Varying slope model

#Alternatively, we can posit a model that allows the counties to vary according to how the location of measurement (basement or floor) influences the radon reading.
#yi=α+βj[i]xi+ϵi

varying_slope = """
data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] x;
  vector[N] y;
} 
parameters {
  real a;
  vector[J] b;
  real mu_b;
  real<lower=0,upper=100> sigma_b;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {

  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = a + x[i] * b[county[i]];
}
model {
  sigma_b ~ uniform(0, 100);
  b ~ normal (mu_b, sigma_b);

  a ~ normal (0, 1);

  sigma_y ~ uniform(0, 100);
  y ~ normal(y_hat, sigma_y);
}
"""

varying_slope_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'x': floor_measure,
                          'y': log_radon}

varying_slope_fit = pystan.stan(model_code=varying_slope, data=varying_slope_data, iter=1000, chains=2)

b_sample = pd.DataFrame(varying_slope_fit['b'])

# Plot the orbital period with horizontal boxes
plt.figure(figsize=(16, 6))
sns.boxplot(data=b_sample, whis=np.inf, color="c")

xvals = np.arange(2)
b = varying_slope_fit['a'].mean()
m = varying_slope_fit['b'].mean(axis=0)
for mi in m:
    plt.plot(xvals, mi*xvals + b, 'bo-', alpha=0.4)
plt.xlim(-0.2, 1.2);

#Varying intercept and slope model

#The most general model allows both the intercept and slope to vary by county:
#yi=αj[i]+βj[i]xi+ϵi

varying_intercept_slope = """
data {
  int<lower=0> N;
  int<lower=0> J;
  vector[N] y;
  vector[N] x;
  int county[N];
}
parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
  vector[J] a;
  vector[J] b;
  real mu_a;
  real mu_b;
}

model {
  mu_a ~ normal(0, 100);
  mu_b ~ normal(0, 100);

  a ~ normal(mu_a, sigma_a);
  b ~ normal(mu_b, sigma_b);
  y ~ normal(a[county] + b[county].*x, sigma);
}
"""

varying_intercept_slope_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'x': floor_measure,
                          'y': log_radon}

varying_intercept_slope_fit = pystan.stan(model_code=varying_intercept_slope, 
                                          data=varying_intercept_slope_data, 
                                          iter=1000, chains=2)

xvals = np.arange(2)
b = varying_intercept_slope_fit['a'].mean(axis=0)
m = varying_intercept_slope_fit['b'].mean(axis=0)
for bi,mi in zip(b,m):
    plt.plot(xvals, mi*xvals + bi, 'bo-', alpha=0.4)
plt.xlim(-0.1, 1.1);

#Adding group-level predictors

#A primary strength of multilevel models is the ability to handle predictors on multiple levels simultaneously. If we consider the varying-intercepts model above:
#yi=αj[i]+βxi+ϵi

#we may, instead of a simple random effect to describe variation in the expected radon value, specify another regression model with a county-level covariate. Here, we use the county uranium reading uj

#, which is thought to be related to radon levels:
#αj=γ0+γ1uj+ζj
#ζj∼N(0,σ2α)

#Thus, we are now incorporating a house-level predictor (floor or basement) as well as a county-level predictor (uranium).

#Note that the model has both indicator variables for each county, plus a county-level covariate. In classical regression, this would result in collinearity. In a multilevel model, the partial pooling of the intercepts towards the expected value of the group-level linear model avoids this.

#Group-level predictors also serve to reduce group-level variation σα
#. An important implication of this is that the group-level estimate induces stronger pooling.

hierarchical_intercept = """
data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] y;
} 
parameters {
  vector[J] a;
  vector[2] b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;
  vector[N] m;

  for (i in 1:N) {
    m[i] = a[county[i]] + u[i] * b[1];
    y_hat[i] = m[i] + x[i] * b[2];
  }
}
model {
  mu_a ~ normal(0, 1);
  a ~ normal(mu_a, sigma_a);
  b ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
"""

hierarchical_intercept_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'u': u,
                          'x': floor_measure,
                          'y': log_radon}

hierarchical_intercept_fit = pystan.stan(model_code=hierarchical_intercept, data=hierarchical_intercept_data, 
                                         iter=1000, chains=2)

#a_means = M_hierarchical.a.trace().mean(axis=0)
m_means = hierarchical_intercept_fit['m'].mean(axis=0)
plt.scatter(u, m_means)
g0 = hierarchical_intercept_fit['mu_a'].mean()
g1 = hierarchical_intercept_fit['b'][:, 0].mean()
xvals = np.linspace(-1, 0.8)
plt.plot(xvals, g0+g1*xvals, 'k--')
plt.xlim(-1, 0.8)

m_se = hierarchical_intercept_fit['m'].std(axis=0)
for ui, m, se in zip(u, m_means, m_se):
    plt.plot([ui,ui], [m-se, m+se], 'b-')
plt.xlabel('County-level uranium'); plt.ylabel('Intercept estimate')



#The standard errors on the intercepts are narrower than for the partial-pooling model without a county-level covariate.
#Correlations among levels

#In some instances, having predictors at multiple levels can reveal correlation between individual-level variables and group residuals. We can account for this by including the average of the individual predictors as a covariate in the model for the group intercept.
#αj=γ0+γ1uj+γ2x¯+ζj

#These are broadly referred to as contextual effects.

# Create new variable for mean of floor across counties
xbar = srrs_mn.groupby('county')['floor'].mean().rename(county_lookup).values

x_mean = xbar[county]

contextual_effect = """
data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=1,upper=J> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] x_mean;
  vector[N] y;
} 
parameters {
  vector[J] a;
  vector[3] b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;

  for (i in 1:N)
    y_hat[i] = a[county[i]] + u[i]*b[1] + x[i]*b[2] + x_mean[i]*b[3];
}
model {
  mu_a ~ normal(0, 1);
  a ~ normal(mu_a, sigma_a);
  b ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
"""

contextual_effect_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'u': u,
                          'x_mean': x_mean,
                          'x': floor_measure,
                          'y': log_radon}

contextual_effect_fit = pystan.stan(model_code=contextual_effect, data=contextual_effect_data, 
                                         iter=1000, chains=2)

contextual_effect_fit['b'].mean(0)

contextual_effect_fit.plot('b');

#So, we might infer from this that counties with higher proportions of houses without basements tend to have higher baseline levels of radon. Perhaps this is related to the soil type, which in turn might influence what type of structures are built.


#Prediction

#Gelman (2006) used cross-validation tests to check the prediction error of the unpooled, pooled, and partially-pooled models

#root mean squared cross-validation prediction errors:

#    unpooled = 0.86
#    pooled = 0.84
#    multilevel = 0.79

#There are two types of prediction that can be made in a multilevel model:

#    a new individual within an existing group
#    a new individual within a new group

#For example, if we wanted to make a prediction for a new house with no basement in St. Louis county, we just need to sample from the radon model with the appropriate intercept.

county_lookup['ST LOUIS']

#That is,
#ỹ i∼N(α69+β(xi=1),σ2y)

#This is simply a matter of adding a single additional line in PyStan:

contextual_pred = """
data {
  int<lower=0> J; 
  int<lower=0> N; 
  int<lower=0,upper=J> stl;
  real u_stl;
  real xbar_stl;
  int<lower=1,upper=J> county[N];
  vector[N] u;
  vector[N] x;
  vector[N] x_mean;
  vector[N] y;
} 
parameters {
  vector[J] a;
  vector[3] b;
  real mu_a;
  real<lower=0,upper=100> sigma_a;
  real<lower=0,upper=100> sigma_y;
} 
transformed parameters {
  vector[N] y_hat;
  real stl_mu;

  for (i in 1:N)
    y_hat[i] = a[county[i]] + u[i] * b[1] + x[i] * b[2] + x_mean[i] * b[3];
    
  stl_mu = a[stl+1] + u_stl * b[1] + b[2] + xbar_stl * b[3];
 }
model {
  mu_a ~ normal(0, 1);
  a ~ normal(mu_a, sigma_a);
  b ~ normal(0, 1);
  y ~ normal(y_hat, sigma_y);
}
generated quantities {
  real y_stl;
  
  y_stl = normal_rng(stl_mu, sigma_y);
}
"""

contextual_pred_data = {'N': len(log_radon),
                          'J': len(n_county),
                          'county': county+1, # Stan counts starting at 1
                          'u': u,
                          'x_mean': x_mean,
                          'x': floor_measure,
                          'y': log_radon,
                          'stl': 69,
                          'u_stl': np.log(cty_mn[cty_mn.cty=='STLOUIS'].Uppm.values)[0],
                          'xbar_stl': xbar[69]}

contextual_pred_fit = pystan.stan(model_code=contextual_pred, data=contextual_pred_data, 
                                         iter=1000, chains=2)

contextual_pred_fit.plot('y_stl');


#TODO: How would we make a prediction from a new county (e.g. one not included in this dataset)?


#Change the first set of models:

#TODO: Make an indicator for the observation being of white race. Allow the intercepts of the logit to vary by race. Place priors on the mean and standard deviation of the random intercepts.


#TODO: Allow the economic slopes and intercept to vary by race


