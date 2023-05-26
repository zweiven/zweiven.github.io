# Load a couple of libraries for data generation
library(MASS)           # makes multivariate normal sampling very simple
library(data.table)     # for data manipulation

set.seed(0) # set random number generator seed for replicability
# Generate a simple cross-sectional dataset
N = 1000          # number of observations
X.dim = 40          # number of x variables
X.cov = 0.5        # covariance among regressors
X.var = 1          # variance of individual regressor
e.var = 5          # variance of error
betas = c(1, 2, rep(0, X.dim-2)) # true parameters

# Define variance-covariance matrix for X
X.vcov = matrix(data = X.cov, 
                nrow = X.dim, 
                ncol = X.dim)
diag(X.vcov) = X.var

# Generate data
## covariates
x = mvrnorm(n = N, 
            mu = rep(0, X.dim), 
            Sigma = X.vcov)
## errors
e = rnorm(n = N, 
          mean = 0,
          sd = sqrt(e.var))

## outcomes
y = x %*% betas  + e

# stick together and write out to file
xs.data = as.data.table(cbind(y, x))
setnames(x = xs.data, 
         new = c("y", paste0("x", 1:X.dim)))
fwrite(x = xs.data, 
       file = "~/Dropbox/MLWorkshop/datasets/xsdata_sparse.csv")

####################################################
# Generate another dataset with some nonlinearities
# from a subset of the x variables
####################################################
betas = rep(1, 5)
y = betas[1]*x[,1] + betas[2] * x[,2] + betas[3] * x[,1]^2 + betas[4] * x[,2]^2 + betas[5] * x[,1] * x[,2] + e
# stick together and write out to file
xs.data = as.data.table(cbind(y, x[,1:2]))
setnames(x = xs.data, 
         new = c("y", paste0("x", 1:2)))
fwrite(x = xs.data, 
       file = "~/Dropbox/MLWorkshop/datasets/xsdata_nonlinear.csv")


###########################################################
# Generate a treatment effect dataset with selection
###########################################################
alphas = rep(1, 5)
treatment.index = x[,1:5] %*% alphas
treatment = as.integer(treatment.index > median(treatment.index))
y = betas[1]*treatment + betas[2] * x[,2] + betas[3] * x[,1]^2 + betas[4] * x[,2]^2 + betas[5] * x[,1] * x[,2] + e
xs.data = as.data.table(cbind(y, treatment, x[,1:5]))
setnames(x = xs.data, 
         new = c("y", "treatment", paste0("x", 1:5)))
fwrite(x = xs.data, 
       file = "~/Dropbox/MLWorkshop/datasets/xsdata_treatment.csv")


#########################
# Generate some panel data for a dataset with a single treated unit
# that might be suitable for a synthetic control study
#########################
N.control.units = 20
N.pre.periods = 10
N.post.periods = 10
N.periods = N.pre.periods + N.post.periods
unit.ids = 1:(N.control.units+1)
period.ids = 1:(N.periods)
sc.data = as.data.table(expand.grid(i=unit.ids, t=period.ids))
sc.data[,treat:= as.integer(i==1 & t>N.pre.periods)]
sc.data[,x1 := rnorm(n=N.periods, mean=i, sd=1), by=i]
sc.data[,x2 := rnorm(n=N.periods, mean=i, sd=1) + 0.5*x1 + rnorm(N.periods, mean=0, sd=1), by=i]
sc.data[,y:= i + 5*treat + 1*x1 + 1*x2 + rnorm(N.periods, mean=0, sd=1)]

fwrite(x = sc.data, 
       file = "~/Dropbox/MLWorkshop/datasets/scdata.csv")


#########################
# Simulate some data for an IV study with 
# a CATE that varies with multiple covariates
#########################
N = 1000
X.dim = 4          # number of x variables
X.cov = 0.5        # covariance among regressors
X.var = 1          # variance of individual regressor
e.var = 2          # variance of error
# true parameters
betas = c(1, # baseline TE
          rep(1, X.dim-1), # baseline covariate effects
          rep(1, X.dim-1)) # TE modifiers for cate
alpha = 2

# Define variance-covariance matrix for X
z = rnorm(N, 0, 1)
X.vcov = matrix(data = X.cov, 
                nrow = X.dim, 
                ncol = X.dim)
diag(X.vcov) = X.var
x = mvrnorm(n = N, 
            mu = rep(0, X.dim), 
            Sigma = X.vcov)
# add instrument relationship to treatment var
x[,1] = x[,1] + z*alpha

## errors
e = rnorm(n = N, 
          mean = 0,
          sd = sqrt(e.var))

## outcomes
x[,2]
y = cbind(x, x[,1]*x[,2:4]) %*% betas  + e

iv.data = as.data.table(cbind(y, x, z))
setnames(iv.data, c("y", "w", "x1", "x2", "x3", "z"))
fwrite(x = iv.data, 
       file = "~/Dropbox/MLWorkshop/datasets/ivdata.csv")


#########################
# Panel data for Double ML DiD
#########################

N.control.units = 500
N.treat.units = 500
N.units = N.control.units + N.treat.units
N.pre.periods = 1
N.post.periods = 1
N.periods = N.pre.periods + N.post.periods
unit.ids = 1:(N.units)
period.ids = 1:(N.periods)
did.data = as.data.table(expand.grid(i=unit.ids, t=period.ids))
did.data[,treat:= as.integer(i<=N.treat.units & t>N.pre.periods)]
did.data[,x1 := rnorm(n=N.periods, mean=i, sd=1), by=i]
did.data[,x2 := rnorm(n=N.periods, mean=i, sd=1) + 0.5*x1 + rnorm(N.periods, mean=0, sd=1), by=i]
did.data[,y:= i + 5*treat + 2*(period.ids-1) + 1*x1 + 1*x2 + rnorm(N.periods, mean=0, sd=1)]

fwrite(x = did.data, 
       file = "~/Dropbox/MLWorkshop/datasets/diddata.csv")
