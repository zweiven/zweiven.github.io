# Methods designed to use machine learning responsibly
# in the service of causal inference.

library(data.table)
library(hdm)  # for post (double) LASSO
library(DoubleML)
library(mlr3) # for the backend behind DoubleML
library(mlr3learners)
library(ArCo)
library(glmnet)
library(MCPanel)
library(grf)


# Load some sample data
xs.data = fread("http://stevejmiller.com/ml/datasets/xsdata_treatment.csv")

# Estimate OLS
ols.mod = lm(y ~ ., data=xs.data)
summary(ols.mod)

#################################
# Estimate post double lasso
#################################
pdl.mod = rlassoEffect(x = as.matrix(xs.data[,-c("y","treatment")]), # matrix of covariates
                       y = xs.data$y,                         # outcomes
                       d = xs.data$treatment,                        # treatment
                       method = "double selection")
pdl.mod$selection.index # see what was selected
summary(pdl.mod)

#################################
# Double ML
#################################
dml.data = DoubleMLData$new(xs.data,
                            y_col="y",
                            d_cols="treatment",
                            x_cols=colnames(xs.data[,-c("y","treatment")]))

# Define the learners we want to use for the nuisance functions
# in both the outcome equation and the selection equation
rf.learner = lrn("regr.ranger", 
                 num.trees=500, 
                 max.depth=20, 
                 min.node.size=2)
outcome.eq.learner = rf.learner$clone()
selection.eq.learner = rf.learner$clone()

# Create a new modeling object and fit it
dml.mod = DoubleMLPLR$new(dml.data, 
                          ml_l = outcome.eq.learner,
                          ml_m = selection.eq.learner,
                          n_folds = 10)
dml.mod$fit()

# Print estimates
print(dml.mod)

###############################
# IV
###############################

iv.data = fread("http://stevejmiller.com/ml/datasets/ivdata.csv")


###############################
# Artificial controls
###############################
sc.data = fread("http://stevejmiller.com/ml/datasets/scdata.csv")
dcast(sc.data, i ~ t, value.var = y)
arco.sc.data = panel_to_ArCo_list(sc.data, time="t", unit="i", variables="y")

arco.mod = fitArCo(data = arco.sc.data, 
                   fn = cv.glmnet, 
                   p.fn = predict, 
                   treated.unit = 1,
                   t0 = 11, 
                   VCOV.type = "iid",
                   boot.cf = T,
                   R = 200,
                   l = 1)
# these are the point estimates for the
# treatment effect and the confidence interval
arco.mod$delta

# We can plot the series together with the 
# fitted counterfactual in the pre- and post-treatment periods
# (with the block bootstrap used to generate a confidence band for the counterfactual)
plot(arco.mod, 
     display.fitted=TRUE ,
     confidence.bands = TRUE, 
     alpha = 0.05)



###############################
# Matrix completion
###############################
# We can actually use the same data example
# as for artificial controls.
# We'll set up the matrix for Y(0)
data.mat = as.matrix(dcast(sc.data[,.(i, t, y)], i ~ t, value.var="y")[,-1])
obs.mask = as.matrix(dcast(sc.data[,.(i, t, untreat=as.integer(!treat))], i ~ t, value.var="untreat")[,-1])
mc.mod = mcnnm_cv(data.mat,
                  obs.mask)
# Point estimate is the average
# of the difference between observed Y(1) and predicted
# Y(0) in the post-period
# First, get predictions, adding back in the 
# fixed effects
preds = mc.mod$L + outer(mc.mod$u, mc.mod$v, '+')

# second, compute mean difference between outcomes
# and predicted counterfactuals among treated observations 
mean((data.mat - preds)[!as.logical(obs.mask)])

# No theoretical results for inference yet.
# Per Athey et al., however, we can do a randomization-inference inspired approach.
# We'll keep the control units only, shuffle their order,
# and consider some to be treated, then look at the distribution of estimates.
# This implementation is from https://synth-inference.github.io/synthdid/articles/paper-results.html#summary-1
# to be a bit more readable for workshop purposes

MCPanelSE = function(data.mat,
                     obs.mask,
                     replications=200) {
  # treated observations are those with zero entries (missing Y(0))
  treated.obs = which(rowSums(obs.mask) < ncol(obs.mask))
  untreated.obs = (1:nrow(obs.mask))[-treated.obs]
  n.treat = length(treated.obs)
  n.untreat = length(untreated.obs)
  
  GetMCEstimate = function(data, obs) { 
    mc.mod = mcnnm_cv(data, obs)
    preds = mc.mod$L + outer(mc.mod$u, mc.mod$v, '+')
    return(mean((data - preds)[!as.logical(obs)]))
  }
  untreat.mat = data.mat[untreated.obs,]
  
  replicated.estimates = replicate(replications, 
                                   GetMCEstimate(untreat.mat[sample(1:n.untreat),],
                                                 obs.mask[c(treated.obs,
                                                            untreated.obs[1:(n.untreat-n.treat)]),]))
  return(sqrt((replications-1)/replications) * sd(replicated.estimates))
}
mc.mod.se = MCPanelSE(data.mat, obs.mask, replications = 100)


# Train an instrumental forest.
# In this example, we have 3 confounders
# that also modify the treatment function linearly (slope=1)
# and an instrument
iv.mod = instrumental_forest(X = as.matrix(iv.data[,c("x1", "x2", "x3")]), # covariates
                             Y = iv.data$y,                                # outcome
                             W = iv.data$w,                                # treatment
                             Z = iv.data$z,                                # instrument
                             num.trees = 5000                              # grow more trees for variance estimation
                             )

# Visualizing/summarizing treatment effect estimates is always
# messy. We'll make a partial dependence plot by
# getting estimates of treatment effects at some test points
# rather than at the training points. 
# Specifically, we'll set x1 to be a grid from -4 to 4 and
# x2 and x3 will be held at zero.
x.test = matrix(0, nrow=51, ncol=3)
x.test[,1] = seq(from=-4, to=4, length.out=51)

# Predict the treatment effect at those points
iv.te.preds = predict(iv.mod, newdata = x.test, estimate.variance = TRUE)

# Plot treatment effect against x1
# Assemble data
pred.comp = data.table(preds = iv.te.preds$predictions, 
                       lower.ci = iv.te.preds$predictions - 1.96*sqrt(iv.te.preds$variance.estimates),
                       upper.ci = iv.te.preds$predictions + 1.96*sqrt(iv.te.preds$variance.estimates),
                       x1=x.test[,1])
# Plot point estimates and CI
plot(preds ~ x1, data=pred.comp, type="l")
lines(upper.ci ~ x1, data=pred.comp, col="red")
lines(lower.ci ~ x1, data=pred.comp, col="red")

# From those TE estimates you can actually see one property of tree-based methods:
# they tend to flatten as they approach the edges of their support.
hist(iv.data$x1)
# That's not unreasonable -- forests are built on piecewise-constant
# building blocks (trees) so they extrapolate as a constant

