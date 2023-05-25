library(data.table)    # for (to me) better ways to work with tabular data
library(caret)         # for common machine learning methods
library(caretEnsemble) # for ensemble approaches
library(haven)         # for reading in .dta files


##################################################
# Basic examples with simulated data
##################################################
set.seed(42)
xs.data = fread("http://stevejmiller.com/ml/datasets/xsdata_sparse.csv")
#xs.data = fread("http://stevejmiller.com/ml/datasets/xsdata_nonlinear.csv")
xs.data    # will print first/last 5 rows of data to console

# To assess true out-of-sample performance, we can split our data into
# training and test sets:
train.indices = createDataPartition(y = xs.data$y,   # y vector
                                    times = 1,       # number of splits to do
                                    p = 0.8,         # probability of ending up in training set
                                    list=F)          # get indices of training samples in matrix rather than list

train.data = xs.data[train.indices, ]
test.data = xs.data[-train.indices,]

# LASSO
candidate.lambdas = c(seq(from=0, to=0.5, length.out = 100))     # lambdas to assess
lasso.mod = train(y = train.data$y,                      # outcome vector
                  x = train.data[,2:ncol(train.data)],                  # predictor matrix
                  method = 'glmnet',                  # the glmnet package does the work behind the scenes (for nonlinear elastic nets)
                  preProcess = c("center", "scale"),  # center and scale predictors
                  trControl = trainControl(method = "cv", 
                                           number = 10), # do 10-fold cross-validation to assess performance
                  tuneGrid = expand.grid(alpha = 1,    # weight attached to ridge penalty component
                                         lambda = candidate.lambdas))  # weight attached to overall penalty
# here's the best lambda
lasso.mod$bestTune
# and the coefficients from that model
coef(lasso.mod$finalModel, lasso.mod$bestTune$lambda)
# and predictions...
# in-sample
lasso.preds.in = predict(object = lasso.mod)
# out-of-sample
lasso.preds.out = predict(object = lasso.mod, 
                          newdata = test.data)

# compare to a simple linear model.
# Note the dot is shorthand for 'everything else in the dataset'
lm.mod = lm(formula = y ~ ., data = train.data)  
lm.preds.in = predict(object = lm.mod)
lm.preds.out = predict(object = lm.mod, 
                       newdata = test.data)

# assess root mean squared error
RMSE(lasso.preds.out, test.data$y)     # of LASSO
RMSE(lm.preds.out, test.data$y)        # of OLS


# We can easily change the calls to caret to estimate some of the 
# other models we've seen.

# Ridge regression just entails setting the alpha part of elastic net to zero
ridge.mod = train(y = train.data$y,                      # outcome vector
                  x = train.data[,2:ncol(train.data)],                  # predictor matrix
                  method = 'glmnet',                  # the glmnet package does the work behind the scenes (for nonlinear elastic nets)
                  preProcess = c("center", "scale"),  # center and scale predictors
                  trControl = trainControl(method = "cv", 
                                           number = 10), # do 10-fold cross-validation to assess performance
                  tuneGrid = expand.grid(alpha = 0,    # weight attached to ridge penalty component
                                         lambda = candidate.lambdas))  # weight attached to overall penalty
ridge.preds.out = predict(object = ridge.mod, 
                          newdata = test.data)

# A full elastic net requires we also provide a set of choices for alpha
# to optimize over.
candidate.alphas = seq(from=0, to=1, length.out=10)
enet.mod = train(y = train.data$y,                      # outcome vector
                  x = train.data[,2:ncol(train.data)],                  # predictor matrix
                  method = 'glmnet',                  # the glmnet package does the work behind the scenes (for nonlinear elastic nets)
                  preProcess = c("center", "scale"),  # center and scale predictors
                  trControl = trainControl(method = "cv", 
                                           number = 10), # do 10-fold cross-validation to assess performance
                  tuneGrid = expand.grid(alpha = candidate.alphas,    # weight attached to ridge penalty component
                                         lambda = candidate.lambdas))  # weight attached to overall penalty
enet.preds.out = predict(object = enet.mod, 
                         newdata = test.data)

# Support Vector Regression
candidate.costs = seq(from=0.1, to=3, length.out=5)
candidate.sigmas = seq(from=0.01, to=1, length.out=5)

# Note one of the underlying packages (kernlab) has issues with
# the way we had specified our model for other approaches.
# Here we'll use formula interface to training, where
# we don't give y as a vector and x as a matrix, but 
# instead specify our model (in symbolic form) and the
# dataset to use.
svm.mod = train(y ~ . ,
                data=train.data,                   # outcome vector
                preProcess = c("center","scale"),
                method = 'svmRadial',                   # do SVM with a radial kernel 
                trControl = trainControl(method = "cv", 
                                         number = 10),  # do 10-fold cross-validation to assess performance
                tuneGrid = expand.grid(sigma = candidate.sigmas, # width of radial kernel
                                       C = candidate.costs))     # prediction error penalty weight
svm.preds.out = predict(object = svm.mod, 
                        newdata = test.data)


# Random Forest
candidate.min.node.sizes = c(3,5,7)
vars.per.split = floor(sqrt(ncol(train.data)-1))
rf.mod = train(y = train.data$y,                      # outcome vector
                 x = train.data[,2:ncol(train.data)],                  # predictor matrix
                 method = 'ranger',                  # the ranger package does the work behind the scenes (for random forests)
                 trControl = trainControl(method = "cv", 
                                          number = 10), # do 10-fold cross-validation to assess performance
                 tuneGrid = expand.grid(mtry = vars.per.split,
                                        splitrule = "variance",
                                        min.node.size = candidate.min.node.sizes))  # tune over min node size setting alone
rf.preds.out = predict(object = rf.mod, 
                         newdata = test.data)


# Neural network
candidate.sizes = seq(from=3, to=7, by=1)
candidate.decays = c(0.05, 0.1)
nnet.mod = train(y = train.data$y,                      # outcome vector
                 x = train.data[,2:ncol(train.data)],                  # predictor matrix
                 method = 'nnet',                  # the glmnet package does the work behind the scenes (for nonlinear elastic nets)
                 preProcess = c("center", "scale"),  # center and scale predictors
                 trControl = trainControl(method = "cv", 
                                          number = 10), # do 10-fold cross-validation to assess performance
                 tuneGrid = expand.grid(size = candidate.sizes,    # number of nodes to use in hidden layer
                                        decay = candidate.decays), # weight attached to complexity penalty
                 linout = T)                                       # make the output layer linear (for regression) rather than logistic (for classification)
nnet.preds.out = predict(object = nnet.mod, 
                         newdata = test.data)

# Gradient boosting (with shallow trees as learners)

xgb.mod = train(y = train.data$y,                      # outcome vector
                x = train.data[,2:ncol(train.data)],                  # predictor matrix
                method = 'xgbTree',                  # eXtreme Gradient Boosting (with trees as learners)
                trControl = trainControl(method = "cv", 
                                           number = 10), # do 10-fold cross-validation to assess performance
                tuneGrid = expand.grid(nrounds=c(25, 50, 75),               # boosting iterations to use
                                       max_depth=c(3),              # maximum depth of each individual tree learner
                                       eta=c(0.3),                  # how much do we downscale (0-1) each boosting iteration's contribution?
                                       gamma=c(1),                  # stopping threshold for improvement in loss function 
                                       colsample_bytree=c(1),       # fraction of columns to consider for each split (we'll use all)
                                       min_child_weight=c(1),       # minimum leaf size in each tree 
                                       subsample=c(1))              # can use a fraction of the data if you want to reduce overfitting
                )  # weight attached to overall penalty
xgb.preds.out = predict(object = xgb.mod, 
                          newdata = test.data)

# Print out results of our little horse race:
all.preds = list("OLS" = lm.preds.out, 
                 "LASSO" = lasso.preds.out, 
                 "Ridge" = ridge.preds.out, 
                 "Elastic net" = enet.preds.out,
                 "SVM" = svm.preds.out,
                 "Random forest" = rf.preds.out,
                 "Neural net" = nnet.preds.out,
                 "xgboost" = xgb.preds.out)
for(i in 1:length(all.preds)) {
  print(paste0(names(all.preds)[i], ": ",
               RMSE(all.preds[[i]], test.data$y)))
}

# We could combine one or more of these approaches in an ensemble
# For example, let's combine LASSO and SVM

ens.list = caretList(y ~ .,
                     data=train.data,
                     preProcess = c("center", "scale"),
                     trControl = trainControl(method = "cv", 
                                              number = 10),
                     tuneList = list(lasso=caretModelSpec(method="glmnet",
                                                          tuneGrid = expand.grid(alpha = 1,    # weight attached to ridge penalty component
                                                                                 lambda = candidate.lambdas)  # weight attached to overall penalty
                                                          ),
                                     svm = caretModelSpec(method="svmRadial",
                                                          tuneGrid=expand.grid(sigma = candidate.sigmas, # width of radial kernel
                                                                               C = candidate.costs)
                                     )))


# Now can construct a simple linear combination
ens.mod = caretEnsemble(ens.list, 
                        trControl=trainControl(method = "cv", 
                                               number = 10))
# See some basics on in-sample performance and weights
summary(ens.mod)

# Predict out of sample for both the ensemble and individual
# model members
ens.preds.out = predict(ens.mod, newdata = test.data)
ens.preds.individual.out = predict(ens.list, newdata=test.data)

# Compare. Stick all predictions from individual model members
# and the overall ensemble into a matrix and then call the RMSE function on each column
apply(cbind(ens.preds.individual.out, "ensemble" = ens.preds.out),
      2,
      FUN = function(preds) RMSE(preds, test.data$y))

