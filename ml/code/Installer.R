# Installer script for packages used in 
# machine learning in economics workshop
options(install.packages.check.source = "no")

install.packages(c("data.table",
                   "haven",
                   "caret",
                   "caretEnsemble",
                   "xgboost",
                   "hdm",
                   "ArCo",
                   "grf",
                   "keras",
                   "tm",
                   "topicmodels",
                   "devtools"),
                 type="binary")
library(devtools)
install_github("susanathey/MCPanel",
               upgrade="always",
               type="binary")

# We'll install DoubleML separately using
# updated source packages as the binary versions
# seem a bit out of sync and install fails :/
install.packages("DoubleML", type="source")

