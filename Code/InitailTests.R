## Loading required libraries
library(dplyr)
library(ggplot2)

## Loading the data
dat <- read.csv("../Data/fraudTrain.csv")

## Analysing the data

# number of frauds
n_frauds <- sum(dat$is_fraud == 1)
prop_frauds <- n_frauds / nrow(dat)
print(paste("Number of frauds: ", n_frauds))
print(paste("Proportion of frauds: ", prop_frauds*100 , "%"))
