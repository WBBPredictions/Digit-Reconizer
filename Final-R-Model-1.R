library(caret)
library(class)
library(beepr)
library(readr)
library(e1071)
Mini_train <- read_csv("~/Desktop/Projects/Group/Mini_train.csv", col_types = cols(X1 = col_skip()))
train <- Mini_train
resp <- Mini_train$label
preds <- Mini_train[,-1]

The_Deal <- fuction(train, point, model)
{
  # Fit the SVM, KNN, Random Forest (RF)
  fit.svm <- svm(label ~ ., data = train, type = "C-classification", kernel = "linear", cross = 5)
  obj <- tune(svm, label ~ ., data = train, ranges = list(gamma = 2^(-1:1), cost = 2^(0:4)), tunecontrol = tune.control(sampling = "fix"))
  summary(fit.svm)
  # Pull certainty from SVM, KNN and RF
  
  # Use those decisions to fit a logistic model (trust or not trust)
    # if not trust: do highest prob of the logistics
    # if trust... then you good!
}

#the trust model: create with following function:

Trust_Me <- function(train)
{
  # Use Cross Val Maker to subset the data
  # train SVM, KNN and RF with Train
  # Find certainties for SVM, KNN, RF
  # cbind, Test[,1] with certainties (Call that Thomas)
  # Fit the Trust_Me model with Thomas the data engine.
  # return(Trust_Me)
}