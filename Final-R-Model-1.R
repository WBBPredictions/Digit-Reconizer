#library(caret)
#library(class)
library(FNN)
library(beepr)
library(readr)
library(liquidSVM)
library(ranger)
source("/Users/travisbarton/Desktop/Projects/Cross Val Maker.R")
Mini_train <- read_csv("~/Desktop/Projects/Group/Mini_train_temp.csv", col_types = cols(X1 = col_skip()))
dat <- Mini_train/255
dat[,1] <- dat[,1]*255
datt <- Cross_val_maker(dat, .2)
Train <- datt$Train
Train$label <- factor(Train$label)
Test <- datt$Test
Test$label <- factor(Test$label)

The_Deal <- fuction(Train, Test, point, model)
{
  s <- Sys.time()
  # Fit the SVM, KNN, Random Forest (RF)
  #fit.svm <- svm(factor(label) ~ ., data = Train, kernel = "linear", cross = 5, cost = 8, gamma = .5)
  fit.svm <- mcSVM(label ~ ., Train, mc_type = "AvA")
  pred.svm <- predict(fit.svm, Test[,-1])
  fit.knn <- knn(Train[, -1], Test[,-1], Train[, 1])
  rf <- ranger(label ~ ., data = Train, num.trees = 500, classification = T)
  pred.rf <- predict(rf, Test[,-1])
  #SVM
  sum(diag(table(pred.svm, Test$label)))/sum(table(pred.svm, Test$label))
  #KNN
  sum(diag(table(fit.knn, Test$label)))/sum(table(fit.knn, Test$label))
  #rf
  sum(diag(table(pred.rf$predictions, Test$label)))/sum(table(pred.rf$predictions, Test$label))
  
  
  
  
  
  
  
  
  (Sys.time() - s)
  #obj <- tune(svm, label ~ ., data = train, ranges = list(gamma = 2^(-1:1), cost = 2^(0:4)), tunecontrol = tune.control(sampling = "fix"))
  #summary(fit.svm)
  #tune.knn(train[, -1], y = factor(resp), k = 1:7)
  #beep()
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