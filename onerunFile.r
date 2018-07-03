
setwd("~/GitHub/Digit-Reconizer")

master_Train <- read.csv("train.csv")
master_Test <- read.csv("test.csv")


###############################################################################


library(caret)
library(readr)
library(fortunes)
library(e1071)
library(FNN)
library(ranger)
library(randomForest)
library(beepr)


Train <- master_Train
Test <- master_Test

label <- Train[,1]

temp <- rbind(Train[,-1], Test)
blanks <- nearZeroVar(temp,freqCut=1000, uniqueCut= .05)


train.raw <- Train[,-1]/255
train.raw <- train.raw[,-blanks]
train.raw <- cbind(label, train.raw)

test.raw <- Test[,-blanks]/255


index <- nearZeroVar(Train[,-1], freqCut=1000, uniqueCut= .05)

dat <- Train[,-1]
dat <- dat[,-index]
dat <- dat/255
cov_matrix <- cov(dat)

pca_data <- prcomp(cov_matrix)

PCA_var <- pca_data$sdev^2
Total_var <- sum(pca_data$sdev^2)

PCA_Digit <- data.frame("Cum_Var" = cumsum(PCA_var/Total_var), "Exact_Var" = PCA_var/Total_var)


new_data <- as.matrix(dat) %*% pca_data$rotation[,1:20]

new_data <- cbind(label, new_data)
Train.pca <- as.data.frame(new_data)
Test.pca <- Test[,-index]
Test.pca <- Test.pca/255

Test.pca <- as.matrix(Test.pca) %*% pca_data$rotation[,1:20]

Becker_Barton_Device <- function(TTrain, TTest, Mtry = 10, numtree = 100, Gamma = 1/ncol(TTrain), Cost = .01, svm.kernel = 'radial', K = 5)
{
  run_time <- list("SVM" = {}, "RF" = {}, "KNN" = {}, "Final_Selection" = {})
  label.raw <- TTrain[,1]
  TTrain <- cbind(label.raw, TTrain[,-1])
  
  s <- Sys.time()
 
  fit.svm <- svm(label.raw ~ ., TTrain, type = "C-classification", cost = Cost, kernel = svm.kernel)
  pred.svm <- predict(fit.svm, TTest)
  run_time$SVM <- Sys.time() - s
  s <- Sys.time()
  
  
  fit.rf <- randomForest(factor(label.raw) ~ ., TTrain, mtry = Mtry, ntree = numtree)
  pred.rf <- predict(fit.rf, TTest)
  run_time$RF = Sys.time() - s
  pred.rf.prob <- predict(fit.rf, TTest, type = 'prob')
  pred.rf.prob <- apply(pred.rf.prob, 1, max)
  
  s <- Sys.time()
  pred.knn <- knn(train = TTrain[,-1], test = TTest, cl = TTrain[,1], k = K, prob = T)
  run_time$KNN <- Sys.time() - s
  
  pred.knn.prob <- attr(pred.knn, 'prob')
  
  
  for(i in 1:length(pred.knn.prob))
  {
    if(pred.knn.prob[i] > .65)
    {
      run_time$Final_Selection[i] = as.numeric(pred.knn[i]) - 1
    }
    else if(pred.rf.prob[i] > .6)
    {
      run_time$Final_Selection[i] = as.numeric(pred.rf[i]) - 1
    }
    else
    {
      run_time$Final_Selection[i] = as.numeric(pred.svm[i]) - 1
    }
  }
  beep()
  return(run_time)
}



s <- Sys.time()
pred.BBD <- Becker_Barton_Device(train.raw, test.raw, svm.kernel = "linear")
s <- Sys.time() - s
(paste("BBD Without PCA took: ",s))

s <- Sys.time()
pred.BBD.PCA <- Becker_Barton_Device(Train.pca, Test.pca)
s <- Sys.time() - s
(paste("BBD with PCA took: ",s))
beep()