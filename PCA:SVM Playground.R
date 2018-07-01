library(FNN)
library(ranger)
library(readr)
library(caret)
library(e1071)
setwd("~/Desktop/Projects/Group")
dattt <- read_csv("~/Desktop/Projects/Group/Mini_train.csv", col_types = cols(X1 = col_skip()))
A <- seq(0, 3, .5)
keeper <- matrix(NA, nrow = 3, ncol = 1)
kepper <- {}
row.names(keeper) = c("SVM", "KNN", "RF")
temp <- {}
temp2 <- {}
temp3 <- {}
#for(i in 1:length(A))
{
  temp <- {}
  temp2 <- {}
  temp3 <- {}
  s <- Sys.time()
  
  #for(j in 1:4)
    {
    
  

datt <- Cross_val_maker(dattt, .1)
Train <- datt$Train
Test <- datt$Test
index <- nearZeroVar(Train[,-1], freqCut=2000, uniqueCut= .25)
label <- Train[,1]

dat <- Train[,-1]
dat <- dat[,-index]
dat <- dat/255
cov_matrix <- cov(dat)

pca_data <- prcomp(cov_matrix)


PCA_var <- pca_data$sdev^2
Total_var <- sum(pca_data$sdev^2)

PCA_Digit <- data.frame("Cum_Var" = cumsum(PCA_var/Total_var), "Exact_Var" = PCA_var/Total_var)

plot(PCA_Digit$Cum_Var[1:28])
abline(.95, 0, col = "red")

# Lets do the first 20 var then, love.

new_data <- as.matrix(dat) %*% pca_data$rotation[,1:26]

new_data <- cbind(label, new_data)
Train <- as.data.frame(new_data)
test_label <- Test[,1]
Test <- Test[,-1]
Test <- Test[,-index]
Test <- Test/255

Test <- as.matrix(Test) %*% pca_data$rotation[,1:26]
Test <- cbind(test_label, Test)

fit.svm <- svm(label ~ ., Train, type = "C-classification", gamma = 1)
fit.rf <- ranger(label ~ ., Train, classification = T, mtry = 10, num.trees = 50)
pred.knn <- knn(Train[,-1], Test[,-1], Train[,1], k = 4)
pred.svm <- predict(fit.svm, Test[,-1])
pred.rf <- predict(fit.rf, Test[,-1])
temp <- c(temp, percent(table(pred.svm, Test[,1])))
temp2 <- c(temp2, percent(table(pred.knn, Test[,1])))
temp3 <- c(temp3, percent(table(pred.rf$predictions, Test[,1])))
}
print(Sys.time() - s)
#keeper <- cbind(keeper, 
#                c((mean(temp)), 
#                  (mean(temp2)), 
#                  (mean(temp3))))
keeper <- c(keeper, mean(temp))

}
beep()
beep()
beep()
