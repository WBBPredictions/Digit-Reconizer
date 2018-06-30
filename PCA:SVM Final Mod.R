library(caret)
library(readr)
library(fortunes)
library(e1071)
Train <- read.csv("train.csv")
Test <- read_csv("test.csv")
index <- nearZeroVar(Train[,-1], freqCut=1000, uniqueCut= .05)
label <- Train[,1]

dat <- Train[,-1]
dat <- dat[,-index]
dat <- dat/255
cov_matrix <- cov(dat)

pca_data <- prcomp(cov_matrix)

PCA_var <- pca_data$sdev^2
Total_var <- sum(pca_data$sdev^2)

PCA_Digit <- data.frame("Cum_Var" = cumsum(PCA_var/Total_var), "Exact_Var" = PCA_var/Total_var)

# Lets do the first 20 var then, love.

new_data <- as.matrix(dat) %*% pca_data$rotation[,1:20]

new_data <- cbind(label, new_data)
Train <- as.data.frame(new_data)
Test <- Test[,-index]
Test <- Test/255

Test <- as.matrix(Test) %*% pca_data$rotation[,1:20]

fit.svm <- svm(label ~ ., Train, type = "C-classification")
pred.svm <- predict(fit.svm, Test)
fortune()
final_data <- data.frame("ImageId" = as.integer(seq(1, 28000, 1)), "Label" = as.integer(pred.svm) - 1)
View(final_data)
write_csv(final_data, path = "new_output2.csv")
