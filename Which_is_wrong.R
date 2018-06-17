library(readr)
train_temp <- read_csv("~/Desktop/Projects/Group/train.csv")
Cross_val_maker <- function(data, alpha)
{
  if(alpha > 1 || alpha <= 0)
  {
    return("Alpha must be between 0 and 1")
  }
  index <- sample(c(1:nrow(data)), round(nrow(data)*alpha))
  train <- data[-index,]
  test <- data[index,]
  return(list("Train" = as.data.frame(train), "Test" = as.data.frame(test)))
}
set.seed(69)
dat <- Cross_val_maker(train_temp, .1)
Train <- dat$Train
Test <- dat$Test
which_is_wrong_DR <- function(Train, Test)
{
  s <- Sys.time()
  fit.svm <- svm(label ~ ., data = Train, type = "C-classification", kernel = "linear", cross = 5, cost = 8, gamma = .5)
  pred.svm <- predict(fit.svm, Test[,-1])
  fit.knn <- knn(Train[, -1], Test[,-1], Train[, 1])
  rf <- randomForest(factor(label) ~ ., data = Train, ntree = 500)
  pred.rf <- predict(rf, Test[,-1])
  svm.table <- table(pred.svm, Test[, 1])
  KNN.table <- table(fit.knn, Test[,1])
  rf.table <- table(pred.rf, Test[,1])
  
  diag(svm.table) = 0
  diag(KNN.table) = 0
  diag(rf.table) = 0
  svm.table <- svm.table/sum(svm.table)
  KNN.table <- KNN.table/sum(KNN.table)
  rf.table <- rf.table/sum(rf.table)
  
  return(list(svm = svm.table, knn = KNN.table, rf = rf.table))
  
  
}
tab <- which_is_wrong_DR(Train, Test)
tab
