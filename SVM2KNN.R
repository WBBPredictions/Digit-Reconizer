library(class)
library(readr)
library(Matrix)
library(e1071)
library(beepr)
#detach(e1071)
<<<<<<< HEAD
dat <- read_csv(file.choose(), col_types = cols(X1 = col_skip()))
datt <- Cross_val_maker(dat, .1)
Train <- datt$Train
Test <- datt$Test

svm.fit <- svm(Train$label~., data = Train, type = "C-classification", kernel = "linear")
summary(svm.fit)

pred <- predict(svm.fit, Test[,-1])
table(pred, Test[,1])

=======
dat <- read_csv("~/Desktop/Projects/Group/Mini_train.csv", col_types = cols(X1 = col_skip()))
# datt <- Cross_val_maker(dat, .1)
# Train <- datt$Train
# Test <- datt$Test
# 
# svm.fit <- svm(Train$label~., data = Train, type = "C-classification", kernel = "linear")
# summary(svm.fit)
# 
# pred <- predict(svm.fit, Test[,-1])
# tab <- table(pred, Test[,1])
# sum(diag(tab))/(sum(tab))
>>>>>>> 0d94983d69604309f72fa45a85a097c20085f8eb

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
The_Big_Test <- function(data, alpha)
{
 
  
  dat <- Cross_val_maker(data, alpha)
  Train <- dat$Train
  Test <- dat$Test
  
  svm.fit <- svm(Train$label~., data = Train, type = "C-classification", kernel = "linear")
 
  pred <- predict(svm.fit, Test[,-1])
  
  table(pred, Test[,1])
  
  Truth <- rep(0, 9)
  Mistake <- rep(0, 9)
  for(i in 1:length(pred))
  {
    if(pred[i] != Test[i, 1])
    {
      Mistake[pred[i]] = Mistake[pred[i]] + 1
      Truth[Test[i, 1]] = Truth[Test[i, 1]] + 1
    }
  }
  #list("Miss" = Mistake, "Truu" = Truth)
  return(table(pred, Test[,1]))

<<<<<<< HEAD
Buisc <- {}
for(i in 1:1000){
  
  Buisc[i] <- The_Big_Test(dat, .1)
}
=======
}
#Run overnight
s = Sys.time()
Buisc <- matrix(0, nrow = 10, ncol = 10)
for(i in 1:500)
{
  
  Buisc <- Buisc + (The_Big_Test(dat, .1))

}
beep()
s = Sys.time() - s
s
>>>>>>> 0d94983d69604309f72fa45a85a097c20085f8eb
