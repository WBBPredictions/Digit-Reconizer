library(readr)
train <- read_csv("~/Desktop/Projects/Group/train.csv")
OneOrZero <- function(datt)
{
  if(datt == 0)
  {
    return(0)
  }
  else
  {
    return(1)
  }
}
mtrain <- as.matrix(train)
dtrain <- as.matrix(c(mapply(OneOrZero, mtrain)))
ddtrain <- matrix(dtrain, nrow = nrow(mtrain), ncol = ncol(mtrain), byrow = FALSE)
holder <- rep(0, ncol(train))
for(i in 1:ncol(train))
{
  holder[i] = sum(ddtrain[,i])/42000
}
which(holder < .05)
