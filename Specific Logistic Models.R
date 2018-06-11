# 9 logistic models
library(readr)
dat <- read_csv("~/Desktop/Projects/Group/Mini_train.csv", col_types = cols(X1 = col_skip()))
DataMod <- function(data, number)
{
  for(i in 1:nrow(data))
  {
    if(data[i,1] == number)
    {
      data[i,1] = 1
    }
    else
    {
      data[i, 1] = 0
    }
  }
  return(data)
}
dat0 <- DataMod(dat, 0)
dat1 <- DataMod(dat, 1)
dat2 <- DataMod(dat, 2)
dat3 <- DataMod(dat, 3)
dat4 <- DataMod(dat, 4)
dat5 <- DataMod(dat, 5)
dat6 <- DataMod(dat, 6)
dat7 <- DataMod(dat, 7)
dat8 <- DataMod(dat, 8)
dat9 <- DataMod(dat, 9)
logitfit0 <- glm(label~ ., data = dat0, family = binomial("logit"))
logitfit1 <- glm(label~ ., data = dat1, family = binomial("logit"))
logitfit2 <- glm(label~ ., data = dat2, family = binomial("logit"))
logitfit3 <- glm(label~ ., data = dat3, family = binomial("logit"))
logitfit4 <- glm(label~ ., data = dat4, family = binomial("logit"))
logitfit5 <- glm(label~ ., data = dat5, family = binomial("logit"))
logitfit6 <- glm(label~ ., data = dat6, family = binomial("logit"))
logitfit7 <- glm(label~ ., data = dat7, family = binomial("logit"))
logitfit8 <- glm(label~ ., data = dat8, family = binomial("logit"))
logitfit9 <- glm(label~ ., data = dat9, family = binomial("logit"))
beep()



