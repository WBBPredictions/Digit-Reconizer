library(readr)
library(magrittr)
train <- read_csv("~/Desktop/Projects/Group/train.csv")
OneOrZero <- function(datt)
{
  if(datt == 0) 
    {return(0)}
  else 
    {return(1)}
}


# 1 way to do it
Data_Reducer_1 <- function(dat, alpha = .00005910579)
{
  # The default alpha comes from "How I chose Alpha" (Below)
  #### Data Prep (it takes a while)
  mtrain <- as.matrix(dat)
  mtrain <- as.matrix(c(mapply(OneOrZero, mtrain)))
  mtrain <- matrix(mtrain, nrow = nrow(dat), ncol = ncol(dat), byrow = FALSE)

  total <- sum(apply(mtrain, 2, sum))
  
  index <- numeric()
  for(i in 1:ncol(mtrain))
  {
    holder <- sum(mtrain[,i])/total
    if(holder < alpha)
    {
      index <- c(index, i)
    }
  }
  
  return(dat[,-c(index)])
  
}
#Example:
Mat <- matrix(seq(0, 9), nrow = 10, ncol = 10, byrow = T)
Data_Reducer_1(Mat, .01)


# Another way to do it
Data_Reducer_2 <- function(dat, alpha = .05)
{
  # The default alpha comes from "How I chose Alpha 2" (Below)
  #### Data Prep (it takes a while)
  mtrain <- as.matrix(dat)
  mtrain <- as.matrix(c(mapply(OneOrZero, mtrain)))
  mtrain <- matrix(mtrain, nrow = nrow(dat), ncol = ncol(dat), byrow = FALSE)
  
  total <- sum(apply(mtrain, 2, sum))
  
  index <- numeric()
  holder <- numeric()
  for(i in 1:ncol(mtrain))
  {
    holder[i] <- sum(mtrain[,i])/total
  }
  holder_index <- order(holder, decreasing = F)
  holder_2 <- 0
  
  for(i in 1:length(holder))
  {
    holder_2 = holder_2 + holder[holder_index[i]]
    if(holder_2 < alpha)
    {
      index <- c(index, holder_index[i])
    }
    else
    {
      return(dat[,-c(index)])
    }
  }
  
  
  
}
#Example:
Mat <- matrix(c(1), nrow = 10, ncol = 10, byrow = T)
diag(Mat) = 0
Mat[9, 10] = 0
Mat[3, 2] = 0
Mat[5, 3] = 0
Data_Reducer_2(Mat, .5)


#creates reducded data
reduced_train <- Data_Reducer_1(train)
reduced_train_2 <- Data_Reducer_2(train)
ncol(reduced_train)
ncol(reduced_train_2)
#Positives 
## Reduced more data
## Allows for standard alpha

#Negatives:
## time expensive
## Does not give preference for ties (just removed in added order)


# How I Chose Alpha:
## When I look at the different columns of ddtrain, I can see that theres a cliff
## when it comes to what percent of 1s each columns has in reference to the total number of 1s
## To keep from cluttering up your name spaces I turned the example into a function for easy running:
Example_function <- function()
{
  holder <- rep(0, ncol(train))
  for(i in 1:ncol(train))
  {
    holder[i] = sum(ddtrain[,i])
    
    
  }
  total <- sum(holder)
  plot(sort(holder), ylab = "Sorted column totals")
  abline(375, 0, col = "red")
}
# The question is, where is that red line (it looks better zoomed)? 
# Its at column total = 375.
# Now that percent is 375/total = 375/6344556 = .00005910579 .
# And thats how I got alpha's default
