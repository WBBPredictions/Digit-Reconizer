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
Data_Reducer <- function(dat, alpha = .00005910579)
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
Data_Reducer(Mat, .01)

#creates reducded data
reduced_train <- Data_Reducer(train)


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
