library(readr)
#Put your path here
FullData <- read_csv(file.choose())
###### IMPORTANT NOTE ###########
#Once you load 'dat' the first time, comment out the above line
#It only needs to be loaded in once into your global enviornment


<<<<<<< HEAD
#Make sure to set your working directory to yhe github page
#setwd("C:\Users\terri\OneDrive\Documents\Data Competitions\Digit Recognizer\test.csv")
=======
#Make sure to set your working directory to the github page
setwd("~/Desktop/Projects/Group")
>>>>>>> 0d94983d69604309f72fa45a85a097c20085f8eb

subsetter <- function(data, size, seed = NULL)
{
  if(size <= 0 || size > 1)
  {
    return("Size must be between 0 and 1")
  }
  if(is.null(seed) == T)
  {
    index <-sample(seq(1, nrow(data)), round(size*nrow(data)), replace = F)
    newdata <- data[index, ]
    return(newdata)
  }
  else
  {
    set.seed(seed = seed)
    index <-sample(seq(1, nrow(data)), round(size*nrow(data)), replace = F)
    newdata <- data[index, ]
    return(newdata)
  }
}

#Change size/seed for different results.
#My default seed will be 69... because I'm a child
newdat <- subsetter(FullData, .4, seed = 69)

#this gets rid of the index column (which tells what rows got selected)
#newdat <- newdat[,-1]
write.csv(newdat, file = "Mini_train.csv")

