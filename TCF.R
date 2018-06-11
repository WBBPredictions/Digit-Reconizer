library(readr)
results <- read_csv("~/Desktop/Projects/Group/results.csv", col_types = cols(X1 = col_skip()))
SVM_TCF_WDiag <- function(x, results)
{
  mat <- matrix(0, nrow = 10, ncol = 10)
  suum <- sum(results)
  for(i in 1:10)
  {
    for(j in 1:10)
    {
      mat[i, j] <- results[i,j]/suum
    }
       
  }
  return(mat)
}
SVM_TCF_WDiag(3, results = results)
  
