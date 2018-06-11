The_Deal <- fuction(train, point, model)
{
  # Fit the SVM, KNN, Random Forest (RF)
  
  # Pull certainty from SVM, KNN and RF
  
  # Use those decisions to fit a logistic model (trust or not trust)
    # if not trust: do highest prob of the logistics
    # if trust... then you good!
}

#the trust model: create with following function:

Trust_Me <- function(train)
{
  # Use Cross Val Maker to subset the data
  # train SVM, KNN and RF with Train
  # Find certainties for SVM, KNN, RF
  # cbind, Test[,1] with certainties (Call that Thomas)
  # Fit the Trust_Me model with Thomas the data engine.
  # return(Trust_Me)
}