digits=read.csv(file.choose(),header=T)
nrow(digits) #42000
ncol(digits) #785
mtrain <- as.matrix(digits)
mtrain <- as.matrix(c(mapply(OneOrZero, mtrain)))
mtrain <- matrix(mtrain, nrow = nrow(digits), ncol = ncol(digits), byrow = FALSE)

summtrain=apply(mtrain,2,sum)
plot(1:784,as.vector(summtrain)[-1])

sumcol=apply(digits,2,sum)
plot(1:784,as.vector(sumcol)[-1])

