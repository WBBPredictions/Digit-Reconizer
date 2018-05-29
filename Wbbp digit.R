digits=read.csv(file.choose(),header=T)
nrow(digits) #42000
ncol(digits) #785

sumcol=apply(digits,2,sum)
plot(1:784,as.vector(sumcol)[-1])
