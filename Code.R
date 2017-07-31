####################################################
#
# R Code for the paper "Evaluation of Ensemble Methods in Imbalanced Domain Regression Tasks
#
####################################################


#Setting the working directory
setwd("")

#Loading the data
load("All22DataSets.RData")

#Libraries
library(xgboost)
library(UBL)
library(uba)
library(performanceEstimation)

#Function with evaluation metrics
eval.stats <- function(trues,preds,ph,ls) {
  
  prec <- util(preds,trues,ph,ls,util.control(umetric="P",event.thr=0.9))
  rec  <- util(preds,trues,ph,ls,util.control(umetric="R",event.thr=0.9))
  F1   <- util(preds,trues,ph,ls,util.control(umetric="Fm",beta=1,event.thr=0.9))
  
  mad=mean(abs(trues-preds))
  mse=mean((trues-preds)^2)
  mape= mean((abs(trues-preds)/trues))*100
  rmse= sqrt(mean((trues-preds)^2))
  mae_phi= mean(phi(trues,phi.parms=ph)*(abs(trues-preds)))
  mape_phi= mean(phi(trues,phi.parms=ph)*(abs(trues-preds)/trues))*100
  mse_phi= mean(phi(trues,phi.parms=ph)*(trues-preds)^2)
  rmse_phi= sqrt(mean(phi(trues,phi.parms=ph)*(trues-preds)^2))
  prec=prec
  rec=rec
  F1=F1
  
  c(
    mse=mse, rmse=rmse, prec=prec,rec=rec,F1=F1
  )
  
}

#Function for calculating the mean squared error for XGBOOST
mse <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(mean((labels-preds)^2))
  return(list(metric = "mse", value = err))
}

####################################################
#
# Workflows
#
####################################################

mc.rpart <- function(form,train,test,minsplit,cp,...) {
  
  require(rpart)
  
  tgt <- which(colnames(train)==as.character(form[[2]]))
  
  ph <- phi.control(train[,tgt], method="extremes")
  ls <- loss.control(train[,tgt])
  
  m <- rpart(form,train,control=rpart.control(minsplit=minsplit, cp=cp))
  p <- predict(m, test)
  
  eval <- eval.stats(test[,tgt],p,ph,ls)
  res <- list(evaluation=eval)
  res
  
}

mc.bagging <- function(form,train,test,minsplit,cp,nbags,...) {
  
  require(ipred)
  
  tgt <- which(colnames(train)==as.character(form[[2]]))
  
  ph <- phi.control(train[,tgt], method="extremes")
  ls <- loss.control(train[,tgt])
  
  m <- bagging(form, train, coob=TRUE, nbagg=nbags, control=rpart.control(minsplit=minsplit,cp=cp))
  p <- predict(m, test)
  
  eval <- eval.stats(test[,tgt],p,ph,ls)
  res <- list(evaluation=eval)
  res
  
}

mc.xgboost <- function(form,train,test,eta,cst,max_depth,nround,...) {
  
  require(xgboost)
  
  tgt <- which(colnames(train)==as.character(form[[2]]))
  
  train[,tgt] <- as.numeric(train[,tgt])
  test[,tgt] <- as.numeric(test[,tgt])
  
  ph <- phi.control(train[,tgt], method="extremes")
  ls <- loss.control(train[,tgt])
  
  m <- xgboost(data=xgb.DMatrix(data.matrix(train[,-tgt]), label=train[,tgt]),objective="reg:linear",eta=eta,feval=mse,colsample_bytree=cst,max_depth=max_depth,nthread=5,nrounds=nround,silent=1)
  p <- predict(m, xgb.DMatrix(data.matrix(test[,-tgt]), label=test[,tgt]))
  
  eval <- eval.stats(test[,tgt],p,ph,ls)
  res <- list(evaluation=eval)
  res
  
}

####################################################
#
# Running experiments
#
####################################################


d <- 1

exp <- performanceEstimation(PredTask(DSs[[d]]@formula,DSs[[d]]@data),
                             c(workflowVariants("mc.rpart",minsplit=c(20,50,100,200),cp=c(0.01,0.05,0.1)),
                               workflowVariants("mc.xgboost",eta=c(0.01,0.05,0.1), max_depth=c(5,10,15), nround=c(25,50,100,200,500), cst=c(seq(0.2,0.9,by=0.1))),
                               workflowVariants("mc.bagging",minsplit=c(20,50,100,200),cp=c(0.01,0.05,0.1),nbags=c(10,20,30,40,50))),
                             EstimationTask("totTime",method=CV(nReps = 2, nFolds=5))
)


####################################################
#
# Results table for each data set
#
####################################################

res <- c()
for(wf in 1:432) {
  res.aux <- c()
  for(i in 1:10) {
    res <- rbind(res,c(getIterationsInfo(exp,workflow=wf,task=1,it=i)$evaluation,wf=wf))
  }
  
}
res <- as.data.frame(res)
res["Model"] <- c(rep("rpart",12*10),rep("xgboost",360*10),rep("bagging",60*10))
res.aux <- res
res <- aggregate(res,by=list(res$Model,res$wf),FUN=mean)
res$Model <- NULL; res$wf <- NULL; colnames(res)[c(1,2)] <- c("Model","wf")

rpart.mse <- res[res$Model=="rpart",][which(res[res$Model=="rpart",]$mse == min(res[res$Model=="rpart",]$mse)),][1,]
rpart.f1 <- res[res$Model=="rpart",][which(res[res$Model=="rpart",]$F1 == max(res[res$Model=="rpart",]$F1)),][1,]

xgb.mse <- res[res$Model=="xgboost",][which(res[res$Model=="xgboost",]$mse == min(res[res$Model=="xgboost",]$mse)),][1,]
xgb.f1 <- res[res$Model=="xgboost",][which(res[res$Model=="xgboost",]$F1 == max(res[res$Model=="xgboost",]$F1)),][1,]

bagging.mse <- res[res$Model=="bagging",][which(res[res$Model=="bagging",]$mse == min(res[res$Model=="bagging",]$mse)),][1,]
bagging.f1 <- res[res$Model=="bagging",][which(res[res$Model=="bagging",]$F1 == max(res[res$Model=="bagging",]$F1)),][1,]

final.res <- rbind(rpart.mse,rpart.f1,xgb.mse,xgb.f1,bagging.mse,bagging.f1)

wilcox.test(res.aux[res.aux$Model=="rpart" & res.aux$wf==final.res[2,]$wf,]$mse,res.aux[res.aux$Model=="rpart" & res.aux$wf==final.res[1,]$wf,]$mse,paired=T)$p.value
wilcox.test(res.aux[res.aux$Model=="rpart" & res.aux$wf==final.res[2,]$wf,]$F1,res.aux[res.aux$Model=="rpart" & res.aux$wf==final.res[1,]$wf,]$F1,paired=T)$p.value

wilcox.test(res.aux[res.aux$Model=="xgboost" & res.aux$wf==final.res[4,]$wf,]$mse,res.aux[res.aux$Model=="xgboost" & res.aux$wf==final.res[3,]$wf,]$mse,paired=T)$p.value
wilcox.test(res.aux[res.aux$Model=="xgboost" & res.aux$wf==final.res[4,]$wf,]$F1,res.aux[res.aux$Model=="xgboost" & res.aux$wf==final.res[3,]$wf,]$F1,paired=T)$p.value

wilcox.test(res.aux[res.aux$Model=="bagging" & res.aux$wf==final.res[6,]$wf,]$mse,res.aux[res.aux$Model=="bagging" & res.aux$wf==final.res[5,]$wf,]$mse,paired=T)$p.value
wilcox.test(res.aux[res.aux$Model=="bagging" & res.aux$wf==final.res[6,]$wf,]$F1,res.aux[res.aux$Model=="bagging" & res.aux$wf==final.res[5,]$wf,]$F1,paired=T)$p.value

final.res$wf <- rep(c("mse","F1"),3)

####################################################
#
# Write CSV file with results
#
####################################################

#write.csv(final.res,file=paste0("ds",d,".csv"),row.names=FALSE)


####################################################
#
# Load results from all data sets
#
####################################################


ds1 <- read.csv("ds1.csv"); ds1["Name"] <- DSs[[1]]@name
ds2 <- read.csv("ds2.csv"); ds2["Name"] <- DSs[[2]]@name
ds3 <- read.csv("ds3.csv"); ds3["Name"] <- DSs[[3]]@name
ds4 <- read.csv("ds4.csv"); ds4["Name"] <- DSs[[4]]@name
ds5 <- read.csv("ds5.csv"); ds5["Name"] <- DSs[[5]]@name
ds6 <- read.csv("ds6.csv"); ds6["Name"] <- DSs[[6]]@name
ds7 <- read.csv("ds7.csv"); ds7["Name"] <- DSs[[7]]@name
ds8 <- read.csv("ds8.csv"); ds8["Name"] <- DSs[[8]]@name
ds9 <- read.csv("ds9.csv"); ds9["Name"] <- DSs[[9]]@name
ds10 <- read.csv("ds10.csv"); ds10["Name"] <- DSs[[10]]@name
ds11 <- read.csv("ds11.csv"); ds11["Name"] <- DSs[[11]]@name
ds12 <- read.csv("ds12.csv"); ds12["Name"] <- DSs[[12]]@name
ds13 <- read.csv("ds13.csv"); ds13["Name"] <- DSs[[13]]@name
ds14 <- read.csv("ds14.csv"); ds14["Name"] <- DSs[[14]]@name
ds15 <- read.csv("ds15.csv"); ds15["Name"] <- DSs[[15]]@name
ds16 <- read.csv("ds16.csv"); ds16["Name"] <- DSs[[16]]@name
ds17 <- read.csv("ds19.csv"); ds17["Name"] <- DSs[[19]]@name
ds18 <- read.csv("ds20.csv"); ds18["Name"] <- DSs[[20]]@name
ds19 <- read.csv("ds21.csv"); ds19["Name"] <- DSs[[21]]@name
ds20 <- read.csv("ds22.csv"); ds20["Name"] <- DSs[[22]]@name

tbl <- rbind(ds1,ds2,ds3,ds4,ds5,ds6,ds7,ds8,ds9,ds10,
             ds11,ds12,ds13,ds14,ds15,ds16,ds17,ds18,ds19,ds20)

####################################################
#
# Produce results table concerning mse and F1^phi metrics
#
####################################################


rpart.1.mse <- tbl[tbl$Model=="rpart" & tbl$wf=="mse",]$mse
rpart.1.f1 <- tbl[tbl$Model=="rpart" & tbl$wf=="mse",]$F1
rpart.2.mse <- tbl[tbl$Model=="rpart" & tbl$wf=="F1",]$mse
rpart.2.f1 <- tbl[tbl$Model=="rpart" & tbl$wf=="F1",]$F1
xgboost.1.mse <- tbl[tbl$Model=="xgboost" & tbl$wf=="mse",]$mse
xgboost.1.f1 <- tbl[tbl$Model=="xgboost" & tbl$wf=="mse",]$F1
xgboost.2.mse <- tbl[tbl$Model=="xgboost" & tbl$wf=="F1",]$mse
xgboost.2.f1 <- tbl[tbl$Model=="xgboost" & tbl$wf=="F1",]$F1
bagging.1.mse <- tbl[tbl$Model=="bagging" & tbl$wf=="mse",]$mse
bagging.1.f1 <- tbl[tbl$Model=="bagging" & tbl$wf=="mse",]$F1
bagging.2.mse <- tbl[tbl$Model=="bagging" & tbl$wf=="F1",]$mse
bagging.2.f1 <- tbl[tbl$Model=="bagging" & tbl$wf=="F1",]$F1

final.tbl <- data.frame(Dataset=unique(tbl$Name),mse=rpart.1.mse,F1=rpart.1.f1,mse=rpart.2.mse,F1=rpart.2.f1,mse=xgboost.1.mse,F1=xgboost.1.f1,mse=xgboost.2.mse,F1=xgboost.2.f1,mse=bagging.1.mse,F1=bagging.1.f1,mse=bagging.2.mse,F1=bagging.2.f1)
final.tbl[,2:ncol(final.tbl)] <- round(final.tbl[,2:ncol(final.tbl)],3)
final.tbl <- final.tbl[match(c("a3","a6","a4","a7","abalone","a1","boston","a5","available_power","a2","cpu_small","heat","fuel_consumption_country","maximal_torque","delta_elevators","bank8FM","delta_ailerons","acceleration","concreteStrength","airfoild"),final.tbl$Dataset),]

xtable(final.tbl)


####################################################
#
# Produce results table for prec^phi and rec^phi metrics
#
####################################################


rpart.1.mse <- tbl[tbl$Model=="rpart" & tbl$wf=="mse",]$prec
rpart.1.f1 <- tbl[tbl$Model=="rpart" & tbl$wf=="mse",]$rec
rpart.2.mse <- tbl[tbl$Model=="rpart" & tbl$wf=="F1",]$prec
rpart.2.f1 <- tbl[tbl$Model=="rpart" & tbl$wf=="F1",]$rec
xgboost.1.mse <- tbl[tbl$Model=="xgboost" & tbl$wf=="mse",]$prec
xgboost.1.f1 <- tbl[tbl$Model=="xgboost" & tbl$wf=="mse",]$rec
xgboost.2.mse <- tbl[tbl$Model=="xgboost" & tbl$wf=="F1",]$prec
xgboost.2.f1 <- tbl[tbl$Model=="xgboost" & tbl$wf=="F1",]$rec
bagging.1.mse <- tbl[tbl$Model=="bagging" & tbl$wf=="mse",]$prec
bagging.1.f1 <- tbl[tbl$Model=="bagging" & tbl$wf=="mse",]$rec
bagging.2.mse <- tbl[tbl$Model=="bagging" & tbl$wf=="F1",]$prec
bagging.2.f1 <- tbl[tbl$Model=="bagging" & tbl$wf=="F1",]$rec

final.tbl <- data.frame(Dataset=unique(tbl$Name),mse=rpart.1.mse,F1=rpart.1.f1,mse=rpart.2.mse,F1=rpart.2.f1,mse=xgboost.1.mse,F1=xgboost.1.f1,mse=xgboost.2.mse,F1=xgboost.2.f1,mse=bagging.1.mse,F1=bagging.1.f1,mse=bagging.2.mse,F1=bagging.2.f1)
final.tbl[,2:ncol(final.tbl)] <- round(final.tbl[,2:ncol(final.tbl)],3)
final.tbl <- final.tbl[match(c("a3","a6","a4","a7","abalone","a1","boston","a5","available_power","a2","cpu_small","heat","fuel_consumption_country","maximal_torque","delta_elevators","bank8FM","delta_ailerons","acceleration","concreteStrength","airfoild"),final.tbl$Dataset),]

xtable(final.tbl,digits=3)


####################################################
#
# Plot critical differences
#
####################################################

rownames(final.tbl) <- final.tbl$Dataset
final.tbl$Dataset <- NULL

final.tbl.mse <- final.tbl[,c(1,3,5,7,9,11)]
colnames(final.tbl.mse) <- c("rpart.v1","rpart.v2","xgboost.v1","xgboost.v2","bagging.v1","bagging.v2")

final.tbl.f1 <- final.tbl[,c(2,4,6,8,10,12)]
colnames(final.tbl.f1) <- c("rpart.v1","rpart.v2","xgboost.v1","xgboost.v2","bagging.v1","bagging.v2")

plotCD(final.tbl.mse,decreasing=FALSE)
plotCD(final.tbl.f1)

## build the PDFs
pdf("cd_mse.pdf",width=5,height=4)
par(mfrow=c(1,1))
plotCD(final.tbl.mse,decreasing=FALSE)
dev.off()

pdf("cd_f1.pdf",width=5,height=4)
par(mfrow=c(1,1))
plotCD(final.tbl.f1)
dev.off()


####################################################
#
# Results table for each data set
#
####################################################

#Ordered by rare cases percentage
final.tbl <- final.tbl[match(c("a3","a6","a4","a7","abalone","a1","boston","a5","available_power","a2","cpu_small","heat","fuel_consumption_country","maximal_torque","delta_elevators","bank8FM","delta_ailerons","acceleration","concreteStrength","airfoild"),final.tbl$Dataset),]

#Ordered by size
# final.tbl <- final.tbl[match(c("delta_elevators","cpu_small","heat","delta_ailerons","bank8FM","abalone","available_power","maximal_torque","fuel_consumption_country","acceleration","airfoild","concreteStrength","boston","a3","a6","a4","a7","a1","a5","a2"),final.tbl$Dataset),]

err.tbl <- data.frame(rpart.mse=final.tbl$mse.1/final.tbl$mse,
                      xgboost.mse=final.tbl$mse.3/final.tbl$mse.2,
                      bagging.mse=final.tbl$mse.5/final.tbl$mse.4)

err.tbl <- err.tbl-1
err.tbl <- err.tbl*100
err.tbl[is.na(err.tbl)] <- 0
err.tbl[err.tbl==Inf] <- 0
err.tbl <- round(err.tbl,digits=3)

final.err <- data.frame(Dataset=rep(1:20,3),Metric=rep("mse",60),Model=c(rep("rpart",20),rep("xgboost",20),rep("bagging",20)),
                        Score=c(err.tbl[,1],err.tbl[,2],err.tbl[,3]))

difmse.rare <- ggplot(final.err,aes(x=Dataset,y=Score,group=Model,linetype=Model,colour=Model)) + geom_line() + ylim(0,100) + ggtitle("Order by % Rare") + xlab("") + ylab( "% Dif.  mse" )
# difmse.size <- ggplot(final.err,aes(x=Dataset,y=Score,group=Model,linetype=Model,colour=Model)) + geom_line() + ylim(0,100) + ggtitle("Order by Size") + xlab("") + ylab("")

##

err.tbl <- data.frame(rpart.F1=final.tbl$F1.1-final.tbl$F1,
                      xgboost.F1=final.tbl$F1.3-final.tbl$F1.2,
                      bagging.F1=final.tbl$F1.5-final.tbl$F1.4)

err.tbl[is.na(err.tbl)] <- 0
err.tbl[err.tbl==Inf] <- 0
err.tbl <- round(err.tbl,digits=3)

final.err <- data.frame(Dataset=rep(1:20,3),Metric=rep("F1",60),Model=c(rep("rpart",20),rep("xgboost",20),rep("bagging",20)),
                        Score=c(err.tbl[,1],err.tbl[,2],err.tbl[,3]))

diff1.rare <- ggplot(final.err,aes(x=Dataset,y=Score,group=Model,linetype=Model,colour=Model)) + geom_line() + ylim(0,1) + xlab("Dataset") + ylab( bquote('Dif.  F'[1]^phi) )
# diff1.size <- ggplot(final.err,aes(x=Dataset,y=Score,group=Model,linetype=Model,colour=Model)) + geom_line() + ylim(0,1) + xlab("Dataset") + ylab( "" )

## Build the PDFs
pdf("rare_n_size.pdf",width=12,height=6)
grid.newpage()
pushViewport(viewport(layout=grid.layout(2,2)))
print(difmse.rare,vp=viewport(layout.pos.row=1,layout.pos.col=1))
print(difmse.size,vp=viewport(layout.pos.row=1,layout.pos.col=2))
print(diff1.rare,vp=viewport(layout.pos.row=2,layout.pos.col=1))
print(diff1.size,vp=viewport(layout.pos.row=2,layout.pos.col=2))
dev.off()




