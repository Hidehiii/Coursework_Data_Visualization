library(car)
library(ggplot2)
data<-read.csv("数据2.csv",header=FALSE,sep=",")
View(data)
fit2<-glm(V11~V2+V3+V4+V5+V6+V7+V8+V9+V10,data=data.frame(data),family = binomial())
summary(fit2)
fit<-lm(V11~V2+V3+V4+V5+V6+V7+V8+V9+V10,data=data.frame(data))
fit
boxplot.stats(data)
qqPlot(fit,labels=rownames(data),id.method="identity",envelope=0.89)
outer<-outlierTest(fit)
outer$p
install.packages("stats")
library("car")
help("qqplot")
pairs(data[,1:11])
year2024=predict(fit,newdata = data.frame(data[1:20,]))###预测
year2024
i=1
data
year=rep(2023:2001)

lm=lm(data[,2]~year)
abline(lm)
pre=predict(lm,newdata = data.frame(year=2024))
pre
pre=c(2024,0,0,0,0,0,0,0,0,0,0)
pre[1]
for (i in 2:10) {
  lm=lm(data[,i]~year)
  pre[i]=predict(lm,newdata = data.frame(year=2024))
}
pre
year2021=predict(fit,newdata = data.frame(data[24,]))
data[24,]=pre

data<-read.csv("数据2.csv",header=FALSE,sep=",")#读取数据
data
pairs(data[,1:11])##pairs方法查看各指标与人口老龄化比例的相关性
data=data[,-3:-5]##去除无关指标
data
fit<-lm(V11~V2+V6+V7+V8+V9+V10,data=data.frame(data))#建立线性回归模型
result=predict(fit,newdata = data.frame(data[,-8]))#对原数据进行预测
result
sum=0
for (i in 1:23) {
  err_rate=(data[i,8]-result[i])/data[i,8]
  sum=sum+err_rate
}
avg=sum/23
avg##预测结果评价误差率
library("car")
qqPlot(fit,labels=rownames(data),id.method="identity",envelope=0.89)##基于线性回归的离群点检测
data2024=c(2024,0,0,0,0,0,0,0)
year=rep(2023:2001)
for (i in 2:7) {
  lm=lm(data[,i]~year)##基于线性回归的各指标2024年数据预测
  data2024[i]=predict(lm,newdata = data.frame(year=2024))
}
data[24,]=data2024
data[24,8]=predict(fit,newdata = data.frame(data[24,-8]))##对预测的2024年数据进行基于fisher线性判别法的预测
data[24,8]
data2024
#-----------------------随机森林算法---------------------------
library("randomForest")
set.seed(1200)
rfmode <- randomForest(V11 ~ V2 + V6 + V7 + V8 + V9 + V10, 
                       data = data.frame(data),
                       importance = TRUE, 
                       proximity = TRUE)

plot(rfmode)
rfmode <- randomForest(V11 ~ V2 + V6 + V7 + V8 + V9 + V10, 
                       data = data.frame(data),
                       ntree=420,
                       importance = TRUE, 
                       proximity = TRUE)
rfmode
importance(rfmode)
varImpPlot(rfmode)
result2<-predict(rfmode,newdata = data.frame(data[,-8]))
result2
sum=0
for (i in 1:23) {
  err_rate=(data[i,8]-result2[i])/data[i,8]
  sum=sum+err_rate
}
avg=sum/23
avg

