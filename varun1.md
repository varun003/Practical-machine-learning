---
title: "Practical Machine Learning"
author: "Varun Negi"
date: "29 April 2017"
output: html_document
---
# Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.


# Getting data


```r
setwd("F:/ANALYTICS DATA/R/DATA MANIPULATION/practical machine learning")

library(readr)

train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")

table(train$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
table(train$user_name)
```

```
## 
##   adelmo carlitos  charles   eurico   jeremy    pedro 
##     3892     3112     3536     3070     3402     2610
```

```r
table(train$user_name,train$classe)
```

```
##           
##               A    B    C    D    E
##   adelmo   1165  776  750  515  686
##   carlitos  834  690  493  486  609
##   charles   899  745  539  642  711
##   eurico    865  592  489  582  542
##   jeremy   1177  489  652  522  562
##   pedro     640  505  499  469  497
```

```r
prop.table(table(train$user_name,train$classe),1)
```

```
##           
##                    A         B         C         D         E
##   adelmo   0.2993320 0.1993834 0.1927030 0.1323227 0.1762590
##   carlitos 0.2679949 0.2217224 0.1584190 0.1561697 0.1956941
##   charles  0.2542421 0.2106900 0.1524321 0.1815611 0.2010747
##   eurico   0.2817590 0.1928339 0.1592834 0.1895765 0.1765472
##   jeremy   0.3459730 0.1437390 0.1916520 0.1534392 0.1651969
##   pedro    0.2452107 0.1934866 0.1911877 0.1796935 0.1904215
```

# Cleaning data

```r
#Doing some basic  some basic data clean-up by removing columns 1 to 6, 
#which are there just for information and reference purposes:

train <- train[,7:160]
test <- test[,7:160]

# and removing all columns that mostly contain NA's

is_data  <- apply(!is.na(train), 2, sum) > 19621  # which is the number of observations
train <- train[, is_data]
test <- test[, is_data]
```


# Data Partitioning 

```r
#Before we can move forward with data analysis, we split the training set into
#two for cross validation purposes. We randomly
#subsample 60% of the set for
#training purposes (actual model building), while the 40% remainder will be used 
#only for testing, evaluation and accuracy measurement.



library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
set.seed(3141592)
inTrain <- createDataPartition(y=train$classe, p=0.60, list=FALSE)
train1  <- train[inTrain,]
train2  <- train[-inTrain,]
dim(train1)
```

```
## [1] 11776    87
```

# Removing non-zero variables from the dataset


```r
nzv_cols <- nearZeroVar(train1)
if(length(nzv_cols) > 0) {
train1 <- train1[, -nzv_cols]
train2 <- train2[, -nzv_cols]
}
dim(train1)
```

```
## [1] 11776    54
```

# Data Manipulation

```r
#lets look at the relative importance of the variables

library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
set.seed(3141592)
fitModel <- randomForest(classe~., data=train1, importance=TRUE, ntree=100)
varImpPlot(fitModel)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png)

```r
#Using the Accuracy and Gini graphs above, we select the top 10 variables that we'll use for model
#building. If the accuracy of the resulting model is acceptable, limiting the number of variables 
#is a good idea to ensure readability and interpretability of the model. 

#A model with 10 parameters is certainly much more user friendly than a model with 53 parameters.

#Our 10 covariates are: yaw_belt, roll_belt, num_window, pitch_belt, magnet_dumbbell_y, 
#magnet_dumbbell_z, pitch_forearm, accel_dumbbell_y, roll_arm, and roll_forearm.

#Let's analyze the correlations between these 10 variables. The following code calculates 
#the correlation matrix, replaces the 1s in the diagonal with 0s, and outputs which variables
#have an absolute value correlation above 75%:
```

# Finding Correlation between variables


```r
correl = cor(train1[,c("yaw_belt","roll_belt","num_window","pitch_belt","magnet_dumbbell_z","magnet_dumbbell_y","pitch_forearm","accel_dumbbell_y","roll_arm","roll_forearm")])
diag(correl) <- 0
which(abs(correl)>0.75, arr.ind=TRUE)
```

```
##           row col
## roll_belt   2   1
## yaw_belt    1   2
```

```r
#So we may have a problem with roll_belt and yaw_belt which have a high correlation (above 75%) with each other:

cor(train1$roll_belt, train1$yaw_belt)
```

```
## [1] 0.8152349
```

We can identify an interesting relationship between roll_belt and magnet_dumbbell_y:

```r
qplot(roll_belt, magnet_dumbbell_y, colour=classe, data=train1)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-1.png)

```r
# Incidentally, a quick tree classifier selects roll_belt as the first discriminant among all 53 covariates (which explains why we have eliminated 
#yaw_belt instead of roll_belt, and not the opposite: it is a "more important" covariate):

library(rpart.plot)
```

```
## Loading required package: rpart
```

```r
fitModel <- rpart(classe~., data=train1, method="class")
prp(fitModel)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7-2.png)
# Data Modeling


```r
##We are now ready to create our model.
##We are using a Random Forest algorithm, using the train() function from the caret package.
##We are using 9 variables out of the 53 as model parameters. These variables were among the 
##most significant variables generated by an initial 
##Random Forest algorithm, and are roll_belt, num_window, pitch_belt, magnet_dumbbell_y, 
##magnet_dumbbell_z, pitch_forearm, accel_dumbbell_y, roll_arm, and roll_forearm. 
##These variable are relatively independent as the maximum correlation among them is 50.57%.
##We are using a 2-fold cross-validation control. This is the simplest k-fold cross-validation ##possible and it will give a reduced computation time. 
##Because the data set is large, using a small number of folds is justified.

set.seed(3141592)
fitModel <- train(classe~roll_belt+num_window+pitch_belt+magnet_dumbbell_y+magnet_dumbbell_z+pitch_forearm+accel_dumbbell_y+roll_arm+roll_forearm,
                  data=train1,
                  method="rf",
                  trControl=trainControl(method="cv",number=2),
                  prox=TRUE,
                  verbose=TRUE,
                  allowParallel=TRUE)

saveRDS(fitModel, "modelRF.Rds")

##We can later use this tree, by allocating it directly to a variable using the command:
  
fitModel <- readRDS("modelRF.Rds")


predictions <- predict(fitModel, newdata=train2)
confusionMat <- confusionMatrix(predictions, train2$classe)
confusionMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    1    0    0    0
##          B    1 1513    0    0    1
##          C    0    4 1368    6    1
##          D    0    0    0 1280    3
##          E    0    0    0    0 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9965, 0.9987)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9973          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9967   1.0000   0.9953   0.9965
## Specificity            0.9998   0.9997   0.9983   0.9995   1.0000
## Pos Pred Value         0.9996   0.9987   0.9920   0.9977   1.0000
## Neg Pred Value         0.9998   0.9992   1.0000   0.9991   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1928   0.1744   0.1631   0.1832
## Detection Prevalence   0.2845   0.1931   0.1758   0.1635   0.1832
## Balanced Accuracy      0.9997   0.9982   0.9992   0.9974   0.9983
```

Accuracy is 99.78%,which is quite good.


## Error Rate

```r
missClass = function(values, predicted) {
  sum(predicted != values) / length(values)
}
OOS_errRate = missClass(train2$classe, predictions)
OOS_errRate
```

```
## [1] 0.002166709
```

The out-of-sample error rate is 0.23%.

# Making Predictions from test dataset

```r
predictions <- predict(fitModel, newdata=test)
test$classe <- predictions

submit <- data.frame(problem_id = test$problem_id, classe = predictions)
write.csv(submit, file = "coursera-submission.csv", row.names = FALSE)
```

#Conclusion

In this assignment, we accurately predicted the classification of 20 observations using a Random Forest algorithm trained on a subset of data using less than 20% of the covariates.

The accuracy obtained (accuracy = 99.77%, and out-of-sample error = 0.23%)


knit('varun.rmd', 'varun.md') # creates md file
markdownToHTML('test.md', 'test.html') # creates html file
