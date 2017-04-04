# Practical Machine Learing assignment
Minh Tu Pham  
### Executive summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

The goal of the project is built a model to predict the manner "classe" in which they did the exercise. The model is then used to predict 20 different test cases.

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

### Automatic or manual transmission? Which is better for MPG? 

1. Data processing 

Load data


```r
library(knitr)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
set.seed(333)
# Data for training and testing model
training <- read.csv('pml-training.csv', header = TRUE, sep = ",");

# create a partition with the training dataset 
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
traindata <- training[inTrain, ];
testdata <- training[-inTrain, ];

# Data for predicting
predicting <- read.csv('pml-testing.csv', header = TRUE, sep = ",");
```

In order to build a good model, only variables have less missing data are considered. It is clear that the data contains a lot of missing data. The missing values are in form of "NA" in numeric type, or in "blank space" in factor type. The overview of missing data can be seen with function "str()".


To make it easier, all the missing data in form of "blank space" are converted into "NA".


```r
traindata[traindata==""] <- NA;
```

Count the number of NA of each variables. And identify the name of variables that have more than 1000 NA. These variables are then remove from model construction.


```r
na_count <-sapply(traindata, function(y) sum(is.na(y)));
na_count <- data.frame(na_count);
# names of varibales (column) that have NA more than 100, these variables should be removed from model construction
na_row <- apply(na_count, 1, function(row) (row > 1000)); 
na_row <- which(na_row);
```

Construct new training, testing and predicting data that exclude the variables having many NA values.


```r
traindata <- traindata[,-na_row];
dim(traindata)
```

```
## [1] 13737    60
```

```r
testdata <- testdata[,-na_row];
dim(testdata)
```

```
## [1] 5885   60
```

```r
predictdata <- predicting[,-na_row];
dim(predictdata)
```

```
## [1] 20 60
```

As the columns 1 - 6 contain information of the name, time records, id which are not suitable for model constructions, these columns are also removed.


```r
traindata <- traindata[,-(1:6)];
dim(traindata)
```

```
## [1] 13737    54
```

```r
testdata <- testdata[,-(1:6)];
dim(testdata)
```

```
## [1] 5885   54
```

```r
predictdata <- predictdata[,-(1:6)];
dim(predictdata)
```

```
## [1] 20 54
```

### Model construction

The two methods, e.g. Random Forests and Generalized Boosted Model, are built based on the training data and tested with the testing.  The best one will be used for the quiz predictions.

1. Random Forests

Train Random Forests with training data and test the model with testdata.



Test the model with testdata


```r
# Testing dataset
TestRF <- predict(Fit1_RF, newdata=testdata)
ResultRF <- confusionMatrix(TestRF, testdata$classe)
ResultRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    3    0    0    0
##          B    0 1136    2    0    0
##          C    0    0 1024    4    0
##          D    0    0    0  960    4
##          E    0    0    0    0 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9962, 0.9988)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9972          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9981   0.9959   0.9963
## Specificity            0.9993   0.9996   0.9992   0.9992   1.0000
## Pos Pred Value         0.9982   0.9982   0.9961   0.9959   1.0000
## Neg Pred Value         1.0000   0.9994   0.9996   0.9992   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1930   0.1740   0.1631   0.1832
## Detection Prevalence   0.2850   0.1934   0.1747   0.1638   0.1832
## Balanced Accuracy      0.9996   0.9985   0.9986   0.9975   0.9982
```

2. Generalized Boosted Model



Test the model with testdata


```r
# Testing dataset
TestGBM <- predict(Fit2_GBM, newdata=testdata)
ResultGBM <- confusionMatrix(TestGBM, testdata$classe)
ResultGBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1667   17    0    0    1
##          B    6 1110    5    0    5
##          C    0   11 1012   10    2
##          D    0    1    9  952   19
##          E    1    0    0    2 1055
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9849          
##                  95% CI : (0.9814, 0.9878)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9809          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9745   0.9864   0.9876   0.9750
## Specificity            0.9957   0.9966   0.9953   0.9941   0.9994
## Pos Pred Value         0.9893   0.9858   0.9778   0.9704   0.9972
## Neg Pred Value         0.9983   0.9939   0.9971   0.9976   0.9944
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2833   0.1886   0.1720   0.1618   0.1793
## Detection Prevalence   0.2863   0.1913   0.1759   0.1667   0.1798
## Balanced Accuracy      0.9958   0.9856   0.9908   0.9908   0.9872
```

The accuracies of Random Forest and Generalized Boosted Model are 0.9978 and 0.9867 respectively.

3. Predictions

Only the Random Forest is chosen for prediction of predicting data.


```r
ResultPredict <- predict(Fit1_RF, newdata=predictdata);
ResultPredict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

