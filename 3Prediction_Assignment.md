---
title: 'Practical Machine Learning: Prediction'
output: html_document
---
##1. Overview
As part of the Specialization Requirement in Data Science, this document was published as a final report of the Peer Assessment project from Coursera Learning Practical Machine Module course.

The purpose of this analysis is to predict how 6 participants performed some of the exercises as described below. Using the "classe" variable in the set of exercises, accompanied by other variables, the report is published by describing the model construction method, cross-verification and expected out of sample error.

##2. Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website 

here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##3. Database and Data Processing
```{r setup, include=TRUE}
knitr::opts_chunk$set(echo = TRUE)
library(knitr)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
set.seed(12345)
```

###Loading and Cleaning Data
```{r}
training_mod <- read.csv("D:/Coursera/pml-training.csv")
test_set <- read.csv("D:/Coursera/pml-testing.csv")

TrainPart  <- createDataPartition(training_mod$classe, p=0.7, list=FALSE)
Train_Grp <- training_mod[TrainPart, ]
Test_Grp  <- training_mod[-TrainPart, ]
dim(Train_Grp)

dim(Test_Grp)
```

The datasets have 160 variables respectively with a lot of of NA's, which require the cleaning procedures. The Near Zero variance (NZ_Var) variables are also detached as well as the ID variables.

```{r}
NZ_Var <- nearZeroVar(Train_Grp)
Train_Grp <- Train_Grp[, -NZ_Var]
Test_Grp  <- Test_Grp[, -NZ_Var]
dim(Train_Grp)

dim(Test_Grp)

na_var    <- sapply(Train_Grp, function(x) mean(is.na(x))) > 0.95
Train_Grp <- Train_Grp[, na_var==FALSE]
Test_Grp  <- Test_Grp[, na_var==FALSE]
dim(Train_Grp)

Train_Grp <- Train_Grp[, -(1:5)]
Test_Grp  <- Test_Grp[, -(1:5)]
dim(Train_Grp)

dim(Test_Grp)
```

As the cleaning process takes place, there are 54 variables left.

##4. Correlation
Analysis between the variables is performed to get the correlation before the modeling procedure is implemented.

```{r}
cor_mx <- cor(Train_Grp[, -54])
corrplot(cor_mx, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```

The graph above shown the extremely correlated variables (dark color).  However, the compact analysis such as a PCA (Principal Components Analysis) could be made as pre-processing phase to the datasets. Nonetheless, due to a slight correlation, this step will not be used for this task.

##5. Prediction Model
To modelling the regression (Train dataset), three methods will be used, including Random Forest, Decision Tree and Generalized Boosted Model.

At the end of each analysis the Confusion Matrix is plotted to envisage the model's precision.

###i. Random Forest

```{r}
set.seed(12345)
RF_ctrl <- trainControl(method="cv", number=3, verboseIter=FALSE)
RF_modFit <- train(classe ~ ., data=Train_Grp, method="rf",trControl=RF_ctrl)
RF_modFit$finalModel

RF_predict <- predict(RF_modFit, newdata=Test_Grp)
RF_confMat <- confusionMatrix(RF_predict, Test_Grp$classe)
RF_confMat

plot(RF_confMat$table, col = RF_confMat$byClass, 
     main = paste("Random Forest: Accuracy =",
                  round(RF_confMat$overall['Accuracy'], 4)))
```
###ii. Decision Trees

```{r}
set.seed(12345)
DT_modFit <- rpart(classe ~ ., data=Train_Grp, method="class")
fancyRpartPlot(DT_modFit)


DT_predict <- predict(DT_modFit, newdata=Test_Grp, type="class")
DT_confMat <- confusionMatrix(DT_predict, Test_Grp$classe)
DT_confMat

plot(DT_confMat$table, col = DT_confMat$byClass, 
     main = paste("Decision Tree: Accuracy =",
                  round(DT_confMat$overall['Accuracy'], 4)))
```

###iii. Generalized Boosted Model
```{r}
set.seed(12345)
GBM_ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
GBM_modFit  <- train(classe ~ ., data=Train_Grp, method = "gbm",
                    trControl = GBM_ctrl, verbose = FALSE)
GBM_modFit$finalModel

GBM_predict <- predict(GBM_modFit, newdata=Test_Grp)
GBM_confMat <- confusionMatrix(GBM_predict, Test_Grp$classe)
GBM_confMat

plot(GBM_confMat$table, col = GBM_confMat$byClass, 
     main = paste("GBM: Accuracy =", round(GBM_confMat$overall['Accuracy'], 4)))
```


The 3 regression modeling methods accuracy are:
a.	Random Forest: 0.9968
b.	Decision Tree: 0.7368
c.	GBM: 0.9857


##6. Run the Model to the Test Data

To predict the 20 quiz results (testing dataset), the Random Forest model will be used.

```{r}
TEST_predict <- predict(RF_modFit, newdata=test_set)
TEST_predict

```

