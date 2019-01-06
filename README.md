# Titanic Is Sinking?

Attempt to build predictive models on the survival rate of titanic passengers.
Overall, the process can be described as follow:

1. Performed imputation on missing data using bagImpute from caret (bagImpute provided better results compare to normal replacement by mean values)
2. Conducted binning on multiple categories (Fares, Ages, Family Size, Siblings Size)
3. Tried to categorize "Name" into "Title" but it didn't yield better result
4. Constructed baseline model with simple Logistic Regression, then move on to more complicated ones.
5. Ensemble using voting method of 6 models


### Rooms for improvement:
- Feature selection should be implemented before building models
- Parameters optimization for modeling (esp. random forest and gradient boosting machine)


# Initialize packages


```r
# load packages and set options
options(stringsAsFactors = FALSE)

# install packages if not available
packages <- c("readr", #read data
              "lubridate", #date time conversion
              "tidyverse", # full set of pkgs
              "dplyr", #data exploratory + manipulation
              "caTools", # features engineering
              "ggplot2","ggthemes", "corrplot", # plotting graphs
              "caret", # ML libs
              "Hmisc" # EDA
)

if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))
}
lapply(packages, require, character.only = TRUE)
```

# Read data

Read train, test and submit sample data from input folder


```r
# set TryCatch for input folder (on kaggle kernel or local)

?tryCatch
test <- tryCatch(read_csv("../input/test.csv"),
                 error = function(e){
                   print('Detect working environment is not Kaggle kernel')
                   read_csv("./input/test.csv")
                 })

train <- tryCatch(read_csv("../input/train.csv"),
                 error = function(e){
                   print('Detect working environment is not Kaggle kernel')
                   read_csv("./input/train.csv")
                 })

# Add target variable
test$Survived <- NA
```

# EDA
Merge train and test dataset to get an overview of all features


```r
full <- rbind(train,test)
```
Visualize missing data using **VIM** package


```r
aggr_plot <- VIM::aggr(full, 
                       col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(data), cex.axis=.7, gap=3,
                       ylab=c("Histogram of missing data","Pattern"))
```

```
## Warning in plot.aggr(res, ...): not enough horizontal space to display
## frequencies
```

![](https://github.com/o0oBluePhoenixo0o/Kaggle-Titanic-0.78/blob/master/images/Missing%20data-1.png)<!-- -->

```
## 
##  Variables sorted by number of missings: 
##     Variable        Count
##        Cabin 0.7746371276
##     Survived 0.3193277311
##          Age 0.2009167303
##     Embarked 0.0015278839
##         Fare 0.0007639419
##  PassengerId 0.0000000000
##       Pclass 0.0000000000
##         Name 0.0000000000
##          Sex 0.0000000000
##        SibSp 0.0000000000
##        Parch 0.0000000000
##       Ticket 0.0000000000
```

- As can be seen "Cabin" feature has a lot of missing data. Since it recorded which cabins that passengers stayed on Titanic during the journey, I assume it is a MAR (missing-at-random) issue- some are recorded, some are not since the ticket class put them in common area (?)

- Another feature that contains many missing data is Age. It might be that the passengers did not put their ages when purchased the tickets.

With these findings, I tried both with traditional imputation (**Method 1**) where missing values are replaced by means or most common value; and with K-nn / BagImpute (**Method 2**) from the caret package.


```r
# Age vs Survived
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram(bins=30) + 
  theme_few() +
  xlab("Age") +
  scale_fill_discrete(name = "Survived") + 
  ggtitle("Age vs Survived")
```

```
## Warning: Removed 177 rows containing non-finite values (stat_bin).
```

![](https://github.com/o0oBluePhoenixo0o/Kaggle-Titanic-0.78/blob/master/images/EDA-%20Impute%20missing%20data-1.png)<!-- -->

```r
# Age vs Sex vs Survived
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram(bins=30) + 
  theme_few() +
  xlab("Age") +
  ylab("Count") +
  facet_grid(.~Sex)+
  scale_fill_discrete(name = "Survived") + 
  theme_few()+
  ggtitle("Age vs Sex vs Survived")
```

```
## Warning: Removed 177 rows containing non-finite values (stat_bin).
```

![](https://github.com/o0oBluePhoenixo0o/Kaggle-Titanic-0.78/blob/master/images/EDA-%20Impute%20missing%20data-2.png)<!-- -->

```r
# Replace Embarked with most common 
train$Embarked <- replace(train$Embarked, which(is.na(train$Embarked)), 'S')
full$Embarked <- replace(full$Embarked, which(is.na(full$Embarked)), 'S')

## Method 1- replace by mean

# # Replace missing age = mean of all ages / same with Fares
# train$Age[is.na(train$Age)] <- round(mean(train$Age, na.rm = T),0)
# train$Fare[is.na(train$Fare)] <- round(mean(train$Fare, na.rm = T),0)

# # Replace missing age = mean of all ages / same with Fares
# test$Age[is.na(test$Age)] <- round(mean(test$Age, na.rm = T),0)
# test$Fare[is.na(test$Fare)] <- round(mean(test$Fare, na.rm = T),0)


## Method 2- baggedImpute on full dataset
# Age + Fare are 2 missing variables

missing.df <- full %>%
  select(-c(Survived, Cabin, PassengerId, Name, Ticket))

missing_model <- caret::preProcess(missing.df, method = "bagImpute")
missing.result <- predict(missing_model, missing.df)

# Add back columns to full, train and test datasets
missing.result <- cbind(full$PassengerId, missing.result)

missing.result <- missing.result %>%
  rename(PassengerId = 'full$PassengerId') %>%
  select(PassengerId, Age, Fare)

full <- full %>%
  select(-c(Age,Fare))
full <- left_join(full,missing.result, by = "PassengerId")

train <- train %>% 
  select(-c(Age,Fare)) 
train <- left_join(train,missing.result, by = "PassengerId")

test <- test %>%
  select(-c(Age,Fare))
test <- left_join(test,missing.result, by = "PassengerId")
```

# Features Engineer

Exclude un-related features such as *PassengerId, Name, Ticket, Cabin*


```r
# Exclude unrelated features
train.df <- train %>%
  select(-c(PassengerId,
            Name,
            Ticket,
            Cabin))
```

## Binning for ages
- 0 to 18
- 18 to 65
- 65 and above


```r
# Age distribution (full dataset)
qplot(full$Age,
      geom = 'histogram', binwidth = 0.5,
      main = 'Age distribution of passengers (full dataset)',
      xlab = 'Age',
      ylab = 'Count')
```

![](https://github.com/o0oBluePhoenixo0o/Kaggle-Titanic-0.78/blob/master/images/FE-%20binning%20for%20ages-1.png)<!-- -->

```r
# Binning for train
train.df <- train.df %>%
  mutate(Age_Range = case_when(
    Age < 18 ~ 'Kids',
    Age >=18 & Age < 60 ~ 'Adults',
    Age >=60 ~ 'Seniors'
  )) %>%
  select (-Age)
```
## Binning for fares
Cheapest < Cheap < Moderate < Expensive < Most expensive


```r
# Fare distribution
qplot(full$Fare,
      geom = 'histogram', binwidth = 10,
      main = 'Fare distribution of passengers',
      xlab = 'Fare',
      ylab = 'Count')
```

![](https://github.com/o0oBluePhoenixo0o/Kaggle-Titanic-0.78/blob/master/images/FE-%20binning%20for%20fares-1.png)<!-- -->

```r
# Binning fare
train.df <- train.df %>%
  mutate(Fare_Range = case_when(
    Fare < 50 ~ 'Cheapest',
    Fare >= 50 & Fare < 100 ~ 'Cheap',
    Fare >= 100 & Fare < 200 ~ 'Moderate',
    Fare >= 200 & Fare < 500 ~ 'Expensive',
    Fare >= 500 ~ 'Most expensive'
  )) %>%
  select(-Fare)
```


## Siblings and partners count binning

More than 4 siblings/partners are considered as "Many", between 2 and 4 are "Some" and less than 2 are "Few"


```r
# Siblings distribution
qplot(full$SibSp,
      geom = 'histogram', binwidth = 0.5,
      main = 'Siblings distribution of passengers',
      xlab = 'Siblings',
      ylab = 'Count')
```

![](https://github.com/o0oBluePhoenixo0o/Kaggle-Titanic-0.78/blob/master/images/FE-%20Bining%20siblings%20and%20partners-1.png)<!-- -->

```r
# Binning siblngs
train.df <- train.df %>%
  mutate(Siblings_Range = case_when(
    SibSp < 2 ~ 'Few',
    SibSp >= 2 & SibSp < 4 ~ 'Some',
    SibSp >= 4 ~ 'Many'
  )) %>%
  select(-SibSp)

#####
# Partner distribution
qplot(full$Parch,
      geom = 'histogram', binwidth = 0.5,
      main = 'Partner distribution of passengers',
      xlab = 'Partners',
      ylab = 'Count')
```

![](https://github.com/o0oBluePhoenixo0o/Kaggle-Titanic-0.78/blob/master/images/FE-%20Bining%20siblings%20and%20partners-2.png)<!-- -->

```r
# Binning partner
train.df <- train.df %>%
  mutate(Partner_Range = case_when(
    Parch < 2 ~ 'Few',
    Parch >= 2 & Parch < 4 ~ 'Some',
    Parch >= 4 ~ 'Many'
  )) %>%
  select(-Parch)
```

Perform FE on test dataset


```r
# Replace Embarked with most common 
test$Embarked <- replace(test$Embarked, which(is.na(test$Embarked)), 'S')

# FE for test dataset
test.df <- test %>%
  select(-c(PassengerId,
            Name,
            Ticket,
            Cabin,
            Survived)) %>%
  mutate(Age_Range = case_when(
    Age < 18 ~ 'Kids',
    Age >=18 & Age < 60 ~ 'Adults',
    Age >=60 ~ 'Seniors'
  )) %>%
  select (-Age) %>%
  mutate(Partner_Range = case_when(
    Parch < 2 ~ 'Few',
    Parch >= 2 & Parch < 4 ~ 'Some',
    Parch >= 4 ~ 'Many'
  )) %>%
  select(-Parch) %>%
  mutate(Siblings_Range = case_when(
    SibSp < 2 ~ 'Few',
    SibSp >= 2 & SibSp < 4 ~ 'Some',
    SibSp >= 4 ~ 'Many'
  )) %>%
  select(-SibSp) %>%
  mutate(Fare_Range = case_when(
    Fare < 50 ~ 'Cheapest',
    Fare >= 50 & Fare < 100 ~ 'Cheap',
    Fare >= 100 & Fare < 200 ~ 'Moderate',
    Fare >= 200 & Fare < 500 ~ 'Expensive',
    Fare >= 500 ~ 'Most expensive'
  )) %>%
  select(-Fare)
```

# Models Development

Split into training and testing set (70-30)


```r
# Set seed for code reproduction

set.seed(1908)
split <- caTools::sample.split(train.df$Survived, SplitRatio = 0.7)

train.df$Survived <- as.factor(train.df$Survived)

real.train <- subset(train.df , split == TRUE)
real.test <- subset(train.df, split == FALSE)

# Cross-validation (5)
train_control <- trainControl(## 5-cross validation
  method = "cv",
  number = 5)
```

Starting off with base-line model with bayesian generalized linear model (logistic regression)


```r
LogiModel <- train(Survived ~.,
                   data = real.train,
                   method = 'bayesglm',
                   trControl = train_control)

# Prediction

pred.Logi <- predict(LogiModel,
                     newdata = real.test[,2:ncol(real.test)],
                     type = "raw")

CM.Logi <- confusionMatrix(as.factor(pred.Logi), real.test$Survived) # 84% acc
CM.Logi
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 141  19
##          1  24  84
##                                         
##                Accuracy : 0.8396        
##                  95% CI : (0.79, 0.8814)
##     No Information Rate : 0.6157        
##     P-Value [Acc > NIR] : 9.608e-16     
##                                         
##                   Kappa : 0.664         
##  Mcnemar's Test P-Value : 0.5419        
##                                         
##             Sensitivity : 0.8545        
##             Specificity : 0.8155        
##          Pos Pred Value : 0.8812        
##          Neg Pred Value : 0.7778        
##              Prevalence : 0.6157        
##          Detection Rate : 0.5261        
##    Detection Prevalence : 0.5970        
##       Balanced Accuracy : 0.8350        
##                                         
##        'Positive' Class : 0             
## 
```

```r
recall(CM.Logi$table) # 86%
```

```
## [1] 0.8545455
```

```r
precision(CM.Logi$table) # 88%
```

```
## [1] 0.88125
```

Testing with other models- Naive Bayes, Random Forest, SVM


```r
NB <- train(Survived ~.,
            data = real.train,
            method = 'naive_bayes',
            #na.action = na.pass,
            trControl = train_control,
            seed = 1234)

# Prediction

pred.NB <- predict(NB,
                   newdata = real.test[,2:ncol(real.test)],
                   na.action = na.pass,
                   type = "raw")

CM.NB <- confusionMatrix(as.factor(pred.NB), real.test$Survived) # 82% acc
CM.NB
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 156  41
##          1   9  62
##                                           
##                Accuracy : 0.8134          
##                  95% CI : (0.7615, 0.8582)
##     No Information Rate : 0.6157          
##     P-Value [Acc > NIR] : 2.048e-12       
##                                           
##                   Kappa : 0.5813          
##  Mcnemar's Test P-Value : 1.165e-05       
##                                           
##             Sensitivity : 0.9455          
##             Specificity : 0.6019          
##          Pos Pred Value : 0.7919          
##          Neg Pred Value : 0.8732          
##              Prevalence : 0.6157          
##          Detection Rate : 0.5821          
##    Detection Prevalence : 0.7351          
##       Balanced Accuracy : 0.7737          
##                                           
##        'Positive' Class : 0               
## 
```

```r
recall(CM.NB$table) # 95%
```

```
## [1] 0.9454545
```

```r
precision(CM.NB$table) # 79%
```

```
## [1] 0.7918782
```



```r
RF <- train(Survived ~.,
            data = real.train,
            method = 'rf',
            trControl = train_control,
            seed = 1234)

# Prediction

pred.rf <- predict(RF,
                   newdata = real.test[,2:ncol(real.test)],
                   type = "raw")

CM.RF <- confusionMatrix(as.factor(pred.rf), real.test$Survived) # 86% acc
CM.RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 158  35
##          1   7  68
##                                           
##                Accuracy : 0.8433          
##                  95% CI : (0.7941, 0.8847)
##     No Information Rate : 0.6157          
##     P-Value [Acc > NIR] : 2.894e-16       
##                                           
##                   Kappa : 0.651           
##  Mcnemar's Test P-Value : 3.097e-05       
##                                           
##             Sensitivity : 0.9576          
##             Specificity : 0.6602          
##          Pos Pred Value : 0.8187          
##          Neg Pred Value : 0.9067          
##              Prevalence : 0.6157          
##          Detection Rate : 0.5896          
##    Detection Prevalence : 0.7201          
##       Balanced Accuracy : 0.8089          
##                                           
##        'Positive' Class : 0               
## 
```

```r
recall(CM.RF$table) # 96%
```

```
## [1] 0.9575758
```

```r
precision(CM.RF$table)# 82%
```

```
## [1] 0.8186528
```

For SVM, since we only have few predictors over observations, I chose svmRadial (Gaussian) over svmLinear). Usually SVM should be able to yield better result with small dataset (which is our case)


```r
SVM <- train(Survived ~.,
                    data = real.train,
                    method = 'svmRadial',
                    trControl = train_control,
                    seed = 1234)
```

```
## Warning in .local(x, ...): Variable(s) `' constant. Cannot scale data.

## Warning in .local(x, ...): Variable(s) `' constant. Cannot scale data.

## Warning in .local(x, ...): Variable(s) `' constant. Cannot scale data.
```

```r
# Prediction

pred.SVM <- predict(SVM,
                           newdata = real.test[,2:ncol(real.test)],
                           type = "raw")

CM.SVM <- confusionMatrix(as.factor(pred.SVM), real.test$Survived) # 84% acc
CM.SVM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 147  28
##          1  18  75
##                                           
##                Accuracy : 0.8284          
##                  95% CI : (0.7778, 0.8715)
##     No Information Rate : 0.6157          
##     P-Value [Acc > NIR] : 2.992e-14       
##                                           
##                   Kappa : 0.6306          
##  Mcnemar's Test P-Value : 0.1845          
##                                           
##             Sensitivity : 0.8909          
##             Specificity : 0.7282          
##          Pos Pred Value : 0.8400          
##          Neg Pred Value : 0.8065          
##              Prevalence : 0.6157          
##          Detection Rate : 0.5485          
##    Detection Prevalence : 0.6530          
##       Balanced Accuracy : 0.8095          
##                                           
##        'Positive' Class : 0               
## 
```

```r
recall(CM.SVM$table) # 89%
```

```
## [1] 0.8909091
```

```r
precision(CM.SVM$table)# 85%
```

```
## [1] 0.84
```


```r
Rpart <- train(Survived ~.,
               data = real.train,
               method = 'rpart',
               trControl = train_control)

# Prediction
pred.Rpart <- predict(Rpart,
                      newdata = real.test[,2:ncol(real.test)],
                      type = "raw")

CM.Rpart <- confusionMatrix(as.factor(pred.Rpart), real.test$Survived) # 80% acc
CM.Rpart
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 162  51
##          1   3  52
##                                           
##                Accuracy : 0.7985          
##                  95% CI : (0.7454, 0.8449)
##     No Information Rate : 0.6157          
##     P-Value [Acc > NIR] : 9.530e-11       
##                                           
##                   Kappa : 0.5334          
##  Mcnemar's Test P-Value : 1.596e-10       
##                                           
##             Sensitivity : 0.9818          
##             Specificity : 0.5049          
##          Pos Pred Value : 0.7606          
##          Neg Pred Value : 0.9455          
##              Prevalence : 0.6157          
##          Detection Rate : 0.6045          
##    Detection Prevalence : 0.7948          
##       Balanced Accuracy : 0.7433          
##                                           
##        'Positive' Class : 0               
## 
```

```r
recall(CM.Rpart$table) # 98%
```

```
## [1] 0.9818182
```

```r
precision(CM.Rpart$table)# 76%
```

```
## [1] 0.7605634
```


```r
GBM <- train(Survived ~.,
             data = real.train,
             method = 'gbm',
             trControl = train_control)
```


```r
# Prediction

pred.GBM <- predict(GBM,
                    newdata = real.test[,2:ncol(real.test)],
                    type = "raw")

CM.GBM <- confusionMatrix(as.factor(pred.GBM), real.test$Survived) # 86% acc
CM.GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   0   1
##          0 159  30
##          1   6  73
##                                           
##                Accuracy : 0.8657          
##                  95% CI : (0.8189, 0.9041)
##     No Information Rate : 0.6157          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.7032          
##  Mcnemar's Test P-Value : 0.0001264       
##                                           
##             Sensitivity : 0.9636          
##             Specificity : 0.7087          
##          Pos Pred Value : 0.8413          
##          Neg Pred Value : 0.9241          
##              Prevalence : 0.6157          
##          Detection Rate : 0.5933          
##    Detection Prevalence : 0.7052          
##       Balanced Accuracy : 0.8362          
##                                           
##        'Positive' Class : 0               
## 
```

```r
recall(CM.GBM$table) # 96%
```

```
## [1] 0.9636364
```

```r
precision(CM.GBM$table)# 84%
```

```
## [1] 0.8412698
```

### Ensemble Modeling (Voting, Bagging, Boosting) to get best results


```r
# Apply models to test.df
pred.Logi <- predict(LogiModel,
                     newdata = test.df[,1:ncol(test.df)],
                     #na.action = na.pass,
                     type = "raw"
)

pred.NB <- predict(NB,
                   newdata = test.df[,1:ncol(test.df)],
                   #na.action = na.pass,
                   type = "raw")

pred.rf <- predict(RF,
                   newdata = test.df[,1:ncol(test.df)],
                   #na.action = na.pass,
                   type = "raw")

pred.Rpart <- predict(Rpart,
                      newdata = test.df[,1:ncol(test.df)],
                      na.action = na.pass,
                      type = "raw")

pred.SVM <- predict(SVM,
                           newdata = test.df[,1:ncol(test.df)],
                           #na.action = na.pass,
                           type = "raw")
pred.GBM <- predict(GBM,
                    newdata = test.df[,1:ncol(test.df)],
                    #na.action = na.pass,
                    type = "raw")


submission <- test %>%
  select(PassengerId)
submission <- cbind(submission,
                    as.numeric(pred.Logi),
                    as.numeric(pred.NB),
                    as.numeric(pred.rf),
                    as.numeric(pred.Rpart),
                    as.numeric(pred.SVM),
                    as.numeric(pred.GBM))
colnames(submission) <- c('PassengerId',
                          'Logi',
                          'NB',
                          'RF',
                          'Rpart',
                          'SVM',
                          'GBM')

submission <- submission %>%
  mutate(Logi = ifelse(Logi == 1,0,1),
         NB = ifelse(NB == 1,0,1),
         RF = ifelse(RF == 1,0,1),
         Rpart = ifelse(Rpart == 1,0,1),
         SVM = ifelse(SVM == 1,0,1),
         GBM = ifelse(GBM == 1,0,1))
```



```r
#compose correlations plot
corrplot.mixed(cor(submission[,2:ncol(submission)]), order="hclust", tl.col="black")
```

![](https://github.com/o0oBluePhoenixo0o/Kaggle-Titanic-0.78/blob/master/images/Ensemble%20-%20Get%20correlation%20plot-1.png)<!-- -->



```r
# GBM only
submission.GBM <- submission %>%
  select(PassengerId, GBM) %>%
  rename(Survived = GBM)

write.csv(submission.GBM,'GBM.csv',
          row.names = FALSE)

##########
# SVM only
submission.SVM <- submission %>%
  select(PassengerId, SVM) %>%
  rename(Survived = SVM)

write.csv(submission.SVM,'SVM.csv',
          row.names = FALSE)

###################
# Naive Bayes only
submission.NB <- submission %>%
  select(PassengerId, NB) %>%
  rename(Survived = NB)

write.csv(submission.NB,'NB.csv',
          row.names = FALSE)

###################
# CART only
submission.Rpart <- submission %>%
  select(PassengerId, Rpart) %>%
  rename(Survived = Rpart)

write.csv(submission.Rpart,'RPart.csv',
          row.names = FALSE)

###################
# Random Forest only
submission.RF <- submission %>%
  select(PassengerId, RF) %>%
  rename(Survived = RF)

write.csv(submission.RF,'RF.csv',
          row.names = FALSE)

###################

# Majority Voting
submission.Major <- submission %>%
  mutate(Survived = Logi + NB + RF + Rpart + SVM + GBM) %>%
  mutate(Survived = ifelse(Survived >= 3,1,0))

submission.Major <- submission.Major %>%
  select(PassengerId, Survived)
write.csv(submission.Major,'Major.csv',
          row.names = FALSE)
```
