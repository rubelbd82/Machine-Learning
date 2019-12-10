backwardElimination <- function(x, sl) {
    numVars = length(x)
    for (i in c(1:numVars)){
      regressor = lm(formula = Profit ~ ., data = x)
      maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
      if (maxVar > sl){
        j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
        x = x[, -j]
      }
      numVars = numVars - 1
    }
    return(summary(regressor))
  }


dataset = read.csv('50_Startups.csv')

dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3)) 

# caTools already added from Packages window
# install("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, 0.8)

training_set = subset(dataset, split = TRUE)
test_set = subset(dataset, split = FALSE)
  
  SL = 0.05
  dataset = dataset[, c(1,2,3,4,5)]
  backwardElimination(training_set, SL)