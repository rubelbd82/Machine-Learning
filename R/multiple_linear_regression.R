#Import dataset: 
dataset = read.csv('50_Startups.csv')

dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3)) 

# caTools already added from Packages window
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, 0.8)

training_set = subset(dataset, split = TRUE)
test_set = subset(dataset, split = FALSE)


regressor = lm(formula = Profit ~ ., 
               data = training_set)
