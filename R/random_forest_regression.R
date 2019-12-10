#Import dataset: 
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Install package if not already done
# install.packages('caTools')

# I've already selected caTools from Packages window of R studio
library(caTools)
#set.seed(123)
#split = sample.split(dataset$Salary, SplitRatio = 2/3)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Fit Random Forest Regression to the dataset
#install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], y = dataset$Salary, ntree = 10)


# Predict a new result with Regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualize Linear Regression result
# install.packages('ggplot2')
library(ggplot2)





# Visualize Random Forest Regression result (Higher resolution and smoother curve)

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() + 
	geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
	geom_line(aes(x = x_grid, y = predict(regressor, newdata=data.frame(Level = x_grid))), 
			colour = 'blue') +
	ggtitle('Truth or Bluff (Random Forest Regression)') + 
	xlab('Level') + 
	ylab('Salary')