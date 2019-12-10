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

# Fit simple linear regression
lin_reg = lm(formula = Salary ~ ., data = dataset)


# Fit Polynomial linear regression
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset)


# Visualize Linear Regression result
# install.packages('ggplot2')
library(ggplot2)

ggplot() + 
	geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
	geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata=dataset)), 
			colour = 'blue') +
	ggtitle('Truth or Bluff (Linear Regression)') + 
	xlab('Level') + 
	ylab('Salary')


ggplot() + 
	geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
	geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata=dataset)), 
			colour = 'blue') +
	ggtitle('Truth or Bluff (Polynomial Linear Regression)') + 
	xlab('Level') + 
	ylab('Salary')


y_pred = predict(lin_reg, data.frame(Level = 6.5))

y_pred = predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, 
							Level3 = 6.5^3, 
							Level4 = 6.5^4))