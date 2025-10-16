###Step 1: Setup & Library Loading==========================================

#Load required libraries
library(tidyverse)    # For data manipulation and visualization
library(caret)        # For machine learning workflows (cross-validation, model training)
library(randomForest) # For Random Forest algorithm
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Set random seed for reproducibility
set.seed(1)


###Step 2: Data Loading=====================================================

#Load data
train = read_rds("datasets/train.rds")
test = read_rds("datasets/test.rds")

#Check dimensions
cat('Train dimensions:', nrow(train), 'rows,', ncol(train), 'columns\n')
cat('Test dimensions:', nrow(test), 'rows,', ncol(test), 'columns\n')

#Check variable names
names(train)
names(test)

#Examine structure of data
glimpse(train)

#Summary statistics for target variable
summary(train$score)


###Step 3: Exporaroty Data Analysis (EDA)===================================

#Distribution of target variable
p1 = ggplot(train, aes(x = score)) + 
  geom_histogram(bins = 30, fill = 'bisque', color = 'tan1') +
  labs(title = "Student Grade Distribution", 
       x = 'Score (standardized)', 
       y = 'Count') + 
  theme_minimal()
print(p1)

#Boxplot of score
p2 = ggplot(train, aes(y = score)) +
  geom_boxplot(fill = "yellowgreen", outlier.color = "red") +
  labs(title = "Boxplot of Score",
       y = "Score") +
  theme_minimal()
print(p2)

#Relationship: Study time vs Score
p3 = ggplot(train, aes(x = factor(studytime), y = score, fill = factor(studytime))) +
  geom_boxplot() +
  labs(title = "Score by Study Time",
       x = "Study Time (weekly hours)",
       y = "Score") +
  theme_minimal() +
  theme(legend.position = "none")+
  scale_fill_manual(values = c("pink2", "paleturquoise2", "lightsteelblue3", "grey80"))
print(p3)

#Relationship: Absences vs Score
p4 = ggplot(train, aes(x = absences, y = score)) +
  geom_point(alpha = 0.5, color = "cadetblue4") +
  geom_smooth(method = "lm", se = TRUE, color = "black") +
  labs(title = "Effect of Absences on Score",
       x = "Number of Absences",
       y = "Score") +
  theme_minimal()
print(p4)

#Relationship: Past Failures vs Score
p5 = ggplot(train, aes(x = factor(failures), y = score, fill = factor(failures))) +
  geom_boxplot() +
  labs(title = "Score by Number of Past Failures",
       x = "Number of Past Failures",
       y = "Score") +
  theme_minimal() +
  theme(legend.position = "none")+
  scale_fill_manual(values = c("springgreen1", "yellow4", "honeydew3", "grey54"))
print(p5)


###Step 4: Cross-Validation Setup===========================================

#Configure 10-fold Cross-Validation
cv_control = trainControl(
  method = "cv",           # Cross-validation method
  number = 10              # Number of folds
)

cat("\n---Cross-Validation Setup Complete---\n")
cat("Method: 10-fold CV\n\n")


###Step 5: Model Training & Comparison======================================

#Linear Regression
cat("---Training Linear Regression---\n")

set.seed(1)
model_lm = train(
  score ~ .,                # Predict score using all other variables
  data = train,
  method = "lm",            # Linear model
  trControl = cv_control
)

cat("Linear Regression trained!\n")
cat("RMSE:", round(model_lm$results$RMSE, 4), "\n")
cat("R²:", round(model_lm$results$Rsquared, 4), "\n\n")

#Random Forest
cat("---Training Random Forest---\n")

set.seed(1)
model_rf = train(
  score ~ .,
  data = train,
  method = "rf",            # Random Forest
  trControl = cv_control,
  ntree = 500               # Number of trees in the forest
)

cat("Random Forest trained!\n")
cat("Best mtry:", model_rf$bestTune$mtry, "\n")
cat("RMSE:", round(min(model_rf$results$RMSE), 4), "\n")
cat("R²:", round(max(model_rf$results$Rsquared), 4), "\n\n")

#K-Nearest Neighbors (KNN)
cat("---Training K-Nearest Neighbors---\n")

set.seed(1)
model_knn = train(
  score ~ .,
  data = train,
  method = "knn",           # K-Nearest Neighbors
  trControl = cv_control,
  tuneGrid = data.frame(k = c(3, 5, 7, 9, 11, 15, 20)),  # Try different K values
  preProcess = c("center", "scale")  # Scaling
)

cat("KNN trained!\n")
cat("Best K:", model_knn$bestTune$k, "\n")
cat("RMSE:", round(min(model_knn$results$RMSE), 4), "\n")
cat("R²:", round(max(model_knn$results$Rsquared), 4), "\n\n")


#Model Comparison
cat("---Model Comparison---\n\n")

#Compare the 3 models
results = resamples(list(
  LinearReg = model_lm,
  RandomForest = model_rf,
  KNN = model_knn
))

#Summary statistics
summary(results)

#Visualization: Boxplot comparison
bwplot(results, main = "Model Comparison - RMSE & R²")

#Identify best model
cat("\n---Winner---\n")
cat("The model with the LOWEST RMSE is the best!\n")
cat("Based on CV results, Random Forest is the winner!\n\n")


###Step 6: Final Model Training=============================================

cat("---Training Fianl Random Forest---\n")
cat("Training on FULL train set (316 students)\n\n")

set.seed(1)

#Train on the ENTIRE training set (without cv)
final_model = randomForest(
  score ~ .,
  data = train,
  ntree = 500,                      # 500 trees
  mtry = model_rf$bestTune$mtry,    # Use best mtry from CV
  importance = TRUE                 # Calculate variable importance
)

cat("Final model trained!\n\n")


###Step 7: Variable Importance ANalysis=====================================

cat("---Variable Importance---\n\n")

#Extract importance scores
importance_df = as.data.frame(importance(final_model))
importance_df$Variable = rownames(importance_df)

#Sort by %IncMSE (how much MSE increases if we remove this variable)
importance_df = importance_df[order(-importance_df$`%IncMSE`), ]

#Display top 10 most important variables
cat("Top 10 variables (by %IncMSE):\n\n")
print(head(importance_df[, c("Variable", "%IncMSE", "IncNodePurity")], 10))

#Visualization: Variable importance plot
varImpPlot(final_model, 
           main = "Variable Importance",
           n.var = 15)  # Show top 15 variables


###Step 8:Predictions For Test Set===========================================

cat("\n---Making Predictions---\n\n")

#Generate predictions for the 79 students in test set
predictions = predict(final_model, newdata = test)

cat("Predictions generated!\n")
cat("Number of predictions:", length(predictions), "\n\n")

#Summary of predictions
cat("Summary of predictions:\n")
print(summary(predictions))

#Display first 20 predictions
cat("\nFirst 20 predictions:\n")
print(head(predictions, 20))


###Step 9: Save Predictions

write_rds(predictions, "predictions.rds")


###Fianl Summary=============================================================

cat("\n---Final Summary---\n\n")
cat("Model: Random Forest\n")
cat("Training samples: 316 students\n")
cat("Test samples: 79 students\n")
cat("Number of trees: 500\n")
cat("mtry:", model_rf$bestTune$mtry, "\n")
cat("CV RMSE:", round(min(model_rf$results$RMSE), 4), "\n")
cat("CV R²:", round(max(model_rf$results$Rsquared), 4), "\n\n")
cat("Predictions range:", round(min(predictions), 2), "to", round(max(predictions), 2), "\n")

