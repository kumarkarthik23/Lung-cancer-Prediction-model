# Name: Ankasandra Naveen Kumar Karthik & Anusha Reddy Dasari 
# Class: ALY6015 Intermediate Analysis
# Date: 10/15/2024

# Clear the environment, console, and plots
rm(list = ls())  # Clear all objects from the global environment
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE)  # Clear all plots
options(scipen = 100)  # Disable scientific notation for readability
cat("\014")  # Clear the console 

# Load necessary libraries for data manipulation, visualization, and modeling
library(dplyr)        # For data manipulation
library(ggplot2)      # For visualizations
library(janitor)      # For cleaning column names
library(corrplot)     # For correlation matrix visualization
library(glmnet)       # For Lasso and Ridge regression
library(caret)        # For data partitioning and model evaluation
library(pROC)         # For plotting ROC curves

# Step 1: Load the dataset
lung_cancer_data <- read.csv("lung_cancer_data_1.csv")  # Ensure the file is in the correct directory

# Step 2: Clean column names using janitor to convert to snake_case format
lung_cancer_data <- lung_cancer_data %>%
  clean_names()

# Step 3: Select relevant columns based on project requirements
selected_data <- lung_cancer_data %>%
  select(
    age, gender, ethnicity,
    smoking_history, smoking_pack_years,
    tumor_size_mm, stage, tumor_location,
    comorbidity_diabetes, comorbidity_hypertension, comorbidity_heart_disease,
    comorbidity_chronic_lung_disease, comorbidity_kidney_disease, comorbidity_autoimmune_disease,
    treatment,
    hemoglobin_level, white_blood_cell_count, albumin_level,
    ldh_level, glucose_level,
    survival_months
  )

# Step 4: Convert categorical columns to factors for proper analysis
selected_data <- selected_data %>%
  mutate(
    gender = as.factor(gender),
    ethnicity = as.factor(ethnicity),
    smoking_history = as.factor(smoking_history),
    stage = as.factor(stage),
    tumor_location = as.factor(tumor_location),
    comorbidity_diabetes = as.factor(comorbidity_diabetes),
    comorbidity_hypertension = as.factor(comorbidity_hypertension),
    comorbidity_heart_disease = as.factor(comorbidity_heart_disease),
    comorbidity_chronic_lung_disease = as.factor(comorbidity_chronic_lung_disease),
    comorbidity_kidney_disease = as.factor(comorbidity_kidney_disease),
    comorbidity_autoimmune_disease = as.factor(comorbidity_autoimmune_disease),
    treatment = as.factor(treatment)
  )

# Step 5: Ensure numeric columns are treated correctly as numeric data types
selected_data <- selected_data %>%
  mutate(
    age = as.numeric(age),
    smoking_pack_years = as.numeric(smoking_pack_years),
    tumor_size_mm = as.numeric(tumor_size_mm),
    hemoglobin_level = as.numeric(hemoglobin_level),
    white_blood_cell_count = as.numeric(white_blood_cell_count),
    albumin_level = as.numeric(albumin_level),
    ldh_level = as.numeric(ldh_level),
    glucose_level = as.numeric(glucose_level),
    survival_months = as.numeric(survival_months)
  )

# Step 6: Check for missing values
missing_values <- colSums(is.na(selected_data))
cat("Missing Values per Column:\n")
print(missing_values)

# Step 7: Filter dataset for patients aged between 18 and 60 with positive survival months
selected_data <- selected_data %>%
  filter(age >= 18 & age <= 60, survival_months > 0)

# Step 8: Exploratory Data Analysis (EDA)

# 1. Bar plot: Tumor Location by Cancer Stage
ggplot(selected_data, aes(x = tumor_location, fill = stage)) +
  geom_bar(position = "dodge", alpha = 0.7) +
  labs(title = "Tumor Location by Cancer Stage", x = "Tumor Location", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2. Boxplot: Hemoglobin Levels by Treatment Type
ggplot(selected_data, aes(x = treatment, y = hemoglobin_level)) +
  geom_boxplot(alpha = 0.6) +
  labs(title = "Hemoglobin Levels by Treatment Type", x = "Treatment Type", y = "Hemoglobin Level (g/dL)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 3. Scatter plot: Age vs Survival Months with trend line
ggplot(selected_data, aes(x = age, y = survival_months)) +
  geom_point(color = "darkgreen", alpha = 0.6) +
  labs(title = "Age vs. Survival Months", x = "Age", y = "Survival Months") +
  geom_smooth(method = "lm", se = FALSE, color = "red")

# 4. Histogram: Distribution of Survival Months
ggplot(selected_data, aes(x = survival_months)) +
  geom_histogram(binwidth = 2, fill = "orange", alpha = 0.7) +
  labs(title = "Distribution of Survival Months", x = "Survival Months", y = "Count")

# 5. Correlation matrix for numeric variables
numeric_data <- selected_data %>% select(where(is.numeric))
correlation_matrix <- cor(numeric_data, use = "complete.obs")
corrplot(correlation_matrix, method = "circle", type = "lower", addCoef.col = "black", 
         tl.col = "red", tl.srt = 45, col = colorRampPalette(c("blue", "white", "red"))(200))

# Research Question :
## Can we predit a lung cancer patient will survive more than 60 months using logistic regression based on the features like (age,tumor_size,smoking_history and bio-medical markers)? 

# Step 9: Create a binary outcome variable for survival beyond 60 months
selected_data$survived_60_months <- ifelse(selected_data$survival_months > 60, 1, 0)

# Step 10: Select relevant features for logistic regression modeling
features <- c("age", "tumor_size_mm", "smoking_pack_years", "hemoglobin_level",
              "white_blood_cell_count", "albumin_level", "ldh_level", 
              "glucose_level")

# Step 11: Prepare data for model training
X <- as.matrix(selected_data[, features])
y <- selected_data$survived_60_months

# Split the dataset into training (70%) and test (30%) sets
set.seed(42)  # For reproducibility
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Step 12: Basic Logistic Regression (without regularization)
basic_glm <- glm(survived_60_months ~ ., data = selected_data[train_index, c(features, "survived_60_months")], family = binomial)
basic_glm_predictions <- predict(basic_glm, newdata = selected_data[-train_index, c(features, "survived_60_months")], type = "response")
basic_glm_pred_class <- ifelse(basic_glm_predictions > 0.5, 1, 0)

# Evaluate and visualize the basic logistic regression model
cat("Basic GLM Performance:\n")
print(confusionMatrix(as.factor(basic_glm_pred_class), as.factor(y_test)))

# ROC Curve for Basic Logistic Regression
basic_glm_roc <- roc(y_test, basic_glm_predictions)
plot(basic_glm_roc, col = "blue", lwd = 2, main = "ROC Curve for Basic Logistic Regression")

# Confusion Matrix Heatmap for Basic Logistic Regression
basic_glm_conf_matrix <- confusionMatrix(as.factor(basic_glm_pred_class), as.factor(y_test))
ggplot(data = as.data.frame(basic_glm_conf_matrix$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Confusion Matrix: Basic Logistic Regression") +
  scale_fill_gradient(low = "red", high = "green")

# Step 13: Lasso (L1) Regularization Logistic Regression
lasso_model <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1)

# Predictions for Lasso with lambda.min and lambda.1se
lasso_predictions_min <- predict(lasso_model, newx = X_test, s = "lambda.min", type = "class")
lasso_predictions_1se <- predict(lasso_model, newx = X_test, s = "lambda.1se", type = "class")
lasso_pred_class_min <- ifelse(lasso_predictions_min > 0.5, 1, 0)
lasso_pred_class_1se <- ifelse(lasso_predictions_1se > 0.5, 1, 0)

# Visualize Lasso coefficient paths
plot(lasso_model$glmnet.fit, xvar = "lambda", label = TRUE, main = "Lasso Coefficient Paths")

# Evaluate and visualize the Lasso model for lambda.min
cat("Lasso (L1) Performance with lambda.min:\n")
print(confusionMatrix(as.factor(lasso_pred_class_min), as.factor(y_test)))

# ROC Curve for Lasso (lambda.min)
lasso_roc_min <- roc(y_test, as.numeric(lasso_pred_class_min))
plot(lasso_roc_min, col = "blue", lwd = 2, main = "ROC Curve for Lasso (lambda.min)")

# Confusion Matrix Heatmap for Lasso (lambda.min)
lasso_conf_matrix_min <- confusionMatrix(as.factor(lasso_pred_class_min), as.factor(y_test))
ggplot(data = as.data.frame(lasso_conf_matrix_min$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Confusion Matrix: Lasso (lambda.min)") +
  scale_fill_gradient(low = "red", high = "green")

# Evaluate and visualize the Lasso model for lambda.1se
cat("Lasso (L1) Performance with lambda.1se:\n")
print(confusionMatrix(as.factor(lasso_pred_class_1se), as.factor(y_test)))

# ROC Curve for Lasso (lambda.1se)
lasso_roc_1se <- roc(y_test, as.numeric(lasso_pred_class_1se))
plot(lasso_roc_1se, col = "green", lwd = 2, main = "ROC Curve for Lasso (lambda.1se)")

# Confusion Matrix Heatmap for Lasso (lambda.1se)
lasso_conf_matrix_1se <- confusionMatrix(as.factor(lasso_pred_class_1se), as.factor(y_test))
ggplot(data = as.data.frame(lasso_conf_matrix_1se$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Confusion Matrix: Lasso (lambda.1se)") +
  scale_fill_gradient(low = "red", high = "green")

# Step 14: Ridge (L2) Regularization Logistic Regression
ridge_model <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 0)

# Predictions for Ridge with lambda.min and lambda.1se
ridge_predictions_min <- predict(ridge_model, newx = X_test, s = "lambda.min", type = "class")
ridge_predictions_1se <- predict(ridge_model, newx = X_test, s = "lambda.1se", type = "class")
ridge_pred_class_min <- ifelse(ridge_predictions_min > 0.5, 1, 0)
ridge_pred_class_1se <- ifelse(ridge_predictions_1se > 0.5, 1, 0)

# Visualize Ridge coefficient paths
plot(ridge_model$glmnet.fit, xvar = "lambda", label = TRUE, main = "Ridge Coefficient Paths")

# Evaluate and visualize the Ridge model for lambda.min
cat("Ridge (L2) Performance with lambda.min:\n")
print(confusionMatrix(as.factor(ridge_pred_class_min), as.factor(y_test)))

# ROC Curve for Ridge (lambda.min)
ridge_roc_min <- roc(y_test, as.numeric(ridge_pred_class_min))
plot(ridge_roc_min, col = "blue", lwd = 2, main = "ROC Curve for Ridge (lambda.min)")

# Confusion Matrix Heatmap for Ridge (lambda.min)
ridge_conf_matrix_min <- confusionMatrix(as.factor(ridge_pred_class_min), as.factor(y_test))
ggplot(data = as.data.frame(ridge_conf_matrix_min$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Confusion Matrix: Ridge (lambda.min)") +
  scale_fill_gradient(low = "red", high = "green")

# Evaluate and visualize the Ridge model for lambda.1se
cat("Ridge (L2) Performance with lambda.1se:\n")
print(confusionMatrix(as.factor(ridge_pred_class_1se), as.factor(y_test)))

# ROC Curve for Ridge (lambda.1se)
ridge_roc_1se <- roc(y_test, as.numeric(ridge_pred_class_1se))
plot(ridge_roc_1se, col = "green", lwd = 2, main = "ROC Curve for Ridge (lambda.1se)")

# Confusion Matrix Heatmap for Ridge (lambda.1se)
ridge_conf_matrix_1se <- confusionMatrix(as.factor(ridge_pred_class_1se), as.factor(y_test))
ggplot(data = as.data.frame(ridge_conf_matrix_1se$table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  labs(title = "Confusion Matrix: Ridge (lambda.1se)") +
  scale_fill_gradient(low = "red", high = "green")

# Research Question:
# Is there a statistically significant difference in survival time across different treatment types (e.g., Surgery, Chemotherapy, Radiation Therapy)?

# Step 1: Check the distribution of 'treatment' variable
cat("Distribution of treatment types:\n")
print(table(selected_data$treatment))

# Step 2: Perform Kruskal-Wallis Test to compare survival months across treatment types
kruskal_test <- kruskal.test(survival_months ~ treatment, data = selected_data)

# Step 3: Output the result of the Kruskal-Wallis test
cat("Kruskal-Wallis Test Results:\n")
print(kruskal_test)

# Step 4: Visualize survival time distribution by treatment type using boxplots
ggplot(selected_data, aes(x = treatment, y = survival_months)) +
  geom_boxplot(aes(fill = treatment), alpha = 0.6) +
  labs(title = "Survival Time by Treatment Type", x = "Treatment Type", y = "Survival Months") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

