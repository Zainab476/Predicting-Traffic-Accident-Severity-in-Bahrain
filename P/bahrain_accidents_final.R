# ============================================================================
# BDSA 602 Course Project: Predicting Traffic Accident Severity in Bahrain
# FINAL 100% WORKING VERSION – December 2025
# ============================================================================

# Load required libraries --------------------------------------------------
library(tidyverse)
library(caret)
library(MASS)
library(class)
library(pROC)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(scales)
library(nnet)        # for multinom()

# Reproducibility ----------------------------------------------------------
set.seed(123)

# ============================================================================
# 1. DATA LOADING – Finds the file automatically
# ============================================================================

possible_paths <- c(
  "cleaned_project_data1.txt",
  file.path("~", "Downloads", "cleaned_project_data1.txt"),
  file.path("~", "OneDrive", "Downloads", "cleaned_project_data1.txt"),
  file.path("~", "Desktop", "cleaned_project_data1.txt")
)

data_path <- possible_paths[file.exists(possible_paths)][1]

if (is.na(data_path)) {
  stop("\nFile 'cleaned_project_data1.txt' not found!\nPut it in the same folder as this script or in your Downloads folder.\n")
}

cat("Loading data from:", data_path, "\n")
data <- read.csv(data_path, sep = ";", header = TRUE, stringsAsFactors = FALSE)

# Basic info ---------------------------------------------------------------
cat("\nDataset Dimensions:\n"); print(dim(data))
cat("\nFirst 10 rows:\n"); print(head(data, 10))
cat("\nMissing values:\n"); print(colSums(is.na(data)))

# ============================================================================
# 2. DATA PREPROCESSING – NO MORE SELECT() CONFLICTS
# ============================================================================

df <- data

# Convert to factors
df$Accident_Severity <- as.factor(df$Accident_Severity)
df$Weather           <- as.factor(df$Weather)
df$Road_Type         <- as.factor(df$Road_Type)
df$Light_Conditions  <- as.factor(df$Light_Conditions)
df$Is_Night          <- as.factor(df$Is_Night)

# Remove Severity_Label safely without using dplyr::select()
if ("Severity_Label" %in% names(df)) {
  model_data <- df[, !names(df) %in% "Severity_Label"]
} else {
  model_data <- df
}

# Train-test split
trainIndex <- createDataPartition(model_data$Accident_Severity, p = 0.7, list = FALSE)
train_data <- model_data[trainIndex, ]
test_data  <- model_data[-trainIndex, ]

cat("\nTraining set:", nrow(train_data), "rows\n")
cat("Test set    :", nrow(test_data), "rows\n")

# ============================================================================
# 3. MODELS
# ============================================================================

cat("\n=== Training Logistic Regression ===\n")
logit_model <- nnet::multinom(Accident_Severity ~ Year + Weather + Road_Type +
                              Light_Conditions + Speed_Limit + Vehicles_Involved +
                              Casualties + Hour + Latitude + Longitude + Is_Night,
                              data = train_data, maxit = 1000, trace = FALSE)

logit_pred <- predict(logit_model, test_data)
logit_cm   <- confusionMatrix(logit_pred, test_data$Accident_Severity)
logit_acc  <- logit_cm$overall["Accuracy"]


cat("\n=== Training LDA ===\n")
lda_model <- lda(Accident_Severity ~ ., data = train_data)
lda_pred  <- predict(lda_model, test_data)$class
lda_cm     <- confusionMatrix(lda_pred, test_data$Accident_Severity)
lda_acc   <- lda_cm$overall["Accuracy"]


cat("\n=== Training QDA ===\n")
qda_model <- qda(Accident_Severity ~ ., data = train_data)
qda_pred  <- predict(qda_model, test_data)$class
qda_cm    <- confusionMatrix(qda_pred, test_data$Accident_Severity)
qda_acc   <- qda_cm$overall["Accuracy"]


cat("\n=== Training KNN ===\n")
# Use base R subsetting – no select() conflict
train_knn <- train_data[, c("Year", "Speed_Limit", "Vehicles_Involved", "Casualties",
                            "Hour", "Latitude", "Longitude", "Is_Night")]
test_knn  <- test_data[,  c("Year", "Speed_Limit", "Vehicles_Involved", "Casualties",
                            "Hour", "Latitude", "Longitude", "Is_Night")]

train_knn$Is_Night <- as.numeric(as.character(train_knn$Is_Night))
test_knn$Is_Night  <- as.numeric(as.character(test_knn$Is_Night))

preProc <- preProcess(train_knn, method = c("center", "scale"))
train_scaled <- predict(preProc, train_knn)
test_scaled  <- predict(preProc, test_knn)

# Find best k (odd numbers up to 31)
k_vals <- seq(1, 31, by = 2)
cv_acc <- sapply(k_vals, function(k) {
  pred <- knn.cv(train_scaled, train_data$Accident_Severity, k = k)
  mean(pred == train_data$Accident_Severity)
})
best_k <- k_vals[which.max(cv_acc)]
cat("Optimal K =", best_k, "\n")

knn_pred <- knn(train_scaled, test_scaled, train_data$Accident_Severity, k = best_k)
knn_cm   <- confusionMatrix(knn_pred, test_data$Accident_Severity)
knn_acc  <- knn_cm$overall["Accuracy"]

# ============================================================================
# 4. FINAL RESULTS
# ============================================================================

results <- data.frame(
  Model    = c("Logistic Regression", "LDA", "QDA", "KNN"),
  Accuracy = round(c(logit_acc, lda_acc, qda_acc, knn_acc), 4)
)

cat("\n\n=== PERFORMANCE COMPARISON TABLE ===\n")
print(results)

cat("\n*** BEST MODEL:", results$Model[which.max(results$Accuracy)], "***\n")
cat("Highest Accuracy:", max(results$Accuracy), "\n")

# Save results
write.csv(results, "model_comparison_results.csv", row.names = FALSE)

sink("confusion_matrices.txt")
cat("=== Logistic Regression ===\n"); print(logit_cm)
cat("\n=== LDA ===\n"); print(lda_cm)
cat("\n=== QDA ===\n"); print(qda_cm)
cat("\n=== KNN ===\n"); print(knn_cm)
sink()

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Results saved to your project folder:\n")
cat("   • model_comparison_results.csv\n")
cat("   • confusion_matrices.txt\n")
cat("Working directory:", getwd(), "\n")