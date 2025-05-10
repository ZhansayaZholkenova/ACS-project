library(tidyverse)
library(caret)
library(ROSE)
library(FSelector)
library(ipred)
library(smotefamily)
library(xgboost)
library(lightgbm)
library(ada)
library(e1071)
library(mltools)
library(pROC)
library(data.table)
library(Boruta)
library(RWeka)
library(rsample)
library(rpart)
library(dplyr)
library(ggplot2)
library(reshape2)
library(corrplot)

####Feature Selection####
df <- read.csv('project_data.csv')
head(df)

# Checking the structure of the dataset
str(df)
ncol(df)
nrow(df)

barplot(table(df$Class), main='Difficulty living independently')
prop.table(table(df$Class))

# checking for duplicate rows in dataset
duplicates <- df[duplicated(df), ]
print(nrow(duplicates)) # we don't have duplicate rows in the dataset

# checking the Class (target) column 
print(unique(df$Class))
# we will rename the "Yes" "No" values in Class to 1(Yes) and 0(No)
df$Class <- ifelse(df$Class == "Yes", 1, 0)

# Identify non-numeric columns (factor or character)
non.numeric <- sapply(df, function(x) is.factor(x) || is.character(x))
print(names(df)[non.numeric])

# checking for unique values in non-numeric columns
print(unique(df$RT))
# seeing that there is only 1 unique value in the RT column, we can disregard it
df <- subset(df, select = -RT)

print(unique(df$DIVISION))
# seeing that there is only 1 unique value in the DIVISION column, we can disregard it
df <- subset(df, select = -DIVISION)

# dropping administrative columns
admin.cols <- c('SERIALNO', 'SPORDER')
df <- df[, !(names(df) %in% admin.cols)]


# Identify numeric variables (numeric or integer)
numeric.columns <- sapply(df, function(x) is.numeric(x) || is.integer(x))
print(names(df)[numeric.columns])

# checking columns with only 1 unique value
one.unique.val <- sapply(df, function(x) length(unique(x)) == 1)
col.one.unique.val <- names(one.unique.val[one.unique.val])
# remove columns with only 1 unique value
df <- df[, !(names(df) %in% col.one.unique.val)]


# checking for 2 unique values
two.unique.val <- sapply(df, function(x) length(unique(x)) == 2)
col.two.unique.val <- names(two.unique.val[two.unique.val])
for (col in col.two.unique.val) {
  cat("Unique values in", col, ":", unique(df[[col]]), "\n")
}
# since these values are already in factor format, we can leave these columns alone

# checking for more than 2 unique values
more.unique.val <- sapply(df, function(x) length(unique(x)) > 2)
col.more.unique.val <- names(more.unique.val[more.unique.val])
for (col in col.more.unique.val) {
  cat("Unique values in", col, ":", unique(df[[col]]), "\n")
}

# identifying categorical columns
nominal.columns <- c('PUMA', 'CIT', 'COW', 'FER', 'GCL', 'GCR', 'HIMRKS', 'JWTRNS', 
                     'MAR', 'MARHD', 'MARHM', 'MARHT', 'MARHW', 'MIG', 'MIL', 'MLPA',
                     'MLPB', 'MLPCD', 'MLPE', 'MLPFG', 'MLPH', 'MLPIK', 'MLPJ',
                     'NWAB', 'NWAV', 'NWLA', 'NWLK', 'NWRE', 'SCH', 'WRK', 'ANC', 'ANC1P', 
                     'ANC1P', 'ESP', 'ESR', 'FOD1P', 'FOD2P', 'HISP', 'INDP', 'LANP', 'MIGPUMA', 
                     'MIGSP', 'MSP', 'NOP', 'OC', 'OCCP', 'PAOC', 'POBP', 'POVPIP', 'POWPUMA', 'POWSP', 
                     'QTRBIR', 'RAC1P', 'RAC2P', 'RAC3P', 'RACNUM', 'RC', 'SCIENGP',
                     'SCIENGRLP', 'SFN', 'SFR', 'VPS', 'WAOB')
interval.col <- c('PWGTP', 'CITWP', 'MARHYP', 'YOEP', 'DECADE', 'JWAP')
ordinal.columns <- c('ENG', 'GCM', 'SCHG', 'SCHL')
ratio.col <- c('INTP', 'JWMNP', 'OIP', 'PAP', 'RETP', 'SEMP', 
               'SSIP', 'SSP', 'WAGP', 'WKHP', 'WKWN', 'DRIVESP', 'PERNP', 'PINCP')

# checking for NAs in ratio columns
na.ratio.col <- sapply(df[ratio.col], function(x) any(is.na(x)))
ratio.col.with.na <- names(na.ratio.col)[na.ratio.col]
print(ratio.col.with.na)

# since all of the NA is the columns are due to actual unemployment, we can replace the NAs with 0
df<- df %>%
  mutate(across(c(JWMNP, WKHP, WKWN, DRIVESP, PERNP), ~ ifelse(is.na(.), 0, .)))

# checking for NAs in ordinal columns
na.ordinal.col <- sapply(df[ordinal.columns], function(x) any(is.na(x)))
ordinal.col.with.na <- names(na.ordinal.col)[na.ordinal.col]
print(ordinal.col.with.na)

# since NAs are meaningful in these column, we can replace all NA values with 0
df$ENG[is.na(df$ENG)] <- 0
df$GCM[is.na(df$GCM)] <- 0
df$SCHG[is.na(df$SCHG)] <- 0

# checking for NAs in interval columns
na.interval.col <- sapply(df[interval.col], function(x) any(is.na(x)))
interval.col.with.na <- names(na.interval.col)[na.interval.col]
print(interval.col.with.na)

# since NAs are meaningful in CITWP, MARHYP, YOEP, DECADE, JWAP, we replace NAs with 0
df[interval.col.with.na] <- lapply(df[interval.col.with.na], function(x) {
  x[is.na(x)] <- 0
  return(x)
})


## Nominal columns
# Count missing values in nominal columns
nominal_na_count <- colSums(is.na(df[nominal.columns]))

# Compute percentage of missing values
missing_nominal_percent <- (nominal_na_count / nrow(df)) * 100

# Create a summary dataframe
missing_summary <- data.frame(
  Column = names(nominal_na_count),
  Missing_Count = nominal_na_count,
  Missing_Percentage = round(missing_nominal_percent, 2)
)

# Filter only columns with missing values
missing_summary <- missing_summary[missing_summary$Missing_Count > 0, ]

# Sort by percentage of missing values in descending order
(missing_summary <- missing_summary[order(-missing_summary$Missing_Percentage), ])

# SFN and SFR refers to the subfamily number and relationship (probably for administrative purposes), 
# its inclusion would not add any significant value to the model. We can proceed to remove it
nominal.admin.col <- c('SFN', 'SFR')
df <- df[, !(names(df) %in% nominal.admin.col)]


#Since all the  columns about serving at certain period have simmilar meaning we can combine them into one served or not
#MLPA - VPS

military_columns <- c("MLPA", "MLPB", "MLPCD", "MLPE", "MLPFG", "MLPH", "MLPIK", "MLPJ")

# Replace NA with 0 in the selected columns because it means that person has not served 
df[military_columns][is.na(df[military_columns])] <- 0

# Create the "Served" column (1 if served in any period, 0 otherwise)
df$military.served <- ifelse(rowSums(df[military_columns] == "1") > 0, 1, 0)


# Remove military service columns and VPS column
df <- df[, !(names(df) %in% c(military_columns, "VPS"))]

# Check the new column
table(df$military.served)
barplot(table(df$military.served), main='Persons who had active military duty')

#Drop the columns more than 90% in nominal columns
na_threshold <- 90
cols_drop <- missing_summary$Column[missing_summary$Missing_Percentage > na_threshold]
# Drop these columns
df <- df[, !(names(df) %in% cols_drop)]

###HANDLING COLLINEARITY###

# creating a separate dataset for collinearity processing
df_numeric_clean <- df
# Remove the target variable (Class) before collinearity processing

class_col <- df_numeric_clean$Class  # Store Class separately
df_numeric_clean$Class <- NULL  # Remove Class for correlation analysis

df_numeric_clean <- df_numeric_clean[, sapply(df_numeric_clean, is.numeric)]
df_numeric_clean[] <- lapply(df_numeric_clean, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))


# Checked and handled missing values
if (any(is.na(df_numeric_clean))) {
  cat("Imputing missing values with median...\n")
  df_numeric_clean[] <- lapply(df_numeric_clean, function(x) ifelse(is.na(x), median(x, na.rm = TRUE), x))
}

## visualising collinearity between columns
cor_matrix <- cor(df_numeric_clean, use = "pairwise.complete.obs")
cor_melted <- melt(cor_matrix)
ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, 
                       limits = c(-1, 1), name = "Correlation") + 
  theme_minimal() +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 4),
        axis.text.y = element_text(size = 4)) + 
  labs(title = "Correlation Heatmap before removing highly correlated columns", x = "variable", y = "variable")


# Function to remove highly correlated variables
remove_collinear <- function(df, threshold = 0.7) {
  cor_matrix <- cor(df, use = "pairwise.complete.obs")  # Used pairwise complete obs to handle missing data
  results <- data.frame(Variable1 = character(), Variable2 = character(), 
                        Correlation = numeric(), Removed_Variable = character())
  
  while(TRUE) {
    abs_cor <- abs(cor_matrix)
    diag(abs_cor) <- 0  # Ignore diagonal
    max_cor <- max(abs_cor, na.rm = TRUE)
    if (max_cor < threshold) break  # Stop when no correlation exceeds threshold
    
    # Identify the most correlated pair
    pair <- which(abs_cor == max_cor, arr.ind = TRUE)[1, ]
    var_A <- rownames(cor_matrix)[pair[1]]
    var_B <- colnames(cor_matrix)[pair[2]]
    
    # Compute average correlation for both variables
    avg_A <- mean(abs(cor_matrix[var_A, ]), na.rm = TRUE)
    avg_B <- mean(abs(cor_matrix[var_B, ]), na.rm = TRUE)
    
    # Determine variable to remove (remove the one with higher average correlation)
    remove_var <- ifelse(avg_A > avg_B, var_A, var_B)
    # Store removal information
    results <- rbind(results, data.frame(Variable1 = var_A, Variable2 = var_B, 
                                         Correlation = round(max_cor, 3), 
                                         Removed_Variable = remove_var))
    # Drop the selected variable from the dataset
    df <- df[, !names(df) %in% remove_var]
    cor_matrix <- cor(df, use = "pairwise.complete.obs")
  }
  
  # Print removal summary
  print(results)
  print(nrow(results))
  
  # Cleaned dataset
  return(df)
}


# Should apply to cloned data frame
# Applied collinearity removal function
df_numeric_clean <- remove_collinear(df_numeric_clean, threshold = 0.7)


## visualising collinearity between columns
cor_matrix <- cor(df_numeric_clean, use = "pairwise.complete.obs")
cor_melted <- melt(cor_matrix)
ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile() + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, 
                       limits = c(-1, 1), name = "Correlation") + 
  theme_minimal() +  
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 4),
        axis.text.y = element_text(size = 4)) + 
  labs(title = "Correlation Heatmap after removing highly correlated columns", x = "variable", y = "variable")


# Merge back the target variable (Class)
df_numeric_clean <- cbind(df_numeric_clean, Class = class_col)
write.csv(df_numeric_clean, "cs699_classproject_cleaned.csv", row.names=FALSE)

####Model Training####

set.seed(123)

df <- read.csv("cs699_classproject_cleaned.csv")

# Separate features and target
X <- df %>% select(-Class)
y <- as.factor(df$Class)  

trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Function to apply SMOTE & ROSE
balance_data <- function(X_train, y_train, method = "SMOTE") {
  train <- cbind(X_train, Class = y_train)
  train[] <- lapply(train, function(x) if (is.factor(x)) as.numeric(as.character(x)) else as.numeric(x))
  
  if (method == "SMOTE") {
    data_balanced <- SMOTE(train, train$Class, K = 5, dup_size = 12)$data
    X_train_balanced <- data_balanced[, colnames(data_balanced) != "class"]
    y_train_balanced <- factor(data_balanced$class, levels = unique(y_train))
  } else if (method == "ROSE") {
    data_balanced <- ROSE(Class ~ ., data = train)$data
    X_train_balanced <- data_balanced[, colnames(data_balanced) != "Class"]
    y_train_balanced <- factor(data_balanced$Class, levels = unique(y_train))
  }
  
  list(X_train_balanced = X_train_balanced, y_train_balanced = y_train_balanced)
}

# Feature selection methods
apply_feature_selection <- function(X_train, y_train, method) {
  train_data <- cbind(X_train, Class = y_train)
  
  if (method == "Information Gain") {
    info_gain <- information.gain(Class ~ ., train_data)
    ranked_features <- info_gain[order(-info_gain$attr_importance), , drop = FALSE]
    selected_features <- rownames(ranked_features)[1:20]
  } else if (method == "Chi-Square") {
    chi_sq <- chi.squared(Class ~ ., train_data)
    ranked_features <- chi_sq[order(-chi_sq$attr_importance), , drop = FALSE]
    selected_features <- rownames(ranked_features)[1:20]
  }else if (method == "PCA") {
    pca_model <- prcomp(X_train, scale. = TRUE)
    variance_explained <- cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2))
    
    # Select enough components to explain 95% of the variance
    num_components <- which(variance_explained >= 0.95)[1]
    X_train_selected <- as.data.frame(pca_model$x[, 1:num_components])
    X_test_selected <- as.data.frame(predict(pca_model, X_test)[, 1:num_components])
    
    return(list(X_train_selected = X_train_selected, X_test_selected = X_test_selected))
  } 
  else if (method == "PCA_smote") {
    X_train_pca <- X_train %>% select(-Class)
    pca_model <- prcomp(X_train_pca, scale. = TRUE)
    variance_explained <- cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2))
    
    # Select enough components to explain 95% of the variance
    num_components <- which(variance_explained >= 0.95)[1]
    X_train_selected <- as.data.frame(pca_model$x[, 1:num_components])
    X_test_selected <- as.data.frame(predict(pca_model, X_test)[, 1:num_components])
    
    return(list(X_train_selected = X_train_selected, X_test_selected = X_test_selected))
  }
  
  
  X_train_selected <- X_train[, selected_features, drop = FALSE]
  X_test_selected <- X_test[, selected_features, drop = FALSE]
  
  list(X_train_selected = X_train_selected, X_test_selected = X_test_selected)
}

# MCC Calculation Function
calculate_mcc <- function(TP, FP, TN, FN) {
  TP <- as.numeric(TP)
  FP <- as.numeric(FP)
  TN <- as.numeric(TN)
  FN <- as.numeric(FN)
  
  numerator <- (TP * TN) - (FP * FN)
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  
  if (is.na(denominator) || denominator == 0) {
    return(NA)
  } else {
    return(round(numerator / denominator, 2))
  }
}

# Model Evaluation
evaluate_model <- function(pred, y_test, model_name, pred_probs = NULL) {
  cm <- confusionMatrix(factor(pred, levels = levels(y_test)), y_test)
  cat(paste0("\n=== Model Evaluation: ", model_name, " ===\n"))
  cat("Confusion Matrix:\n")
  print(cm$table)
  
  # For binary classification, compute per-class TPR and TNR manually
  cm_table <- cm$table
  classes <- levels(y_test)
  metrics <- data.frame(Class = classes, TPR = NA, FPR = NA, Precision = NA, Recall = NA, 
                        F1 = NA, ROC = NA, MCC = NA, Kappa = NA, stringsAsFactors = FALSE)
  
  all_TP <- 0
  all_FP <- 0
  all_TN <- 0
  all_FN <- 0
  total_samples <- sum(cm$table)
  
  for (i in seq_along(classes)) {
    class_i <- classes[i]
    TP <- cm_table[class_i, class_i]
    FN <- sum(cm_table[, class_i]) - TP
    FP <- sum(cm_table[class_i, ]) - TP
    TN <- sum(cm_table) - TP - FN - FP
    
    # Store sums for weighted average calculation
    all_TP <- all_TP + TP
    all_FP <- all_FP + FP
    all_TN <- all_TN + TN
    all_FN <- all_FN + FN
    
    TPR <- round(ifelse((TP + FN) == 0, 0, TP / (TP + FN)), 2)
    FPR <- round(ifelse((FP + TN) == 0, 0, FP / (FP + TN)), 2)
    Precision <- round(ifelse((TP + FP) == 0, 0, TP / (TP + FP)), 2)
    Recall <- TPR
    F1 <- round(ifelse((Precision + Recall) == 0, 0, 2 * (Precision * Recall) / (Precision + Recall)), 2)
    
    # ROC Calculation using the roc() function with quiet = TRUE
    if (!is.null(pred_probs)) {
      roc_curve <- roc(as.numeric(y_test) - 1, as.numeric(pred_probs), quiet = TRUE)
      roc_auc <- round(auc(roc_curve), 2)
    } else {
      roc_auc <- NA
    }
    
    # Manually Calculate MCC
    MCC <- calculate_mcc(TP, FP, TN, FN)
    Kappa <- round(cm$overall["Kappa"], 2)
    
    metrics[i, ] <- c(class_i, TPR, FPR, Precision, Recall, F1, roc_auc, MCC, Kappa)
  }
  
  # Calculate Weighted Averages
  wt_TPR <- round(all_TP / (all_TP + all_FN), 2)
  wt_FPR <- round(all_FP / (all_FP + all_TN), 2)
  wt_Precision <- round(all_TP / (all_TP + all_FP), 2)
  wt_Recall <- wt_TPR
  wt_F1 <- round(ifelse((wt_Precision + wt_Recall) == 0, 0, 2 * (wt_Precision * wt_Recall) / (wt_Precision + wt_Recall)), 2)
  
  # Suppressing messages when calculating weighted average ROC
  if (!is.null(pred_probs)) {
    roc_curve <- roc(as.numeric(y_test) - 1, pred_probs, quiet = TRUE)
    wt_ROC <- round(auc(roc_curve), 2)
  } else {
    wt_ROC <- NA
  }
  
  # Manually Calculate Weighted MCC
  wt_MCC <- calculate_mcc(all_TP, all_FP, all_TN, all_FN)
  
  # Calculate Weighted Kappa by averaging individual Kappas weighted by the number of samples in each class
  kappa_values <- as.numeric(metrics$Kappa[!is.na(metrics$Kappa)])
  kappa_weights <- as.numeric(rowSums(cm$table)) / total_samples
  wt_Kappa <- round(sum(kappa_values * kappa_weights, na.rm = TRUE), 2)
  
  metrics <- rbind(metrics, data.frame(Class = "Weighted Average", 
                                       TPR = wt_TPR, FPR = wt_FPR, Precision = wt_Precision, 
                                       Recall = wt_Recall, F1 = wt_F1, 
                                       ROC = wt_ROC, MCC = wt_MCC, Kappa = wt_Kappa))
  
  cat("\nPerformance Measure Table:\n")
  print(metrics)
}

# Standard run_combination for models that use balancing
run_combination <- function(balancing_method, feature_selection_method, model_name) {
  # Apply the chosen balancing method
  balanced_data <- balance_data(X_train, y_train, method = balancing_method)
  X_train_balanced <- balanced_data$X_train_balanced
  y_train_balanced <- balanced_data$y_train_balanced
  
  # Apply feature selection method on the balanced data
  selected_features <- apply_feature_selection(X_train_balanced, y_train_balanced, method = feature_selection_method)
  X_train_selected <- selected_features$X_train_selected
  X_test_selected <- selected_features$X_test_selected
  
  # Print the combination being run
  cat(paste0("\nRunning: ", balancing_method, " + ", feature_selection_method, " + ", model_name, "\n"))
  
  if (model_name == "Bagging") {
    model <- bagging(
      Class ~ ., 
      data = cbind(X_train_selected, Class = y_train_balanced), 
      nbagg = 60,
      control = rpart.control(maxdepth = 3)
    )
    pred_probs_df <- predict(model, newdata = X_test_selected, type = "prob")
    
    # Extract probabilities for the positive class '1'
    pred_probs <- as.numeric(pred_probs_df[, 2])
    
    # Generate predictions based on probability threshold
    pred <- ifelse(pred_probs > 0.6, 1, 0)
    
    
  } else if (model_name == "Logistic Regression") {
    model <- train(
      Class ~ ., 
      data = cbind(X_train_selected, Class = y_train_balanced),
      method = "glm",
      family = "binomial",
      trControl = trainControl(method = "none")
    )
    
    # Generate prediction probabilities (ensure it's a numeric vector)
    pred_probs_df <- predict(model, newdata = X_test_selected, type = "prob")
    pred_probs <- as.numeric(pred_probs_df[, 2])  # Extract probabilities for class '1'
    
    # Generate predictions based on probability threshold
    pred <- ifelse(pred_probs > 0.6, 1, 0)
    
  }else if (model_name == "XGBoost") {
    dtrain <- xgb.DMatrix(data = as.matrix(X_train_selected), label = as.numeric(y_train_balanced) - 1)
    dtest <- xgb.DMatrix(data = as.matrix(X_test_selected))
    
    params <- list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = 3,
      eta = 0.2,
      scale_pos_weight = 12
    )
    
    # Train the model directly without cross-validation
    model <- xgboost(
      data = dtrain,
      max_depth = params$max_depth,
      eta = params$eta,
      nrounds = 50,   
      objective = params$objective,
      scale_pos_weight = params$scale_pos_weight,
      verbose = 0
    )
    
    pred_probs <- predict(model, newdata = dtest)
    pred <- ifelse(pred_probs > 0.6, 1, 0)
  }
  else if (model_name == "LightGBM") {
    lgb_train <- lgb.Dataset(data = as.matrix(X_train_selected), label = as.numeric(y_train_balanced) - 1)
    lgb_test <- as.matrix(X_test_selected)
    
    lgb_params <- list(
      objective = "binary",
      metric = "auc",
      boosting = "gbdt",
      max_depth = 5,
      num_leaves = 12,
      learning_rate = 0.04,
      feature_fraction = 0.84,
      bagging_fraction = 0.75,
      bagging_freq = 2,
      scale_pos_weight = 14.06,
      lambda_l1=0.41,
      seed=123
    )
    
    model <- lgb.train(
      params = lgb_params,
      data = lgb_train,
      nrounds = 175,
      verbose = 0
    )
    pred_probs <- predict(model, lgb_test)
    pred <- ifelse(pred_probs > 0.535, 1, 0)
    
  } else if (model_name == "KNN") {
    preProc <- preProcess(X_train_selected, method = c("center", "scale"))
    X_train_scaled <- predict(preProc, X_train_selected)
    X_test_scaled <- predict(preProc, X_test_selected)
    
    model <- train(
      X_train_scaled, y_train_balanced,
      method = "knn",
      trControl = trainControl(method = "cv", number = 5),
      tuneGrid = expand.grid(k = 3:15)
    )
    pred_probs_df <- predict(model, X_test_scaled, type = "prob")
    pred_probs <- as.numeric(pred_probs_df[, 2])
    pred <- ifelse(pred_probs > 0.6, 1, 0)
    
    
  } else if (model_name == "SVM") {
    X_train_selected <- as.data.frame(lapply(X_train_selected, as.numeric))
    X_test_selected <- as.data.frame(lapply(X_test_selected, as.numeric))
    
    X_train_scaled <- scale(X_train_selected)
    X_test_scaled <- scale(X_test_selected)
    
    # Train the model with probability enabled
    model <- svm(
      x = X_train_scaled,
      y = y_train_balanced,
      kernel = "radial",
      cost = 1,
      gamma = 1 / ncol(X_train_scaled),
      probability = TRUE  # Enable probability estimation
    )
    
    pred_with_prob <- predict(model, X_test_scaled, probability = TRUE)
    pred_probs <- attr(pred_with_prob, "probabilities")[, "1"]
    pred <- ifelse(pred_probs > 0.6, 1, 0)
    
  }
  
  # Evaluate the model performance
  evaluate_model(pred, y_test, model_name, pred_probs = pred_probs)
}

# Separate function for XGBoost_Weighted 
run_xgb_weighted <- function(feature_selection_method) {
  # Apply feature selection on original (unbalanced) training data
  selected_features <- apply_feature_selection(X_train, y_train, method = feature_selection_method)
  X_train_selected <- selected_features$X_train_selected
  X_test_selected <- selected_features$X_test_selected
  
  # Print the combination being run (balancing is ignored)
  cat(paste0("\n", feature_selection_method, " + XGBoost_Weighted (balancing ignored)\n"))
  
  dtrain <- xgb.DMatrix(data = as.matrix(X_train_selected), label = as.numeric(y_train) - 1)
  dtest <- xgb.DMatrix(data = as.matrix(X_test_selected), label = as.numeric(y_test) - 1)
  xgb_model <- xgboost(
    data = dtrain,
    max_depth = 3,
    eta = 0.2,
    nrounds = 76,
    objective = "binary:logistic",
    scale_pos_weight = 12.62,
    verbose = 0
  )
  xgb_pred_probs <- predict(xgb_model, newdata = dtest)
  xgb_pred <- ifelse(xgb_pred_probs > 0.5, 1, 0)
  evaluate_model(xgb_pred, y_test, "XGBoost_Weighted")
}

cat("\nSuccessfully integrated all models and feature selection methods.")

#--- Integrated Calls ---
# uncomment to run the model
# it is recommended to run the entire script to ensure that the datasets for each model are clean

# SMOTE + Information Gain
run_combination(balancing_method = "SMOTE", feature_selection_method = "Information Gain", model_name = "LightGBM")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "Information Gain", model_name = "Logistic Regression")
#run_combination(balancing_method = "SMOTE", feature_selection_method = "Information Gain", model_name = "Bagging")
#run_combination(balancing_method = "SMOTE", feature_selection_method = "Information Gain", model_name = "XGBoost")
#run_combination(balancing_method = "SMOTE", feature_selection_method = "Information Gain", model_name = "KNN")
#run_combination(balancing_method = "SMOTE", feature_selection_method = "Information Gain", model_name = "SVM")
#run_xgb_weighted("Information Gain")

# SMOTE + Chi-Square
# run_combination(balancing_method = "SMOTE", feature_selection_method = "Chi-Square", model_name = "Bagging")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "Chi-Square", model_name = "XGBoost")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "Chi-Square", model_name = "LightGBM")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "Chi-Square", model_name = "KNN")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "Chi-Square", model_name = "SVM")
# run_xgb_weighted("Chi-Square")

# SMOTE + PCA
# run_combination(balancing_method = "SMOTE", feature_selection_method = "PCA_smote", model_name = "Logistic Regression")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "PCA_smote", model_name = "Bagging")
run_combination(balancing_method = "SMOTE", feature_selection_method = "PCA_smote", model_name = "XGBoost")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "PCA_smote", model_name = "LightGBM")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "PCA_smote", model_name = "KNN")
# run_combination(balancing_method = "SMOTE", feature_selection_method = "PCA_smote", model_name = "SVM")
# run_xgb_weighted("PCA")

# ROSE + Information Gain
# run_combination(balancing_method = "ROSE", feature_selection_method = "Information Gain", model_name = "Logistic Regression")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Information Gain", model_name = "Bagging")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Information Gain", model_name = "XGBoost")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Information Gain", model_name = "LightGBM")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Information Gain", model_name = "KNN")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Information Gain", model_name = "SVM")
#run_xgb_weighted("Information Gain")

# ROSE + Chi-Square
# run_combination(balancing_method = "ROSE", feature_selection_method = "Chi-Square", model_name = "Logistic Regression")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Chi-Square", model_name = "Bagging")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Chi-Square", model_name = "XGBoost")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Chi-Square", model_name = "XGBoost_Weighted")  # New weighted version
# run_combination(balancing_method = "ROSE", feature_selection_method = "Chi-Square", model_name = "LightGBM")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Chi-Square", model_name = "KNN")
# run_combination(balancing_method = "ROSE", feature_selection_method = "Chi-Square", model_name = "SVM")
# 
# # ROSE + PCA
# run_combination(balancing_method = "ROSE", feature_selection_method = "PCA", model_name = "Logistic Regression")
# run_combination(balancing_method = "ROSE", feature_selection_method = "PCA", model_name = "Bagging")
# run_combination(balancing_method = "ROSE", feature_selection_method = "PCA", model_name = "XGBoost")
# run_combination(balancing_method = "ROSE", feature_selection_method = "PCA", model_name = "XGBoost_Weighted")  # New weighted version
# run_combination(balancing_method = "ROSE", feature_selection_method = "PCA", model_name = "LightGBM")
# run_combination(balancing_method = "ROSE", feature_selection_method = "PCA", model_name = "KNN")
# run_combination(balancing_method = "ROSE", feature_selection_method = "PCA", model_name = "SVM")
