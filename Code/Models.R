## Loading the required libraries
library(dplyr)
library(lubridate)
library(tidyr)
library(caret)
library(e1071)
library(cluster)
library(reshape2)
library(dbscan)
library(solitude)   # Isolation Forest
library(depmixS4)   # HMMs
library(igraph)
library(ggplot2)
library(nimble)     # MCMC implementation (bonus)
# Load required metrics function
library(caret)
library(pROC)

evaluate_model <- function(true_labels, pred_labels, model_name) {
  true_labels <- factor(true_labels, levels = c(0, 1))
  pred_labels <- factor(pred_labels, levels = c(0, 1))
  
  cm <- confusionMatrix(pred_labels, true_labels, positive = "1")
  roc_obj <- roc(as.numeric(true_labels), as.numeric(pred_labels))
  
  cat("\n", model_name, "\n")
  cat("Accuracy :", cm$overall["Accuracy"], "\n")
  cat("Precision:", cm$byClass["Precision"], "\n")
  cat("Recall   :", cm$byClass["Recall"], "\n")
  cat("F1 Score :", cm$byClass["F1"], "\n")
  cat("AUC      :", auc(roc_obj), "\n")
  cat("------------------------------------\n")
}
## Reading the data
df <- read.csv("./Data/fraudTrain.csv")


set.seed(123)

# Sample 10k observations with class balance
df_sample <- df %>%
  sample_n(size = min(10000, n()), replace = FALSE) %>%
  ungroup()

# Preprocess this sample the same way
df_sample$trans_date_trans_time <- ymd_hms(df_sample$trans_date_trans_time)
df_sample$dob <- ymd(df_sample$dob)
df_sample$age <- as.integer(time_length(difftime(df_sample$trans_date_trans_time, df_sample$dob), "years"))
df_sample$hour <- hour(df_sample$trans_date_trans_time)
df_sample$distance <- sqrt((df_sample$lat - df_sample$merch_lat)^2 + (df_sample$long - df_sample$merch_long)^2)

# Train-test split
set.seed(123)
train_idx <- createDataPartition(df_sample$is_fraud, p = 0.7, list = FALSE)
train_data <- df_sample[train_idx, ]
val_data <- df_sample[-train_idx, ]


features <- df_sample %>%
  dplyr::select(amt, age, city_pop, hour, distance) %>%
  drop_na()

features_scaled <- scale(features)

## Isolation Forest
iso_model <- isolationForest$new()
iso_model$fit(df_sample[, c("amt", "age", "city_pop", "hour", "distance")])
df_sample$iso_score <- iso_model$predict(features)$anomaly_score

ggplot(df_sample, aes(x = iso_score, fill = as.factor(is_fraud))) +
  geom_density(alpha = 0.5) +
  labs(title = "Isolation Forest Anomaly Score by Class",
       x = "Anomaly Score", fill = "Is Fraud") +
  theme_minimal()

#train-test accuracy


iso_model$fit(train_data[, c("amt", "age", "city_pop", "hour", "distance")])
val_iso_pred <- iso_model$predict(val_data[, c("amt", "age", "city_pop", "hour", "distance")])
val_data$iso_anomaly <- ifelse(val_iso_pred$anomaly_score > 0.65, 1, 0)

evaluate_model(val_data$is_fraud, val_data$iso_anomaly, "Isolation Forest")


## One class SVM
svm_model <- svm(features_scaled, type = "one-classification", kernel = "radial", nu = 0.99)
df_sample$svm_pred <- predict(svm_model, features_scaled)

svm_model <- svm(train_data[, c("amt", "age", "city_pop", "hour", "distance")],
                 type = "one-classification", kernel = "radial", nu = 0.99)



val_scaled <- scale(val_data[, c("amt", "age", "city_pop", "hour", "distance")],
                    center = attr(features_scaled, "scaled:center"),
                    scale = attr(features_scaled, "scaled:scale"))

val_data$svm_anomaly <- as.integer(predict(svm_model, val_scaled) == FALSE)

evaluate_model(val_data$is_fraud, val_data$svm_anomaly, "One-Class SVM")


## DBSCAN


library(dbscan)
library(caret)  # for confusionMatrix

# Grid to search over
eps_vals <- seq(0.5, 3, by = 0.2)
minPts_vals <- seq(5, 20, by = 5)

# Store results
results <- data.frame()

for (eps in eps_vals) {
  for (minPts in minPts_vals) {
    db <- dbscan(features_scaled, eps = eps, minPts = minPts)
    pred <- ifelse(db$cluster == 0, 1, 0)  # 1 = anomaly
    
    # Compute F1
    cm <- confusionMatrix(factor(pred), factor(df_sample$is_fraud), positive = "1")
    precision <- cm$byClass["Precision"]
    recall <- cm$byClass["Recall"]
    f1 <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
    
    # Store
    results <- rbind(results, data.frame(eps = eps, minPts = minPts, F1 = f1))
  }
}

# Best results
results <- results[order(-results$F1), ]
head(results)

db <- dbscan(features_scaled, eps = 1.7, minPts = 15)
df_sample$dbscan_cluster <- db$cluster
df_sample$dbscan_anomaly <- ifelse(df_sample$dbscan_cluster == 0, 1, 0)

evaluate_model(df_sample$is_fraud, df_sample$dbscan_anomaly, "DBSCAN")


###


library(reshape2)
detect_mat <- df_sample %>%
  dplyr::select(is_fraud, iso_score, svm_pred, dbscan_anomaly) %>%
  mutate(
    iso_flag = ifelse(iso_score > 0.65, 1, 0)  # You can tune this
  ) %>%
  dplyr::select(is_fraud, iso_flag, svm_pred, dbscan_anomaly)

detect_long <- melt(detect_mat, id.vars = "is_fraud")

ggplot(detect_long, aes(x = variable, fill = as.factor(value))) +
  geom_bar(position = "fill") +
  facet_wrap(~is_fraud, labeller = labeller(is_fraud = c("0" = "Not Fraud", "1" = "Fraud"))) +
  labs(y = "Proportion", fill = "Flagged", title = "Model Flags by Fraud Status")

## HMMs
features <- as.data.frame(lapply(features, function(x) ifelse(is.infinite(x), NA, x)))
features <- features[complete.cases(features), ]
apply(features, 2, sd)

features_scaled <- as.data.frame(scale(features))

hmm_model <- depmix(
  list(amt ~ 1, age ~ 1, city_pop ~ 1, hour ~ 1, distance ~ 1),
  data = features_scaled,
  nstates = 2,
  family = list(gaussian(), gaussian(), gaussian(), gaussian(), gaussian())
)

set.seed(123)
hmm_fit <- fit(hmm_model)
summary(hmm_fit)

posterior_probs <- posterior(hmm_fit)
df_sample$state <- posterior_probs$state

# Assign meaningful labels to the states based on average amount (as proxy for risk)
state_summary <- df_sample %>%
  group_by(state) %>%
  summarise(mean_amt = mean(amt, na.rm = TRUE)) %>%
  arrange(desc(mean_amt)) %>%
  mutate(risk_label = c("High-Risk", "Low-Risk"))

# Merge labels back to the data
df_sample <- df_sample %>%
  left_join(state_summary %>% dplyr::select(state, risk_label), by = "state")

# Plot with better labels and limits
ggplot(df_sample, aes(x = amt, fill = risk_label)) +
  geom_histogram(bins = 50, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = c("High-Risk" = "#e74c3c", "Low-Risk" = "#2ecc71")) +
  xlim(0, quantile(df_sample$amt, 0.99)) +
  labs(
    title = "Transaction Amount Distribution by HMM-Inferred Risk",
    x = "Transaction Amount",
    y = "Count",
    fill = "HMM-Inferred Risk"
  ) +
  theme_minimal(base_size = 14)


## attempt to label and benchmark

# Assign state to train and val separately
hmm_train <- train_data %>% dplyr::select(amt, age, city_pop, hour, distance) %>% scale()
hmm_model <- depmix(list(amt ~ 1, age ~ 1, city_pop ~ 1, hour ~ 1, distance ~ 1),
                    data = as.data.frame(hmm_train), nstates = 2,
                    family = list(gaussian(), gaussian(), gaussian(), gaussian(), gaussian()))

set.seed(123)
hmm_fit <- fit(hmm_model)
train_post <- posterior(hmm_fit)
train_data$state <- train_post$state

# Predict states for validation
hmm_val <- val_data %>% dplyr::select(amt, age, city_pop, hour, distance) %>% scale()
val_post <- posterior(hmm_fit, newdata = as.data.frame(hmm_val))
val_data$state <- val_post$state

val_summary <- val_data %>%
  group_by(state) %>%
  summarise(mean_amt = mean(amt, na.rm = TRUE)) %>%
  arrange(desc(mean_amt)) %>%
  mutate(risk = c("High", "Low"))

val_data <- val_data %>%
  left_join(val_summary %>% dplyr::select(state, risk), by = "state")

cat("HMM State High-Risk Fraud Rate:\n")
print(table(val_data$is_fraud, val_data$risk))



### Tree based Detection
# Sample smaller subset for speed
df_graph <- df_sample %>%
  dplyr::select(cc_num, merchant, trans_date_trans_time, is_fraud) %>%
  mutate(trans_date = trans_date_trans_time) %>%
  arrange(trans_date) %>%
  distinct()

# Build edge list: card numbers using the same merchant
edge_list <- df_graph %>%
  group_by(merchant) %>%
  filter(n() > 1) %>%  # Only merchants with >1 transaction
  summarise(pairs = combn(cc_num, 2, simplify = FALSE)) %>%
  unnest(cols = pairs) %>%
  mutate(
    from = sapply(pairs, `[`, 1),  # Extract first element of the pair
    to = sapply(pairs, `[`, 2)     # Extract second element of the pair
  ) %>%
  dplyr::select(from, to)

# Create the graph
g <- graph_from_data_frame(edge_list, directed = FALSE)

# Add node attributes
V(g)$fraud <- ifelse(V(g)$name %in% df_graph$cc_num[df_graph$is_fraud == 1], 1, 0)

# Detect communities
communities <- cluster_louvain(g)
V(g)$community <- communities$membership

# Visualize with fraud coloring
plot(
  g,
  vertex.label = NA,
  vertex.size = 3,
  vertex.color = ifelse(V(g)$fraud == 1, "red", "lightblue"),
  main = "Graph-Based Fraud Network",
  layout = layout_with_fr(g)
)

###############################################################################
#### Testing Data
###############################################################################
# Load test data
test_data <- read.csv("../Data/fraudTest.csv")

# Feature engineering (same as training set)
test_data <- test_data %>%
  mutate(
    trans_date_trans_time = ymd_hms(trans_date_trans_time),
    hour = hour(trans_date_trans_time),
    age = year(trans_date_trans_time) - year(ymd(dob)),
    distance = sqrt((lat - merch_lat)^2 + (long - merch_long)^2)
  ) %>%
  dplyr::select(amt, age, city_pop, hour, distance, is_fraud)

## Isolation Forest
iso_pred <- iso_model$predict(test_data[, c("amt", "age", "city_pop", "hour", "distance")])
# Add predictions to your test data
test_data$iso_score <- iso_pred$anomaly_score
test_data$iso_anomaly <- ifelse(test_data$iso_score > 0.65, 1, 0)  # You can change threshold

## One class SVM
test_data$svm_anomaly <- as.integer(predict(svm_model, test_data[, 1:5]) == FALSE)

## DBSCAN
db_test <- dbscan(scale(test_data[, 1:5]), eps = 0.5, minPts = 5)
test_data$dbscan_anomaly <- as.integer(db_test$cluster == 0)

## Visualisations
detect_mat <- test_data %>%
  dplyr::select(is_fraud, iso_anomaly, svm_anomaly, dbscan_anomaly) %>%
  mutate(across(everything(), as.factor))

melted <- melt(detect_mat, id.vars = "is_fraud")

## Prediction Comp
ggplot(melted, aes(x = value, fill = is_fraud)) +
  geom_bar(position = "dodge") +
  facet_wrap(~variable) +
  labs(
    title = "Anomaly Model Predictions vs Actual Fraud Labels",
    x = "Predicted as Anomaly?",
    y = "Count",
    fill = "True Fraud"
  ) +
  scale_x_discrete(labels = c("0" = "Legit", "1" = "Anomaly")) +
  theme_minimal()

## Confusion Matrix
confusionMatrix(
  factor(test_data$iso_anomaly),
  factor(test_data$is_fraud),
  positive = "1"
)

## ROC
library(pROC)

roc_iso <- roc(test_data$is_fraud, test_data$iso_anomaly)
roc_svm <- roc(test_data$is_fraud, test_data$svm_anomaly)

plot(roc_iso, col = "blue", main = "ROC Curve for Anomaly Detectors")
lines(roc_svm, col = "red")
legend("bottomright", legend = c("Isolation Forest", "One-Class SVM"), col = c("blue", "red"), lty = 1)

write.csv(test_data, "../Data/fraudTest_predictions.csv", row.names = FALSE)