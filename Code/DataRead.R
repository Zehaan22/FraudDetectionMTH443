## Loading the required libraries
library(dplyr)
library(lubridate)
library(tidyr)
library(caret)
library(e1071)
library(cluster)
library(dbscan)
library(solitude)   # Isolation Forest
library(depmixS4)   # HMMs
library(igraph)
library(ggplot2)
library(nimble)     # MCMC implementation (bonus)


## Reading the data
df <- read.csv("../Data/fraudTrain.csv")

set.seed(123)

# Sample 10k observations with class balance
df_sample <- df %>%
  group_by(is_fraud) %>%
  sample_n(size = min(5000, n()), replace = FALSE) %>%
  ungroup()

# Preprocess this sample the same way
df_sample$trans_date_trans_time <- ymd_hms(df_sample$trans_date_trans_time)
df_sample$dob <- ymd(df_sample$dob)
df_sample$age <- as.integer(time_length(difftime(df_sample$trans_date_trans_time, df_sample$dob), "years"))
df_sample$hour <- hour(df_sample$trans_date_trans_time)
df_sample$distance <- sqrt((df_sample$lat - df_sample$merch_lat)^2 + (df_sample$long - df_sample$merch_long)^2)

features <- df_sample %>%
  dplyr::select(amt, age, city_pop, hour, distance) %>%
  drop_na()

features_scaled <- scale(features)

## Isolation Forest
iso <- isolationForest$new()
iso$fit(features)
df_sample$iso_score <- iso$predict(features)$anomaly_score

ggplot(df_sample, aes(x = iso_score, fill = as.factor(is_fraud))) +
  geom_density(alpha = 0.5) +
  labs(title = "Isolation Forest Anomaly Score by Class",
       x = "Anomaly Score", fill = "Is Fraud") +
  theme_minimal()

## One class SVM
svm_model <- svm(features_scaled, type = "one-classification", kernel = "radial", nu = 0.05)
df_sample$svm_pred <- predict(svm_model, features_scaled)

## DBSCAN
db <- dbscan(features_scaled, eps = 1.5, minPts = 10)
df_sample$dbscan_cluster <- db$cluster
df_sample$dbscan_anomaly <- ifelse(df_sample$dbscan_cluster == 0, 1, 0)

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
  list(amt ~ 1, hour ~ 1),
  data = features_scaled,
  nstates = 2,
  family = list(gaussian(), gaussian())
)

# Try fitting again
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
test_data <- read.csv("Data/fraudTest.csv")

# Feature engineering (same as training set)
test_data <- test_data %>%
  mutate(
    trans_date_trans_time = ymd_hms(trans_date_trans_time),
    hour = hour(trans_date_trans_time),
    age = year(trans_date_trans_time) - year(ymd(dob)),
    distance = sqrt((lat - merch_lat)^2 + (long - merch_long)^2)
  ) %>%
  select(amt, age, city_pop, hour, distance, is_fraud)



