## Loading required libraries
library(dplyr)
library(tidyr)
library(ggplot2)

## Loading the data
dat <- read.csv("../Data/fraudTrain.csv")

## Analysing the data

frauds.per.cat <- dat %>%
  group_by(category) %>%
  summarise(count.fraud = length(which(is_fraud == 1)),
            count.clean = length(which(is_fraud == 0)))

# Make sure the long format stacks Clean first (bottom), then Fraud (top)
frauds.per.cat.long <- frauds.per.cat %>%
  pivot_longer(cols = c(count.clean, count.fraud),  # order matters!
               names_to = "type",
               values_to = "count") %>%
  mutate(type = recode(type,
                       count.fraud = "Fraud",
                       count.clean = "Clean"),
         type = factor(type, levels = c("Clean", "Fraud")))  # fraud on top

# Plot
ggplot(frauds.per.cat.long, aes(x = category, y = count, fill = type)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Clean" = "green3", "Fraud" = "red2")) +
  labs(title = "Frauds per Category",
       x = "Category",
       y = "Transaction Count",
       fill = "Transaction Type") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

