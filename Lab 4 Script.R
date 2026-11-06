##Lab 4

#1.1. Load the required packages
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("broom.mixed")
install.packages("performance")
install.packages("gt")
install.packages("ggeffects")
install.packages("glmmTBM")
install.packages("DHARMa")
install.packages("titanic")
install.packages("psych")

#1.2. Load the dataset
library(glmmTMB)
library(dplyr)
library(tidyr)
library(ggplot2)
library(broom.mixed)
library(performance)
library(gt)
library(readr)
library(DHARMa)
library(ggeffects)
library(titanic)
library(psych)

titanic <- titanic_train

#Keep the same complete-case sample for every model
titanic <- titanic |>
  select(Survived, Sex, Age, Pclass) |>
  drop_na() |>
  mutate(
    Sex.Binary = case_when(
      Sex == "male" ~ 0,
      Sex == "female" ~ 1)) |>
  mutate(
    Survived.Char = case_when(
      Survived == 0 ~ "Died",
      Survived == 1 ~ "Survived")) |>
  mutate(
    Pclass.Char = case_when(
      Pclass == "1" ~ "First Class",
      Pclass == "2" ~ "Second Class",
      Pclass == "3" ~ "Third Class")) |>
  mutate(Pclass = factor(Pclass, levels = c(1, 2, 3))) |>
  select(Survived, Survived.Char, Sex, Sex.Binary, Pclass, Pclass.Char, Age)

n <- nrow(titanic)
n

# 1.3. Explore the data
# Inspect the first several observations
head(titanic)
glimpse(titanic)
psych::describe(titanic$Age)

# 1.4. Explore the data
head(titanic)
glimpse(titanic)

# Descriptive statistics for age
age_stats <- psych::describe(titanic$Age)

age_desc <- tibble(
  variable = "Age",
  category = "Age (years)",
  n = as.integer(age_stats$n),
  proportion = NA_real_,
  mean = age_stats$mean,
  sd = age_stats$sd,
  min = age_stats$min,
  max = age_stats$max,
  skew = age_stats$skew,
  kurtosis = age_stats$kurtosis
)

# Counts and percentages for categorical variables
categorical_desc <- titanic |>
  transmute(
    Survived = Survived.Char,
    Sex = case_when(
      Sex == "male" ~ "Male",
      Sex == "female" ~ "Female"
    ),
    Pclass = Pclass.Char
  ) |>
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "category"
  ) |>
  count(variable, category, name = "n") |>
  group_by(variable) |>
  mutate(proportion = n / sum(n)) |>
  ungroup()

# Combine all four variables into one data frame
descriptive_stats <- bind_rows(
  categorical_desc,
  age_desc
) |>
  mutate(
    variable = factor(
      variable,
      levels = c("Survived", "Sex", "Age", "Pclass")
    )
  ) |>
  arrange(variable, category)

# Table 1: Descriptive statistics for age
age_desc |>
  select(category, n, mean, sd, min, max) |>
  gt() |>
  tab_header(
    title = "Descriptive Statistics for Age"
  ) |>
  cols_label(
    category = "Variable",
    n = "N",
    mean = "Mean",
    sd = "SD",
    min = "Min",
    max = "Max"
  ) |>
  fmt_integer(columns = n) |>
  fmt_number(
    columns = c(mean, sd, min, max),
    decimals = 2
  ) |>
  tab_source_note(
    source_note = "Age is measured in years."
  )

# Table 2: Counts and percentages for categorical variables
categorical_desc |>
  mutate(
    variable = factor(
      variable,
      levels = c("Survived", "Sex", "Pclass")
    )
  ) |>
  arrange(variable, category) |>
  gt(
    groupname_col = "variable",
    rowname_col = "category"
  ) |>
  tab_header(
    title = "Counts and Percentages of Categorical Variables",
    subtitle = paste("Complete-case sample: N =", nrow(titanic))
  ) |>
  tab_stubhead(label = "Variable / Category") |>
  cols_label(
    n = "N",
    proportion = "Percentage"
  ) |>
  fmt_integer(columns = n) |>
  fmt_percent(
    columns = proportion,
    decimals = 1
  )
