#Lab 4
#Title: Binary Logistic Regression: Model Specification, Comparison, and Interpretation

#Part 1:Configure the project and examine the data 
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

### Counts and percentages for categorical variables
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

### Combine all four variables into one data frame
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


## Table 1: Descriptive statistics for age
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
  )

## Table 2: Counts and percentages for categorical variables
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

# Part 2: Specify competing binary logistic regression models
# 2.1. Prediction
# 2.2. Specify four nested binary logistic regression models
# Model 0: Intercept-only model
mod0 <- glmmTMB(
  Survived ~ 1,
  family = binomial(link = "logit"),
  data = titanic
)

# Model 1: Add sex (0 = male, 1 = female)
mod1 <- glmmTMB(
  Survived ~ Sex.Binary,
  family = binomial(link = "logit"),
  data = titanic
)

# Model 2: Add ticket class (reference = first class)
mod2 <- glmmTMB(
  Survived ~ Sex.Binary + Pclass,
  family = binomial(link = "logit"),
  data = titanic
)

# Model 3: Add age (in years)
mod3 <- glmmTMB(
  Survived ~ Sex.Binary + Pclass + Age,
  family = binomial(link = "logit"),
  data = titanic
)


# 2.3. Explore the fitted models
summary(mod0)
summary(mod1)
summary(mod2)
summary(mod3)

# Part 3: Compare model performance
## 3.1. Conduct nested-model likelihood-ratio tests
model_lrt <- anova(mod0, mod1, mod2, mod3)

# Organize the nested-model test results
nested_tests <- model_lrt |>
  as.data.frame() |>
  tibble::rownames_to_column(var = "Model") |>
  select(Model, `Chi Df`, Chisq, `Pr(>Chisq)`) |>
  rename(
    nested_df = `Chi Df`,
    nested_chisq = Chisq,
    nested_p = `Pr(>Chisq)`
  ) |>
  mutate(
    residual_df = c(
      df.residual(mod0),
      df.residual(mod1),
      df.residual(mod2),
      df.residual(mod3)
    )
  ) |>
  select(
    Model, residual_df, nested_df, nested_chisq, nested_p
  )

nested_tests

## 3.2. Extract model-level performance statistics
model_performance <- bind_rows(
  mod0 = broom.mixed::glance(mod0),
  mod1 = broom.mixed::glance(mod1),
  mod2 = broom.mixed::glance(mod2),
  mod3 = broom.mixed::glance(mod3),
  .id = "Model"
) |>
  select(Model, AIC, BIC, logLik, nobs) |>
  mutate(
    deviance = -2 * logLik
  ) |>
  select(Model, nobs, AIC, BIC, logLik, deviance)

model_performance

# 3.3. Combine model-performance statistics and nested-model tests
model_comparison <- model_performance |>
  select(Model, AIC, BIC, logLik, deviance) |>
  left_join(
    nested_tests,
    by = "Model"
  )

## Create the model-comparison table
model_comparison |>
  gt() |>
  tab_header(
    title = "Comparison of Binary Logistic Regression Models",
    subtitle = paste("Complete-case sample: N =", nrow(titanic))
  ) |>
  cols_label(
    Model = "Model",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    residual_df = "Residual df",
    nested_df = "Nested df",
    nested_chisq = "Nested χ²",
    nested_p = "Nested p"
  ) |>
  fmt_number(
    columns = c(AIC, BIC, logLik, deviance, nested_chisq),
    decimals = 2
  ) |>
  fmt_integer(
    columns = c(residual_df, nested_df)
  ) |>
  fmt(
    columns = nested_p,
    fns = function(x) {
      ifelse(
        is.na(x),
        NA_character_,
        ifelse(x < .001, "< .001", sprintf("%.3f", x))
      )
    }
  ) |>
  sub_missing(
    columns = c(nested_df, nested_chisq, nested_p),
    missing_text = "—"
  ) |>
  cols_align(
    align = "left",
    columns = Model
  ) |>
  cols_align(
    align = "center",
    columns = -Model
  ) |>
  tab_source_note(
    source_note = paste(
      "mod0: Intercept only;",
      "mod1: Sex;",
      "mod2: Sex + ticket class;",
      "mod3: Sex + ticket class + age."
    )
  ) 

# Part 4: Evaluate binary logistic regression diagnostics
## 4.1. Examine the assumptions
candidate_model <- mod3

# Generate reproducible simulated residuals
sim_res <- DHARMa::simulateResiduals(
  fittedModel = candidate_model,
  n = 1000,
  seed = 9041
)

# Overall diagnostic plots
plot(sim_res)

# Examine residual patterns across age
DHARMa::plotResiduals(
  sim_res,
  form = titanic$Age
)

# Examine residual patterns across categorical predictors
DHARMa::plotResiduals(
  sim_res,
  form = factor(titanic$Sex.Binary)
)

DHARMa::plotResiduals(
  sim_res,
  form = titanic$Pclass
)

# Part 5: Select and report your preferred model
## 5.1. Select a model

## 5.2. Extract coefficients and 95% confidence intervals
coefficient_results <- broom.mixed::tidy(
  preferred_model,
  effects = "fixed",
  component = "cond",
  conf.int = TRUE,
  conf.level = 0.95,
  conf.method = "Wald"
)

# Create the coefficient table
coefficient_results |>
  mutate(
    predictor = recode(
      term,
      "(Intercept)" = "Intercept",
      "Sex.Binary" = "Female (vs. male)",
      "Pclass2" = "Second class (vs. first class)",
      "Pclass3" = "Third class (vs. first class)",
      "Age" = "Age (years)"
    ),
    ci_95 = sprintf(
      "[%.3f, %.3f]",
      conf.low,
      conf.high
    )
  ) |>
  select(
    predictor, estimate, std.error, ci_95, p.value
  ) |>
  gt() |>
  tab_header(
    title = "Coefficients for the Preferred Logistic Regression Model",
    subtitle = "mod3: Sex + ticket class + age"
  ) |>
  cols_label(
    predictor = "Predictor",
    estimate = "b",
    std.error = "SE",
    ci_95 = "95% CI",
    p.value = "p"
  ) |>
  fmt_number(
    columns = c(estimate, std.error),
    decimals = 3
  ) |>
  fmt(
    columns = p.value,
    fns = function(x) {
      ifelse(
        is.na(x),
        NA_character_,
        ifelse(x < .001, "< .001", sprintf("%.3f", x))
      )
    }
  ) |>
  cols_align(
    align = "left",
    columns = predictor
  ) |>
  cols_align(
    align = "center",
    columns = -predictor
  ) |>
  tab_source_note(
    source_note = paste(
      "b = unstandardized log-odds coefficient;",
      "SE = standard error; CI = confidence interval.",
      "Confidence intervals are Wald intervals.",
      "Reference categories are male and first class."
    )
  )


## 5.3. Calculate odds ratios and their confidence intervals
coefficient_results <- coefficient_results |>
  mutate(
    odds_ratio = exp(estimate),
    or_low = exp(conf.low),
    or_high = exp(conf.high)
  )

sex_result <- coefficient_results |>
  filter(term == "Sex.Binary")

class2_result <- coefficient_results |>
  filter(term == "Pclass2")

class3_result <- coefficient_results |>
  filter(term == "Pclass3")

age_result <- coefficient_results |>
  filter(term == "Age")

## 5.4. Overall model performance
### Compare the preferred model with the null model
overall_lrt <- anova(mod0, preferred_model) |>
  as.data.frame()

### Extract overall model statistics
preferred_performance <- broom.mixed::glance(preferred_model) |>
  transmute(
    N = nobs,
    AIC,
    BIC,
    logLik,
    deviance = -2 * logLik,
    residual_df = df.residual(preferred_model),
    model_chisq = overall_lrt$Chisq[2],
    model_df = overall_lrt[["Chi Df"]][2],
    model_p = overall_lrt[["Pr(>Chisq)"]][2]
  )

### Format the performance table
preferred_performance |>
  gt() |>
  tab_header(
    title = "Overall Performance of the Preferred Model",
    subtitle = "mod3: Sex + ticket class + age"
  ) |>
  cols_label(
    N = "N",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    residual_df = "Residual df",
    model_chisq = "Model χ²",
    model_df = "Model df",
    model_p = "p"
  ) |>
  fmt_integer(
    columns = c(N, residual_df, model_df)
  ) |>
  fmt_number(
    columns = c(AIC, BIC, logLik, deviance, model_chisq),
    decimals = 2
  ) |>
  fmt(
    columns = model_p,
    fns = function(x) {
      ifelse(x < .001, "< .001", sprintf("%.3f", x))
    }
  ) |>
  tab_source_note(
    source_note = paste(
      "The likelihood-ratio χ² test compares the preferred",
      "model with the intercept-only model."
    )
  )

## 5.5. Overarching summary

## 5.6. Visualize age and predicted survival probability
# Generate predicted survival probabilities across age
age_predictions <- ggeffects::ggpredict(
  preferred_model,
  terms = "Age [all]",
  condition = c(
    Sex.Binary = 0,
    Pclass = "1"
  )
) |>
  as.data.frame()

# Plot predicted probabilities and 95% confidence intervals
ggplot(age_predictions, aes(x = x, y = predicted)) +
  geom_ribbon(
    aes(ymin = conf.low, ymax = conf.high),
    fill = "#669BBC",
    alpha = 0.20
  ) +
  geom_line(
    color = "#2C5F7C",
    linewidth = 1
  ) +
  scale_y_continuous(
    labels = scales::label_percent(accuracy = 1),
    limits = c(0, 1)
  ) +
  labs(
    title = "Age and Predicted Survival Probability",
    subtitle = "Predictions for male passengers in first class",
    x = "Age (years)",
    y = "Predicted probability of survival",
    caption = "Shaded band represents the 95% confidence interval."
  ) +
  theme_minimal(base_size = 12)

# Part 6: Findings and Limitations

