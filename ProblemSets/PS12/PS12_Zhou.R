# ============================================================
# PS12_Zhou.R
# Econ 5253 - Data Science for Economists, Spring 2026
# Author: Yeyang Zhou
# ============================================================

library(sampleSelection)
library(tidyverse)
library(modelsummary)

# ── 1. Load data ─────────────────────────────────────────────
wagedata <- read.csv("wages12.csv")

# Keep numeric copies for probit / counterfactual (avoid factor naming issues)
wagedata$married_num <- wagedata$married
wagedata$kids_num    <- wagedata$kids
wagedata$union_num   <- wagedata$union
wagedata$college_num <- wagedata$college

# ── 2. Format factors (Q5) ───────────────────────────────────
wagedata <- wagedata %>%
  mutate(
    college = as.factor(college),
    married = as.factor(married),
    union   = as.factor(union)
  )

# ── 3. Summary table (Q6) ────────────────────────────────────
datasummary_skim(wagedata, output = "datasummary.tex")

missing_rate <- mean(is.na(wagedata$logwage))
cat(sprintf("Missing rate of logwage: %.4f (%.1f%%)\n",
            missing_rate, missing_rate * 100))

# ── 4. Imputation and regression (Q7) ────────────────────────

# (a) Complete cases (listwise deletion)
model_cc <- lm(logwage ~ hgc + union + college + exper + I(exper^2),
               data = wagedata)

# (b) Mean imputation
wagedata_mean <- wagedata %>%
  mutate(logwage = ifelse(is.na(logwage),
                          mean(logwage, na.rm = TRUE),
                          logwage))

model_mean <- lm(logwage ~ hgc + union + college + exper + I(exper^2),
                 data = wagedata_mean)

# (c) Heckman selection (Heckit)
wagedata_heck <- wagedata %>%
  mutate(
    valid   = as.integer(!is.na(logwage)),
    logwage = ifelse(is.na(logwage), 0, logwage)
  )

model_heck <- selection(
  selection = valid   ~ hgc + union + college + exper + married + kids,
  outcome   = logwage ~ hgc + union + college + exper + I(exper^2),
  data      = wagedata_heck,
  method    = "2step"
)

# Print betas to console
b_cc   <- coef(model_cc)["hgc"]
b_mean <- coef(model_mean)["hgc"]
b_heck <- coef(model_heck, part = "outcome")["hgc"]

cat(sprintf("\n=== Returns to schooling (beta1 on hgc) ===\n"))
cat(sprintf("Complete cases : %.4f\n", b_cc))
cat(sprintf("Mean imputation: %.4f\n", b_mean))
cat(sprintf("Heckman 2-step : %.4f\n", b_heck))
cat(sprintf("True value     : 0.0910\n"))

# Regression table.
# modelsummary chokes on selection objects because the selection equation
# and outcome equation share variable names. Wrap the outcome equation in
# a minimal S3 object so modelsummary sees only the outcome coefficients.
heck_cf <- coef(model_heck, part = "outcome")
heck_vc <- tryCatch(
  vcov(model_heck, part = "outcome"),
  error = function(e) {
    # Fallback: extract outcome block from full vcov matrix
    v   <- vcov(model_heck)
    nm  <- names(heck_cf)
    v[nm, nm, drop = FALSE]
  }
)
heck_se <- sqrt(diag(heck_vc))

heck_obj <- structure(
  list(cf = heck_cf, se = heck_se, n = nobs(model_heck)),
  class = "heck_outcome"
)

tidy.heck_outcome <- function(x, ...) {
  z <- x$cf / x$se
  tibble::tibble(
    term      = names(x$cf),
    estimate  = as.numeric(x$cf),
    std.error = as.numeric(x$se),
    statistic = as.numeric(z),
    p.value   = as.numeric(2 * pnorm(-abs(z)))
  )
}
glance.heck_outcome <- function(x, ...) tibble::tibble(Num.Obs. = x$n)

# Register S3 methods so modelsummary can dispatch on them
registerS3method("tidy",   "heck_outcome", tidy.heck_outcome)
registerS3method("glance", "heck_outcome", glance.heck_outcome)

modelsummary(
  list("Complete Cases"  = model_cc,
       "Mean Imputation" = model_mean,
       "Heckman"         = heck_obj),
  stars    = TRUE,
  gof_omit = "AIC|BIC|Log|F|RMSE",
  output   = "regtable.tex",
  title    = "Imputation Methods: Returns to Schooling"
)

# ── 5. Probit model (Q8) ──────────────────────────────────────
# Use numeric versions of married/kids/college to keep coefficient names
# predictable for the counterfactual step below.
probit <- glm(union_num ~ hgc + college_num + exper + married_num + kids_num,
              data   = wagedata,
              family = binomial(link = "probit"))

cat("\n=== Probit estimates ===\n")
print(summary(probit)$coefficients)

# ── 6. Counterfactual policy (Q9) ────────────────────────────
# Compute predicted probabilities by hand (more robust than mutating
# coef() and calling predict()).

X    <- model.matrix(probit)
beta <- coef(probit)

# Baseline
xb_base   <- as.numeric(X %*% beta)
pred_base <- pnorm(xb_base)

# Counterfactual: married_num and kids_num coefficients = 0
beta_cf                <- beta
beta_cf["married_num"] <- 0
beta_cf["kids_num"]    <- 0
xb_cf                  <- as.numeric(X %*% beta_cf)
pred_cf                <- pnorm(xb_cf)

cat(sprintf("\n=== Counterfactual policy results ===\n"))
cat(sprintf("Baseline avg P(union)        : %.4f\n", mean(pred_base)))
cat(sprintf("Counterfactual avg P(union)  : %.4f\n", mean(pred_cf)))
cat(sprintf("Difference (CF - Baseline)   : %.4f\n",
            mean(pred_cf) - mean(pred_base)))
