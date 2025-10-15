# install.packages(c("forecast","sandwich","lmtest","car","readr","dplyr","zoo"))
library(forecast)
library(sandwich)
library(lmtest)
library(car)
library(readr)
library(dplyr)
library(zoo)

## --- Paths
home_dir   <- path.expand("~")
paper_path <- file.path(home_dir, "Documents/github/neoval-project")
data_path  <- file.path(paper_path, "data")

## --- Helper: AICc
aicc <- function(aic, nobs, k) {
  aic + (2 * k * (k + 1)) / max(nobs - k - 1, 1)
}

## --- Fit ARIMAX in levels (no intercept)
fit_arimax <- function(y, X, order=c(1,0,0)) {
  # y: numeric vector; X: matrix/data.frame (same rows), order=(p,0,q)
  fit <- Arima(y, order=order, xreg=as.matrix(X),
               include.mean = FALSE, method = "ML")
  return(fit)
}

## --- Ljung–Box p-value at a set of lags
lb_p <- function(resid, lag=12) {
  # Using Ljung-Box; 'fitdf' adjustment can be debated when xreg present; we keep it simple
  as.numeric(Box.test(resid, lag=lag, type="Ljung-Box")$p.value)
}

## --- Expanded (p,0,q) grid like your Python
search_arimax_levels <- function(y, X) {
  cand <- list(c(0,0,0), c(1,0,0), c(0,0,1), c(1,0,1),
               c(2,0,0), c(0,0,2), c(2,0,1), c(1,0,2), c(2,0,2))
  out <- list()
  for (od in cand) {
    ok <- try({
      fit <- fit_arimax(y, X, order=od)
      k   <- length(coef(fit))                 # number of estimated params
      aic <- AIC(fit)
      aic_c <- aicc(aic, nobs=length(y), k=k)
      p12 <- lb_p(residuals(fit), lag=12)
      out[[paste(od, collapse=",")]] <- list(order=od, aicc=aic_c, lb_p12=p12, fit=fit)
    }, silent=TRUE)
  }
  # Sort by AICc then by LB p (desc)
  res <- do.call(rbind, lapply(out, function(z) data.frame(
    order=paste(z$order, collapse=","),
    aicc=z$aicc, lb_p12=z$lb_p12
  )))
  res <- res[order(res$aicc, -res$lb_p12), , drop=FALSE]
  list(table=res, winner=out[[rownames(res)[1]]])
}

## --- OLS + HAC (Newey–West) and VIF
ols_hac_vif <- function(y, X, max_lags=12) {
  df <- data.frame(y=y, X)
  fm <- as.formula(paste("y ~", paste(colnames(X), collapse=" + ")))
  ols <- lm(fm, data=df)
  hac <- coeftest(ols, vcov.=NeweyWest(ols, lag=max_lags, prewhite=FALSE))
  # VIF (add constant handled internally)
  vif_df <- data.frame(variable=names(coef(ols))[-1], VIF = vif(ols))
  list(ols=ols, hac=hac, vif=vif_df)
}

## --- Build ARDL(1) X (adds lag-1 of each factor)
build_ardl1 <- function(X, y) {
  X_lag <- dplyr::mutate_all(as.data.frame(X), ~dplyr::lag(., 1))
  colnames(X_lag) <- paste0(colnames(X), "_lag1")
  D <- cbind(y=y, X, X_lag)
  D <- D[complete.cases(D), ]
  list(y=D[, "y"], X=D[, setdiff(colnames(D), "y"), drop=FALSE])
}

## ========================= MAIN =========================

# --- Load data
df_coefs   <- read_csv(file.path(data_path, "df_reg_coefs.csv"), show_col_types = FALSE)
df_factors <- read_csv(file.path(data_path, "df_factor_trends.csv"), show_col_types = FALSE) # month_date, market, mining, lifestyle
df_idx     <- read_csv(file.path(data_path, "indexes_city_and_sa4.csv"), show_col_types = FALSE)

# --- Indexing & alignment
df_factors$month_date <- as.Date(df_factors$month_date)
df_idx$month_date     <- as.Date(df_idx$month_date)

df_factors <- df_factors %>% arrange(month_date)
df_idx     <- df_idx %>% arrange(month_date)

# Choose region (example)
region <- "SYDNEY - BLACKTOWN"  # <-- change as needed

stopifnot(region %in% colnames(df_idx))

# Merge & align monthly; ensure regular monthly sequence
df <- df_factors %>%
  inner_join(df_idx[, c("month_date", region)], by="month_date") %>%
  rename(y = !!region)

# If needed, complete missing months:
all_months <- seq(min(df$month_date), max(df$month_date), by="month")
df <- merge(data.frame(month_date=all_months), df, by="month_date", all.x=TRUE)
df <- na.omit(df[, c("month_date","y","market","mining","lifestyle")])

y <- df$y
X <- as.matrix(df[, c("market","mining","lifestyle")])

## --- 1) OLS + HAC + VIF
ohv <- ols_hac_vif(y, X, max_lags=12)
cat("\n=== OLS (levels) coefficients ===\n")
print(coef(ohv$ols))
cat("\nNewey–West HAC (12 lags):\n")
print(ohv$hac)
cat("\n=== Variance Inflation Factors ===\n")
print(ohv$vif)

## --- 2) ARIMAX search in levels (d=0)
cat("\nRunning expanded ARIMAX grid search on (p,0,q)...\n")
srch <- search_arimax_levels(y, X)
print(head(srch$table, 9))
best <- srch$winner
cat("\nBest order:", paste(best$order, collapse=","), "\n")
cat("AICc:", best$aicc, "\n")
cat("Ljung–Box p(12):", best$lb_p12, "\n")

cat("\n=== ARIMAX exogenous coefficients (levels) ===\n")
print(coef(best$fit)[colnames(X)])

## --- 3) Optional: ARDL(1) + ARIMAX levels
run_ardl1 <- TRUE
if (run_ardl1) {
  cat("\nRunning ARDL(1)+ARIMAX robustness...\n")
  dy <- build_ardl1(as.data.frame(X), y)
  sr2 <- search_arimax_levels(dy$y, dy$X)
  bst2 <- sr2$winner
  cat("ARDL(1) best order:", paste(bst2$order, collapse=","), "\n")
  cat("AICc:", bst2$aicc, "  LB p(12):", bst2$lb_p12, "\n")
  b_main <- coef(best$fit)
  b_ardl <- coef(bst2$fit)
  cols <- colnames(X)
  comp <- data.frame(
    ARIMAX_contemp = b_main[cols],
    ARDL1_contemp  = b_ardl[cols],
    ARDL1_lag1     = b_ardl[paste0(cols, "_lag1")]
  )
  cat("\n=== Contemporaneous vs lagged betas (ARDL(1)) ===\n")
  print(round(comp, 4))
}
