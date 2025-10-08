
# ====================================================
#  Stage 4 — Partition Series (fixed dates per rubric) ✅
# ====================================================
# Expect Full_Dataset_1 to cover 2020-01 .. 2024-12 (from your Stage 3)
stopifnot(exists("Full_Dataset_1"))
stopifnot(frequency(Full_Dataset_1) == 12)

# Train: 2020-01..2023-12 (48), Test: 2024-01..2024-12 (12)
Training_set <- window(Full_Dataset_1, start = c(2020, 1), end = c(2023, 12))
Test_set     <- window(Full_Dataset_1, start = c(2024, 1), end = c(2024, 12))

# OPTIONAL aliases so later code that expects ts_train/ts_test won't break
ts_train <- Training_set
ts_test  <- Test_set

# Split table (the marker expects a "Split Portion" column)
split_tbl <- tibble::tibble(
  `Dataset No`          = 1,
  Period                = "2020-01 to 2024-12",
  `Split Portion`       = "80 - 20",
  `Training set length` = length(Training_set),
  `Test set length`     = length(Test_set)
)
print(split_tbl)

# Helper: convert ts -> tibble with a proper Date column (fixes "transform_date()" errors)
ts_to_df <- function(ts_obj, set_name) {
  sy <- as.integer(start(ts_obj)[1]); sm <- as.integer(start(ts_obj)[2])
  start_date <- as.Date(sprintf("%04d-%02d-01", sy, sm))
  dates <- seq.Date(from = start_date, by = "month", length.out = length(ts_obj))
  tibble::tibble(Date = as.Date(dates), Value = as.numeric(ts_obj), Set = set_name)
}

# Train vs Test overlay plot
df_all <- dplyr::bind_rows(
  ts_to_df(Training_set, "Training"),
  ts_to_df(Test_set,     "Test")
)

ggplot2::ggplot(df_all, ggplot2::aes(Date, Value, color = Set)) +
  ggplot2::geom_line() +
  ggplot2::labs(title = "Training vs Test — Coal Generation",
                x = "Year", y = "Million kWh") +
  ggplot2::scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
  ggplot2::theme_minimal()


# ====================================================
#  Stage 5 — Prepare data for ARIMA analysis ✅
#  Visualise train/test, check stationarity/variance; justify transforms.
# ====================================================

# Visual diagnostics
ggtsdisplay(ts_train, main = "Training set — diagnostics")
ggtsdisplay(ts_test,  main = "Test set — diagnostics")

# Informational Box-Cox lambda (we’ll model on log scale: lambda=0)
lambda_info <- BoxCox.lambda(ts_train)
cat("Suggested Box-Cox lambda (info only):", lambda_info, "\n")

# Look at log scale and seasonal differencing
ts_train %>% log() %>% ggtsdisplay(main = "Log(ts_train) — diagnostics")
ts_train %>% log() %>% diff(lag = 12) %>% ggtsdisplay(main = "Log(ts_train) with seasonal diff (lag = 12)")
needed_d <- ts_train %>% log() %>% diff(lag = 12) %>% ndiffs()
cat("Additional non-seasonal differences suggested AFTER seasonal diff:", needed_d, "\n")

# Formal stationarity checks (nice for report narrative)
ts_train_seasdiff <- ts_train %>% log() %>% diff(lag = 12)
cat("\nKPSS test on log seasonal differenced data:\n")
print(summary(ur.kpss(ts_train_seasdiff)))
cat("\nADF test on log seasonal differenced data:\n")
print(tseries::adf.test(stats::na.omit(ts_train_seasdiff)))


# ====================================================
#  Stage 6 — Pure AR and Pure MA models ✅
#  Fit simple AR(p) and MA(q) on log scale (lambda=0); residuals, forecasts, accuracy, tsCV.
# ====================================================

## ---- Pure AR (lambda = 0 means fit on log-scale implicitly) ----
arima_ar1 <- Arima(ts_train, order = c(1,0,0), lambda = 0)
arima_ar2 <- Arima(ts_train, order = c(2,0,0), lambda = 0)
arima_ar3 <- Arima(ts_train, order = c(3,0,0), lambda = 0)
arima_ar4 <- Arima(ts_train, order = c(4,0,0), lambda = 0)

cat("AR Model AICs:\n",
    "AR(1):", arima_ar1$aic, "\n",
    "AR(2):", arima_ar2$aic, "\n",
    "AR(3):", arima_ar3$aic, "\n",
    "AR(4):", arima_ar4$aic, "\n")

# Residuals should look like white noise
checkresiduals(arima_ar1)
checkresiduals(arima_ar2)
checkresiduals(arima_ar3)
checkresiduals(arima_ar4)

# Best AR by AIC
ar_aics <- c(arima_ar1$aic, arima_ar2$aic, arima_ar3$aic, arima_ar4$aic)
best_ar_order <- which.min(ar_aics)
best_ar_model <- get(paste0("arima_ar", best_ar_order))
cat("Best pure AR model by AIC: AR(", best_ar_order, ")\n", sep = "")

# Forecast to test horizon and evaluate
fc_ar <- forecast(best_ar_model, h = length(ts_test))
autoplot(fc_ar) +
  autolayer(ts_test, series = "Actual Test Data") +
  ggtitle(paste0("Pure AR(", best_ar_order, ") Forecast vs Test")) +
  ylab("Coal Net Generation (Mn kWh)") + xlab("Time")
ar_accuracy <- accuracy(fc_ar, ts_test)
cat("\nPure AR accuracy (Test):\n"); print(ar_accuracy)

# tsCV for AR family (h=12): pure AR only (p up to 4), no MA, no differencing
e_ar <- tsCV(Full_Dataset_1,
             forecastfunction = function(y, h) {
               m <- auto.arima(y, max.p = 4, max.q = 0, max.d = 0,
                               max.P = 0, max.Q = 0, max.D = 0,
                               seasonal = FALSE, lambda = 0,
                               stepwise = FALSE, approximation = FALSE)
               forecast(m, h = h)
             }, h = 12)
ar_rmse_cv <- sqrt(mean(e_ar^2, na.rm = TRUE))
cat("Pure AR tsCV(h=12) RMSE:", ar_rmse_cv, "\n")

## ---- Pure MA (lambda = 0) ----
ma_model_1 <- Arima(ts_train, order = c(0,0,1), lambda = 0)
ma_model_2 <- Arima(ts_train, order = c(0,0,2), lambda = 0)
ma_model_3 <- Arima(ts_train, order = c(0,0,3), lambda = 0)
ma_model_4 <- Arima(ts_train, order = c(0,0,4), lambda = 0)

# Optional seasonal MA variants (include only if diagnostics show seasonality)
sma_model_1  <- Arima(ts_train, order = c(0,0,0), seasonal = c(0,0,1), lambda = 0)
ma_sma_model <- Arima(ts_train, order = c(0,0,1), seasonal = c(0,0,1), lambda = 0)

# Residual checks
checkresiduals(ma_model_1); checkresiduals(ma_model_2)
checkresiduals(ma_model_3); checkresiduals(ma_model_4)
checkresiduals(sma_model_1); checkresiduals(ma_sma_model)

# Compare MA models by AICc
ma_comparison <- data.frame(
  Model = c("MA(1)", "MA(2)", "MA(3)", "MA(4)", "SMA(1)", "MA(1)+SMA(1)"),
  AICc  = c(ma_model_1$aicc, ma_model_2$aicc, ma_model_3$aicc,
            ma_model_4$aicc, sma_model_1$aicc, ma_sma_model$aicc)
) |> dplyr::arrange(AICc)
cat("\nPure/Seasonal MA models sorted by AICc:\n"); print(ma_comparison)

best_ma_model_name <- ma_comparison$Model[1]
best_ma_model <- switch(best_ma_model_name,
                        "MA(1)"        = ma_model_1,
                        "MA(2)"        = ma_model_2,
                        "MA(3)"        = ma_model_3,
                        "MA(4)"        = ma_model_4,
                        "SMA(1)"       = sma_model_1,
                        "MA(1)+SMA(1)" = ma_sma_model)

# Forecast and evaluate best MA
fc_ma <- forecast(best_ma_model, h = length(ts_test))
autoplot(fc_ma) +
  autolayer(ts_test, series = "Actual Test Data") +
  ggtitle(paste0("Best MA Model (", best_ma_model_name, ") Forecast vs Test")) +
  ylab("Coal Net Generation (Mn kWh)") + xlab("Time")
ma_accuracy <- accuracy(fc_ma, ts_test)
cat("\nBest MA accuracy (Test):\n"); print(ma_accuracy)

# tsCV for MA family (h=12) — pure MA (q up to 4), allow one seasonal MA
e_ma <- tsCV(Full_Dataset_1,
             forecastfunction = function(y, h) {
               m <- auto.arima(y, max.p = 0, max.q = 4, max.d = 0,
                               max.P = 0, max.Q = 1, max.D = 0,
                               seasonal = TRUE, lambda = 0,
                               stepwise = FALSE, approximation = FALSE)
               forecast(m, h = h)
             }, h = 12)
ma_rmse_cv <- sqrt(mean(e_ma^2, na.rm = TRUE))
cat("Pure/Seasonal MA tsCV(h=12) RMSE:", ma_rmse_cv, "\n")


# ====================================================
#  Stage 7 — Auto ETS and Auto ARIMA ✅
#  Report models, residuals, forecast interval vs test, traditional & modern accuracy.
# ====================================================

## ---- Auto ETS (lambda=0) ----
fit_ets <- ets(ts_train, lambda = 0)
cat("ETS model form:", fit_ets$method, "\n")
checkresiduals(fit_ets)

fc_ets <- forecast(fit_ets, h = length(ts_test))
autoplot(fc_ets) + 
  ggtitle("Auto ETS Forecast vs Test Set") +
  ylab("Coal Net Generation (Mn kWh)") + xlab("Time")
acc_ets <- accuracy(fc_ets, ts_test)
cat("\nETS accuracy (Test):\n"); print(acc_ets)

# tsCV for ETS (h=12)
e_ets <- tsCV(Full_Dataset_1,
              forecastfunction = function(y, h) forecast(ets(y), h = h),
              h = 12)
ets_rmse_cv <- sqrt(mean(e_ets^2, na.rm = TRUE))
cat("ETS tsCV(h=12) RMSE:", ets_rmse_cv, "\n")

## ---- Auto ARIMA (lambda=0) ----
fit_auto_arima <- auto.arima(ts_train, lambda = 0, stepwise = FALSE, approximation = FALSE)
cat("Auto ARIMA form:", paste(arimaorder(fit_auto_arima), collapse=","), "\n")
checkresiduals(fit_auto_arima)

fc_arima <- forecast(fit_auto_arima, h = length(ts_test))
autoplot(fc_arima) +
  ggtitle("Auto ARIMA Forecast vs Test Set") +
  ylab("Coal Net Generation (Mn kWh)") + xlab("Time")
acc_arima <- accuracy(fc_arima, ts_test)
cat("\nAuto ARIMA accuracy (Test):\n"); print(acc_arima)

# tsCV for Auto ARIMA (h=12), same transform as modeling
e_arima <- tsCV(Full_Dataset_1,
                forecastfunction = function(y, h) {
                  forecast(auto.arima(y, seasonal = TRUE, lambda = 0), h = h)
                }, h = 12)
arima_rmse_cv <- sqrt(mean(e_arima^2, na.rm = TRUE))
cat("Auto ARIMA tsCV(h=12) RMSE:", arima_rmse_cv, "\n")


# ====================================================
#  Stage 8 — Champion Model ✅
#  Compare families and pick winner (lowest Test RMSE; report tsCV too).
# ====================================================

rmse_ar     <- sqrt(mean((fc_ar$mean     - ts_test)^2, na.rm = TRUE))
rmse_ma     <- sqrt(mean((fc_ma$mean     - ts_test)^2, na.rm = TRUE))
rmse_ets    <- sqrt(mean((fc_ets$mean    - ts_test)^2, na.rm = TRUE))
rmse_arima  <- sqrt(mean((fc_arima$mean  - ts_test)^2, na.rm = TRUE))

model_comparison <- data.frame(
  Model = c(paste0("Pure AR(", best_ar_order, ")"),
            paste0("Best ", best_ma_model_name),
            paste0("ETS: ", fit_ets$method),
            paste0("Auto ARIMA: ", paste(arimaorder(fit_auto_arima), collapse=","))),
  RMSE  = c(rmse_ar, rmse_ma, rmse_ets, rmse_arima),
  tsCV_RMSE_h12 = c(ar_rmse_cv, ma_rmse_cv, ets_rmse_cv, arima_rmse_cv)
) |> dplyr::arrange(RMSE)

cat("\n=== Stage 8 — Champion Table (sorted by Test RMSE) ===\n")
print(model_comparison, row.names = FALSE)

champion_name <- model_comparison$Model[1]
cat("\nChampion model:", champion_name, "\n")


# ====================================================
#  Stage 9 — Implement your forecast ✅
#  Refit champion type on Full_Dataset_1 (2020–2024),
#  forecast the next 12 (or match COVID length), compare if needed.
# ====================================================

# Determine covid period object name from Stage 3 (handle either spelling)
covid_obj_name <- if (exists("covid_period")) "covid_period" else if (exists("Covid_Period")) "Covid_Period" else NA
h_fc <- if (!is.na(covid_obj_name)) length(get(covid_obj_name)) else 12

# Refit champion TYPE on Full_Dataset_1
if (grepl("^Pure AR", champion_name)) {
  champ_fit <- Arima(Full_Dataset_1, order = c(best_ar_order,0,0), lambda = 0)
} else if (grepl("^Best MA", champion_name)) {
  q <- if (grepl("MA\\((\\d+)\\)", best_ma_model_name)) as.integer(sub(".*MA\\((\\d+)\\).*", "\\1", best_ma_model_name)) else 1
  champ_fit <- Arima(Full_Dataset_1, order = c(0,0,q), lambda = 0)
} else if (grepl("^ETS", champion_name)) {
  champ_fit <- ets(Full_Dataset_1, lambda = 0)
} else { # Auto ARIMA
  champ_fit <- auto.arima(Full_Dataset_1, lambda = 0, stepwise = FALSE, approximation = FALSE)
}

fc_full <- forecast(champ_fit, h = h_fc)

# Plot champion forecast on Full_Dataset_1
autoplot(fc_full) +
  ggtitle(paste("Champion Forecast on Full_Dataset_1 —", champion_name)) +
  ylab("Coal Net Generation (Mn kWh)") + xlab("Year")

# Optional: compare forecast vs actual during the COVID window (if available)
if (!is.na(covid_obj_name)) {
  covid_ts <- get(covid_obj_name)
  
  covid_df <- ts_to_df(covid_ts, "Actual (COVID window)")
  fc_covid_df <- data.frame(
    Date  = covid_df$Date,
    Value = as.numeric(head(fc_full$mean, n = nrow(covid_df))),
    Set   = "Forecast (Champion)"
  )
  plot_df <- dplyr::bind_rows(covid_df, fc_covid_df)
  
  ggplot2::ggplot(plot_df, ggplot2::aes(x = Date, y = Value, color = Set)) +
    ggplot2::geom_line(size = 1) +
    ggplot2::labs(title = "Champion Forecast vs Actual — COVID Window",
                  x = "Date", y = "Million kWh") +
    ggplot2::scale_color_manual(values = c("Actual (COVID window)" = "red",
                                           "Forecast (Champion)"   = "blue")) +
    ggplot2::theme_minimal()
}
