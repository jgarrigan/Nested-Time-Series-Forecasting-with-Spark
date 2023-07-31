

# INSTALL PACKAGES --------------------------------------------------------

if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse,
  timetk,
  sparklyr,
  tidymodels,
  modeltime,
  tsibbledata
)


# SPARK CONNECTION --------------------------------------------------------

sc <- spark_connect(master = "local")

parallel_start(sc, .method = "spark")


# IMPORT DATA -------------------------------------------------------------

# LOAD DATA
aus_retail_tbl <- tsibbledata::aus_retail %>%
  tk_tbl() %>%
  select(Industry, Month, Turnover) %>%
  set_names(c("id", "date", "value")) %>%
  mutate(date = as.Date(tsibble::yearmonth(date)))


# NESTING DATA ------------------------------------------------------------

nested_data_tbl <- aus_retail_tbl %>%
  extend_timeseries(
    .id_var        = id,
    .date_var      = date,
    .length_future = 52
  ) %>%
  nest_timeseries(
    .id_var        = id,
    .length_future = 52
  ) %>%
  split_nested_timeseries(
    .length_test = 52
  )

nested_data_tbl


# XGB RECIPE AND WORKFLOW ---------------------------------------------------

rec_xgb <- recipe(value ~ ., extract_nested_train_split(nested_data_tbl)) %>%
  step_timeseries_signature(date) %>%
  step_rm(date) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

wflw_xgb <- workflow() %>%
  add_model(boost_tree("regression") %>% set_engine("xgboost")) %>%
  add_recipe(rec_xgb)


# PROPHET RECIPE AND WORKFLOW ---------------------------------------------

rec_prophet <- recipe(value ~ date, extract_nested_train_split(nested_data_tbl))

wflw_prophet <- workflow() %>%
  add_model(
    prophet_reg("regression", seasonality_yearly = TRUE) %>% 
      set_engine("prophet")
  ) %>%
  add_recipe(rec_prophet)


# NESTED FORECAST ---------------------------------------------------------

nested_modeltime_tbl <- nested_data_tbl %>%
  modeltime_nested_fit(
    wflw_xgb,
    wflw_prophet,
    
    control = control_nested_fit(allow_par = TRUE, verbose = TRUE)
  )

nested_modeltime_tbl


# MODEL TEST ACCURACY -----------------------------------------------------

nested_modeltime_tbl %>%
  extract_nested_test_accuracy() %>%
  table_modeltime_accuracy(.interactive = F)

nested_modeltime_tbl %>%
  extract_nested_test_forecast() %>%
  group_by(id) %>%
  plot_modeltime_forecast(.facet_ncol = 2, .interactive = F)

# Unregisters the Spark Backend
parallel_stop()

# Disconnects Spark
spark_disconnect_all()

