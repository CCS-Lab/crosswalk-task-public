#################################################################################

#if (!requireNamespace("devtools", quietly = TRUE) || packageVersion("devtools") < "1.6") {
#  install.packages("devtools")
#}
#devtools::install_github("CCS-Lab/easyml", subdir = "R")
library(ggplot2)
library(easyml)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))                                               
print(sprintf("current working directory: %s", getwd()))

library(grid)

############################### Setup #######################################

data_file_path <- "data/data_integrated_irl.csv"

performance_risk_taking_indicator_cols_5 = c(
  "average_score_full",
  "bonus_score_episode_ratio",
  "road_crossing_success_ratio_total",
  "sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_high_to_low_penalty",
  "run_over_episode_ratio"
)
performance_risk_taking_indicator_labels <- c(
  "Average Score",
  "Episode\nClear Ratio",
  "Road Crossing\nSuccess Ratio",
  "Crosswalk\nSearching",
  "Run Over Ratio"
)

irl_risk_taking_indicator_cols_6 = c(
  "safest_road_crossing_irl",
  "waiting_for_car_to_pass_irl",
  
  "high_risk_but_empty_last_road_irl",
  "approaching_crosswalk_in_current_road_irl",
  "approaching_next_crosswalks_irl",
  "move_up_toward_car_irl"
)
irl_risk_taking_indicator_labels <- c(
  "[IRL] Safest\nRoad Crossing",
  "[IRL] Waiting in\nCrosswalk for\nCar to Pass",
  
  "[IRL] High Risk but\nEmpty Last Road",
  "[IRL] Approaching\nCrosswalk in\nCurrent Road",
  "[IRL] Approaching\nNext Crosswalks",
  "[IRL] Move Up\nToward Car"
)

dependent_variable = "DOSPERT"

run_ElasticNet <- function(
  exclude_variables=NULL,
  alpha = 0.5, # Elastic Net
  n_samples = 1000,
  n_divisions = 1000,
  n_iterations = 100,
  n_lambda = 100,
  random_state = 42,
  n_core = 4,
  train_size=0.8, # default=0.667
  standardize = FALSE
) {
  output <- easy_glmnet(
    .data = df,
    dependent_variable = dependent_variable,
    exclude_variables = exclude_variables,
    train_size = train_size,
    n_samples = n_samples,
    n_divisions = n_divisions,
    n_iterations = n_iterations,
    random_state = random_state,
    progress_bar = TRUE,
    n_core = n_core,
    model_args = list(
      alpha = alpha,
      n_lambda = n_lambda,
      lambda.min.ratio = 1e-4,
      type.measure = "mse",
      nfolds = 5,
      standardize = standardize
    )
  )
  return(output)
}

plot_performance <- function(output, title="", bins = 25, ylim = 150, test=TRUE, save_path = NULL, height = 4.0, width = 3.5, dpi = 300) {
  if (test) {
    if (is.null(output$model_performance_test)) {
      stop("output$model_performance_test is empty")
    }
    mp <- output$model_performance_test
  } else {
    if (is.null(output$model_performance_train)) {
      stop("output$model_performance_train is empty")
    }
    mp <- output$model_performance_train
  }
  if (is.null(title) || title == "") {
    if (test) {
      title <- paste0("Test (r = ", round(mean(mp, na.rm = TRUE), 4), ")")
    } else {
      title <- paste0("Train (r = ", round(mean(mp, na.rm = TRUE), 4), ")")
    }
  }
  
  if (!is.null(save_path)) {
    ext <- tolower(tools::file_ext(save_path))
    
    if (ext == "png") {
      png(
        filename = save_path,
        width = width,
        height = height,
        units = "in",
        res = dpi
      )
    } else if (ext == "eps") {
      postscript(
        file = save_path,
        width = width,
        height = height,
        horizontal = FALSE,
        onefile = FALSE,
        paper = "special"
      )
    } else if (ext == "pdf") {
      pdf(
        file = save_path,
        width = width,
        height = height
      )
    } else {
      stop("Unsupported file format: ", ext)
    }
    
    on.exit(dev.off())
  }
  
  hist(
    mp,
    main = title,
    breaks = bins,
    col = "#4C72B0",
    border = "white",
    xlab = "Correlation score", 
    ylab = "Frequency",
    xlim = c(-0.6, 1), 
    ylim = c(0, ylim),
    cex.lab = 1.2, 
    cex.axis = 1.0
  )
  grid(
    col = "gray85",
    lty = "solid",
    lwd = 0.8
  )
  # redraw on top of the grid;;
  hist(
    mp,
    main = title,
    breaks = bins,
    col = "#4C72B0",
    border = "white",
    add = TRUE
  )  
  cat("r =", round(mean(mp, na.rm = TRUE), 4), "\n")
}

print_coef_table <- function(model) {
  coef_mat <- as.matrix(model$coefficients)
  N <- nrow(model$data)
  z <- 1.96
  
  coef_mean <- colMeans(coef_mat)
  coef_sd   <- apply(coef_mat, 2, sd)
  coef_se   <- coef_sd / sqrt(N)
  coef_ci_l <- coef_mean - z * coef_se
  coef_ci_u <- coef_mean + z * coef_se
  
  selection_rate <- colMeans(coef_mat != 0)
  
  coef_table <- data.frame(
    Mean = coef_mean,
    CI_95 = sprintf("[%.2f, %.2f]", coef_ci_l, coef_ci_u),
    Selection_rate = round(selection_rate, 2)
  )
  
  print(coef_table)
}

plot_with_95CI_bars <- function(
    model,
    ordered_predictors,
    save_path = NULL,
    total_width = 3.5,
    height = 3.5,
    dpi = 300,
    panel_width = 1.7,
    coefficient_range = c(-2.1, 2.1)
) {
  
  model$coefficients_processed$lower_bound <- model$coefficients_processed$mean - 1.96 * model$coefficients_processed$sd
  model$coefficients_processed$upper_bound <- model$coefficients_processed$mean + 1.96 * model$coefficients_processed$sd
  
  model$coefficients_processed$predictor <- ordered_predictors
  
  p <- plot_coefficients_processed(
    model$coefficients_processed
  )
  
  p$data$predictor <- as.character(p$data$predictor)
  
  p$data$predictor <- factor(
    p$data$predictor,
    levels = rev(ordered_predictors)
  )
  
  reference_n      <- 6
  reference_height <- 3.5
  reference_width  <- 0.20
  
  cap_width <- reference_width *
    (length(ordered_predictors) / reference_n) *
    (reference_height / height)
  
  errorbar_layers <- which(
    vapply(
      p$layers,
      function(layer) inherits(layer$geom, "GeomErrorbar"),
      logical(1)
    )
  )
  
  for (i in errorbar_layers) {
    p$layers[[i]]$geom_params$width <- cap_width
  }
  
  final_plot <- p +
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      linewidth = 0.5,
      colour = "grey40"
    ) +
    scale_x_discrete(
      limits = rev(ordered_predictors),
      labels = function(x) gsub("\n", "<br>", x)
    ) +
    scale_y_continuous(
      name = "Coefficient estimates",
      limits = coefficient_range
    ) +
    theme(
      legend.position = "none",
      plot.title = element_blank(),
      
      axis.title.x = element_text(
        size = 12,
        colour = "black"
      ),
      
      axis.title.y = element_blank(),
      
      axis.text.y = ggtext::element_markdown(
        colour = "black",
        hjust = 1,
        halign = 1,
        lineheight = 0.9,
        margin = margin(r = 6)
      )
    )
  
  g <- ggplot2::ggplotGrob(final_plot)
  
  panel_col <- unique(
    g$layout$l[g$layout$name == "panel"]
  )
  
  g$widths[panel_col] <- grid::unit(
    panel_width,
    "in"
  )
  
  if (!is.null(save_path)) {
    
    save_dir <- dirname(save_path)
    
    if (!dir.exists(save_dir)) {
      dir.create(
        save_dir,
        recursive = TRUE,
        showWarnings = FALSE
      )
    }
    
    ggplot2::ggsave(
      filename = save_path,
      plot = g,
      width = total_width,
      height = height,
      units = "in",
      dpi = dpi,
      bg = "white",
      limitsize = FALSE
    )
    
    message(
      "Saved figure: ",
      normalizePath(save_path, mustWork = FALSE)
    )
  }
  
  grid::grid.newpage()
  grid::grid.draw(g)
  
  invisible(g)
}

############################### Behavior-metrics only #######################################

df <- read.csv(data_file_path)

cols <- c(
  performance_risk_taking_indicator_cols_5,
  dependent_variable
)
df <- df[, cols, drop = FALSE]

performance_only = run_ElasticNet(train_size=0.8, n_iterations = 100, n_lambda = 100, random_state = 42)
saveRDS(performance_only, file = "regression/elasticnet1.rds")

#### Results ####

model <- readRDS("regression/elasticnet1.rds")
plot_performance(model, title="", bins = 20, ylim = 170, test=TRUE, save_path = "regression/elasticnet_model1_performance.png")
plot_performance(model, title="", bins = 15, ylim = 170, test=FALSE)
# test: r=0.2313  train: r=0.2777

print_coef_table(model)
#                                                                                   Mean          CI_95 Selection_rate
# (Intercept)                                                                  29.406197 [29.41, 29.41]              1
# average_score_full                                                           -1.412932 [-1.42, -1.41]              1
# bonus_score_episode_ratio                                                     0.000000   [0.00, 0.00]              0
# road_crossing_success_ratio_total                                             0.000000   [0.00, 0.00]              0
# sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_high_to_low_penalty -1.567418 [-1.58, -1.55]              1
# run_over_episode_ratio                                                        1.369809   [1.36, 1.38]              1

# 95% CI
plot_with_95CI_bars(
  model,
  c(performance_risk_taking_indicator_labels),
  save_path = "regression/elasticnet_model1.png",
  total_width = 2.6,
  height = 4.5,
  panel_width = 1.5
)

############################### IRL-metrics only #######################################

df <- read.csv(data_file_path)

cols <- c(
  irl_risk_taking_indicator_cols_6,
  dependent_variable
)
df <- df[, cols, drop = FALSE]

irl_only = run_ElasticNet(train_size=0.8, n_iterations = 100, n_lambda = 100, random_state = 42)
saveRDS(irl_only, file = "regression/elasticnet2.rds")

#### Results ####

model <- readRDS("regression/elasticnet2.rds")
plot_performance(model, title="", bins = 20, ylim = 170, test=TRUE, save_path = "regression/elasticnet_model2_performance.png")
plot_performance(model, title="", bins = 15, ylim = 170, test=FALSE)
# test: r=0.3058  # train: r=0.3641

print_coef_table(model)
#                                                 Mean          CI_95 Selection_rate
# (Intercept)                               29.4061972 [29.41, 29.41]              1
# safest_road_crossing_irl                  -1.8892257 [-1.90, -1.88]              1
# waiting_for_car_to_pass_irl               -1.5709283 [-1.58, -1.56]              1
# high_risk_but_empty_last_road_irl          1.5176139   [1.51, 1.53]              1
# approaching_crosswalk_in_current_road_irl  0.6211044   [0.62, 0.63]              1
# approaching_next_crosswalks_irl            0.6671390   [0.67, 0.67]              1
# move_up_toward_car_irl                     1.0717920   [1.07, 1.08]              1

# 95% CI
plot_with_95CI_bars(
  model,
  c(irl_risk_taking_indicator_labels),
  save_path = "regression/elasticnet_model2.png",
  total_width = 2.8,
  height = 4.5,
  panel_width = 1.5
)

################################## Full #######################################

df <- read.csv(data_file_path)

cols <- c(
  performance_risk_taking_indicator_cols_5,
  irl_risk_taking_indicator_cols_6,
  dependent_variable
)
df <- df[, cols, drop = FALSE]

full_model = run_ElasticNet(train_size=0.8, n_iterations = 100, n_lambda = 100, random_state = 42)
saveRDS(full_model, file = "regression/elasticnet3.rds")

#### Results ####

model <- readRDS("regression/elasticnet3.rds")
plot_performance(model, title="", bins = 20, ylim = 170, test=TRUE, save_path = "regression/elasticnet_model3_performance.png")
plot_performance(model, title="", bins = 13, ylim = 170, test=FALSE)
# test: r=0.3553 # train: r=0.4238

print_coef_table(model)
#                                                                                       Mean          CI_95 Selection_rate
# (Intercept)                                                                   2.940620e+01 [29.41, 29.41]              1
# average_score_full                                                           -1.853961e+00 [-1.86, -1.85]              1
# bonus_score_episode_ratio                                                    -9.226542e-05  [-0.00, 0.00]              0
# road_crossing_success_ratio_total                                             6.405818e-04  [-0.00, 0.00]              0
# sideway_movements_in_crosswalk_not_visible_safe_zone_cnt_high_to_low_penalty -1.633446e+00 [-1.64, -1.63]              1
# run_over_episode_ratio                                                        0.000000e+00   [0.00, 0.00]              0
# safest_road_crossing_irl                                                     -1.372814e+00 [-1.38, -1.37]              1
# waiting_for_car_to_pass_irl                                                  -1.444106e+00 [-1.45, -1.44]              1
# high_risk_but_empty_last_road_irl                                             1.481352e+00   [1.48, 1.49]              1
# approaching_crosswalk_in_current_road_irl                                     1.094589e+00   [1.09, 1.10]              1
# approaching_next_crosswalks_irl                                               5.878651e-01   [0.59, 0.59]              1
# move_up_toward_car_irl                                                        5.707097e-01   [0.57, 0.57]              1

# 95% CI
plot_with_95CI_bars(
  model, 
  c(performance_risk_taking_indicator_labels, irl_risk_taking_indicator_labels), 
  save_path = "regression/elasticnet_model3.png",
  total_width = 2.8,
  height = 4.5,
  panel_width = 1.5
)
