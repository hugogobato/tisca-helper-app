import streamlit as st
import google.generativeai as genai
import textwrap # For formatting the LLM response nicely

# --- Configuration ---
# Load API key from secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("GEMINI_API_KEY not found in secrets.toml. Please add it.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini: {e}")
    st.stop()

MODEL_NAME = "gemini-2.5-flash-preview-05-20" # Or "gemini-pro" if you prefer, Flash is faster & cheaper for chat

# Your TISCA System Prompt (as provided)
TISCA_SYSTEM_PROMPT = """
You will be helping researchers implement my algorithm Test-Informed Simulation Count Algorithm (TISCA). Below you can find information about it:
3 The Proposed Algorithm: TISCA
Addressing the identified methodological gaps in the evaluation of treatment effect
estimators—namely, the prevalent lack of formal statistical testing for model comparisons and the arbitrary selection of simulation replications—requires a systematic
and statistically grounded approach. To this end, we introduce the Test-Informed
Simulation Count Algorithm (TISCA). TISCA is designed to integrate rigorous hypothesis testing directly into the Monte Carlo simulation workflow, simultaneously determining the minimum number of simulations (J) required to achieve a pre-specified
statistical power for detecting meaningful differences in model performance.
3.1 Methodological Foundation: The Welch’s t-test
At the core of TISCA lies the comparison of performance metrics (e.g., PEHE, RMSEAT E,
Coverage, CIL) between the proposed model and the most performing benchmark
model across J simulation runs, which consequently means that if the proposed model
is statistically significantly superior to the most performing benchmark model for a
certain performance metric, it must also be superior to the other benchmark models.
A natural choice for comparing the means of these metrics between two groups (e.g.,
performance of model A vs. model B across simulations) is the t-test (Welch, 1947).However, a standard Student’s t-test relies on the assumption of equal variances between the two groups being compared (Welch, 1947).
During our bibliometric analysis, alongside general observations in simulation studies, we noticed that that the variability (standard deviation or variance) of performance
metrics across Monte Carlo runs often differs between models, especially models that
are highly different in nature (e.g., OLS linear model and Bayesian Causal Forest (Hahn
et al., 2020)). For instance, a more complex model might exhibit higher variance in its
PEHE estimates across simulations compared to a simpler, more stable benchmark.
Violating the homogeneity of variance assumption can lead to unreliable p-values and
potentially incorrect conclusions if a standard t-test is employed (Zimmerman, 2004).
Therefore, TISCA utilizes the Welch’s t-test (Welch, 1947) as its default comparison method. The Welch’s t-test does not assume equal variances and adjusts its degrees
of freedom accordingly using the Welch-Satterthwaite equation, providing a more robust comparison when the variances of the performance metrics between models are
unequal.
A second assumption for the t-test (both Student’s and Welch’s) is the normality of
the underlying data within each group. In our context, this refers to the distribution of
a specific performance metric (e.g., PEHE values) obtained from the J simulation runs
for a given model. While true normality is seldom guaranteed, we leverage the Central
Limit Theorem (CLT). Many key performance metrics, such as RMSEAT E (an average
error) and PEHE (average squared error), are derived from averaging or summing operations across units within each simulation. As the number of simulation replications
(J) increases, the distribution of the mean of these metrics tends towards normality due
to the CLT (Brosamler, 1988; Dudley, 1978). We consider this assumption reasonably
met, particularly for the sample sizes (J) typically encountered or targeted in simulation studies (e.g., J >> 30), which aligns with common heuristics for the applicability
of the CLT (Kwiatkowski et al., 1992). Crucially, TISCA deliberately avoids performing formal statistical tests for normality (e.g., Shapiro-Wilk test) prior to conducting the Welch’s t-test. The practice of
”pre-testing” assumptions like normality has been widely criticized in the statistical literature (n.d.; Rasch et al., 2009; Rochon et al., 2012; Schucany & Tony Ng, 2006; Zimmerman, 2004). Such pre-testing conditions the subsequent primary test (the Welch’s
t-test in our case) on the outcome of the pre-test, altering the true Type I error rate and
distorting the statistical inference. Given the known robustness of the t-test (especially
Welch’s) to moderate deviations from normality, particularly with reasonable sample
sizes (J) (Welch, 1947; Zimmerman, 2004), and the established issues with pre-testing,
we proceed directly with the Welch’s t-test, relying on the CLT for justification.
3.2 The TISCA Algorithm Workflow (R code)
# TISCA: Test-Informed Simulation Count Algorithm
# Generalized R Package for Determining Optimal Simulation Counts

#' TISCA: Test-Informed Simulation Count Algorithm
#'
#' @description
#' A generalized implementation of the Test-Informed Simulation Count Algorithm (TISCA)
#' for determining the optimal number of simulations required to achieve a target
#' statistical power when comparing model performance metrics.
#'
#' @param sim_func A function that takes a seed parameter and returns a data frame
#'        containing performance metrics for all models being compared in a single simulation.
#'        The function must have the signature: function(seed) { ... }
#' @param comparison_pairs A list of pairs, where each pair is a vector of two strings:
#'        c("metric_proposed", "metric_benchmark"). These strings must match column names
#'        in the data frame returned by sim_func.
#' @param mdes A numeric vector of Minimum Detectable Effect Sizes, one for each comparison pair.
#'        For metrics where lower is better (e.g., RMSE, PEHE), use negative values.
#'        For metrics where higher is better (e.g., coverage), use positive values.
#' @param target_power The desired statistical power (1-β), typically 0.80.
#' @param alpha The significance level (α), typically 0.05.
#' @param batch_size The number of simulations to run in each iteration.
#' @param initial_count The number of simulations to run before the first power check.
#'        Defaults to batch_size if NULL.
#' @param correction_method The method for adjusting p-values in multiple testing.
#'        Options: "none", "bonferroni", "holm", "BH" (Benjamini-Hochberg).
#' @param verbose Whether to print progress updates.
#' @param save_results Whether to save results to CSV files.
#' @param output_dir Directory to save output files. Defaults to current directory.
#'
#' @return A list containing:
#'   \\item{J_final}{The final number of simulations required}
#'   \\item{P_raw}{Vector of raw p-values}
#'   \\item{P_adj}{Vector of adjusted p-values}
#'   \\item{Stats}{Vector of test statistics}
#'   \\item{P_achieved}{Vector of achieved powers}
#'   \\item{results}{Data frame with all simulation results}
#'   \\item{comparison_pairs}{The comparison pairs used}
#'   \\item{mdes}{The minimum detectable effect sizes used}
#'   \\item{summary_table}{A data frame summarizing the results}
#'   \\item{power_tracking}{Data frame tracking power estimates across simulation counts}
#'   \\item{pvalue_tracking}{Data frame tracking p-values across simulation counts}
#'
#' @examples
#' # Define a simple simulation function
#' sim_func <- function(seed) {
#'   set.seed(seed)
#'   # Simulate data and fit models
#'   # Return performance metrics as a data frame
#'   data.frame(
#'     model_a_rmse = rnorm(1, 0.5, 0.1),
#'     model_b_rmse = rnorm(1, 0.6, 0.1),
#'     model_a_coverage = rnorm(1, 0.9, 0.05),
#'     model_b_coverage = rnorm(1, 0.85, 0.05)
#'   )
#' }
#'
#' # Define comparison pairs
#' comparison_pairs <- list(
#'   c("model_a_rmse", "model_b_rmse"),
#'   c("model_a_coverage", "model_b_coverage")
#' )
#'
#' # Define minimum detectable effect sizes
#' mdes <- c(-0.1, 0.05)  # Negative for RMSE (lower is better), positive for coverage
#'
#' # Run TISCA
#' results <- run_tisca(
#'   sim_func = sim_func,
#'   comparison_pairs = comparison_pairs,
#'   mdes = mdes,
#'   target_power = 0.80,
#'   alpha = 0.05,
#'   batch_size = 10,
#'   initial_count = 20
#' )
#'
#' @export
run_tisca <- function(
  sim_func,                # Function that runs one simulation and returns metrics
  comparison_pairs,        # List of pairs to compare (metric_p, metric_b)
  mdes,                    # Vector of minimum detectable effect sizes
  target_power = 0.80,     # Target statistical power
  alpha = 0.05,            # Significance level
  batch_size = 50,         # Number of simulations per batch
  initial_count = NULL,    # Initial simulation count (defaults to batch_size if NULL)
  correction_method = "none", # Multiple testing correction method
  verbose = TRUE,          # Whether to print progress updates
  save_results = TRUE,     # Whether to save results to CSV files
  output_dir = "."         # Directory to save output files
) {
  # Check if required packages are installed and load them
  required_packages <- c("stats", "utils", "progress")
  for (pkg in required_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
    library(pkg, character.only = TRUE)
  }

  if (save_results) {
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
      install.packages("ggplot2")
    }
    library(ggplot2)
  }

  # Validate inputs
  if (!is.function(sim_func)) {
    stop("sim_func must be a function")
  }

  if (!is.list(comparison_pairs) || length(comparison_pairs) == 0) {
    stop("comparison_pairs must be a non-empty list")
  }

  for (pair in comparison_pairs) {
    if (!is.vector(pair) || length(pair) != 2) {
      stop("Each comparison pair must be a vector of length 2")
    }
  }

  if (!is.numeric(mdes) || length(mdes) != length(comparison_pairs)) {
    stop("mdes must be a numeric vector with the same length as comparison_pairs")
  }

  if (!is.numeric(target_power) || target_power <= 0 || target_power >= 1) {
    stop("target_power must be a number between 0 and 1")
  }

  if (!is.numeric(alpha) || alpha <= 0 || alpha >= 1) {
    stop("alpha must be a number between 0 and 1")
  }

  if (!is.numeric(batch_size) || batch_size <= 0) {
    stop("batch_size must be a positive number")
  }

  if (!is.null(initial_count) && (!is.numeric(initial_count) || initial_count <= 0)) {
    stop("initial_count must be a positive number or NULL")
  }

  if (!correction_method %in% c("none", "bonferroni", "holm", "BH")) {
    stop("correction_method must be one of: 'none', 'bonferroni', 'holm', 'BH'")
  }

  # Set initial count to batch_size if not specified
  if (is.null(initial_count)) {
    initial_count <- batch_size
  }

  # Create output directory if it doesn't exist
  if (save_results && !dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Helper function for Welch's t-test
  welch_t_test <- function(data, metric_p, metric_b) {
    # Extract the metrics for both models
    values_p <- data[[metric_p]]
    values_b <- data[[metric_b]]

    # Perform Welch's t-test
    test_result <- t.test(values_p, values_b, paired = FALSE, var.equal = FALSE)

    return(list(
      p_value = test_result$p.value,
      t_stat = test_result$statistic,
      sd_p = sd(values_p),
      sd_b = sd(values_b),
      mean_p = mean(values_p),
      mean_b = mean(values_b)
    ))
  }

  # Helper function to adjust p-values for multiple testing
  adjust_p_values <- function(p_values, method) {
    if (method == "none") {
      return(p_values)
    } else {
      return(p.adjust(p_values, method = method))
    }
  }

  # Helper function to estimate power for Welch's t-test
  estimate_welch_power <- function(n1, n2, sd1, sd2, delta, alpha) {
    # Calculate degrees of freedom using Welch-Satterthwaite equation
    df <- ((sd1^2/n1 + sd2^2/n2)^2) /
          ((sd1^2/n1)^2/(n1-1) + (sd2^2/n2)^2/(n2-1))

    # Calculate non-centrality parameter
    ncp <- delta / sqrt(sd1^2/n1 + sd2^2/n2)

    # Calculate critical value
    crit <- qt(1-alpha/2, df)

    # Calculate power
    power <- 1 - pt(crit, df, ncp) + pt(-crit, df, ncp)

    return(power)
  }

  # Initialization Phase
  J <- 0
  results_agg <- data.frame() # Empty dataframe to store results
  K <- length(comparison_pairs)
  P_current <- rep(0, K)      # Vector of K zeros for current power

  # Create data structures to track power and p-values over iterations
  power_tracking <- data.frame(
    simulations = numeric(),
    comparison = character(),
    power = numeric()
  )

  pvalue_tracking <- data.frame(
    simulations = numeric(),
    comparison = character(),
    p_raw = numeric(),
    p_adj = numeric()
  )

  # Create a progress bar if verbose
  if (verbose) {
    cat("Starting TISCA algorithm...\\n")
    cat("Target power:", target_power, "\\n")
    cat("Significance level:", alpha, "\\n")
    cat("Batch size:", batch_size, "\\n")
    cat("Initial count:", initial_count, "\\n")
    cat("Correction method:", correction_method, "\\n")
    cat("Number of comparisons:", K, "\\n\\n")
  }

  # Optional Initial Run
  if (initial_count > 0 && initial_count >= batch_size) {
    if (verbose) {
      cat("Running initial", initial_count, "simulations...\\n")
      pb <- progress_bar$new(total = initial_count)
    }

    # Run initial simulations
    for (i in 1:initial_count) {
      if (verbose) pb$tick()

      # Try to run the simulation with error handling
      tryCatch({
        metrics_run <- sim_func(seed = i)

        # Validate that the simulation function returns a data frame
        if (!is.data.frame(metrics_run)) {
          stop("Simulation function must return a data frame")
        }

        # Validate that all required metrics are in the data frame
        all_metrics <- unique(unlist(comparison_pairs))
        missing_metrics <- all_metrics[!all_metrics %in% names(metrics_run)]
        if (length(missing_metrics) > 0) {
          stop(paste("Missing metrics in simulation output:",
                     paste(missing_metrics, collapse = ", ")))
        }

        # Add to results
        results_agg <- rbind(results_agg, metrics_run)
      }, error = function(e) {
        stop(paste("Error in simulation function:", e$message))
      })
    }
    J <- initial_count

    # Perform initial power calculation
    p_values_raw <- numeric(K)
    test_stats <- numeric(K)
    mean_diffs <- numeric(K)

    for (k in 1:K) {
      # Extract comparison information
      metric_p <- comparison_pairs[[k]][1]
      metric_b <- comparison_pairs[[k]][2]
      comparison_name <- paste(metric_p, "vs", metric_b)

      # Perform Welch's t-test
      test_result <- welch_t_test(results_agg, metric_p, metric_b)

      # Store results
      p_values_raw[k] <- test_result$p_value
      test_stats[k] <- test_result$t_stat
      mean_diffs[k] <- test_result$mean_p - test_result$mean_b

      # Estimate power if standard deviations are valid
      if (test_result$sd_p > 0 && test_result$sd_b > 0) {
        P_current[k] <- estimate_welch_power(
          J, J,
          test_result$sd_p, test_result$sd_b,
          mdes[k], alpha
        )
      } else {
        P_current[k] <- 0
      }

      # Track power
      power_tracking <- rbind(power_tracking, data.frame(
        simulations = J,
        comparison = comparison_name,
        power = P_current[k]
      ))
    }

    # Adjust p-values for multiple testing
    p_values_adj <- adjust_p_values(p_values_raw, correction_method)

    # Track p-values
    for (k in 1:K) {
      comparison_name <- paste(comparison_pairs[[k]][1], "vs", comparison_pairs[[k]][2])
      pvalue_tracking <- rbind(pvalue_tracking, data.frame(
        simulations = J,
        comparison = comparison_name,
        p_raw = p_values_raw[k],
        p_adj = p_values_adj[k]
      ))
    }

    if (verbose) {
      cat("\\nAfter initial", J, "simulations:\\n")
      for (k in 1:K) {
        cat(sprintf("Comparison %d: Power = %.4f, p-value = %.4f, adjusted p-value = %.4f\\n",
                    k, P_current[k], p_values_raw[k], p_values_adj[k]))
      }
      cat("Minimum power:", min(P_current), "\\n\\n")
    }
  }

  # Iterative Simulation and Power Check Loop
  iteration <- 1

  while (min(P_current) < target_power) {
    if (verbose) {
      cat(sprintf("Iteration %d: Running batch of %d simulations (total will be %d)...\\n",
                  iteration, batch_size, J + batch_size))
      pb <- progress_bar$new(total = batch_size)
    }

    # Run a new batch of simulations
    start_seed <- J + 1
    end_seed <- J + batch_size

    for (i in start_seed:end_seed) {
      if (verbose) pb$tick()

      # Try to run the simulation with error handling
      tryCatch({
        metrics_run <- sim_func(seed = i)

        # Validate that the simulation function returns a data frame
        if (!is.data.frame(metrics_run)) {
          stop("Simulation function must return a data frame")
        }

        # Validate that all required metrics are in the data frame
        all_metrics <- unique(unlist(comparison_pairs))
        missing_metrics <- all_metrics[!all_metrics %in% names(metrics_run)]
        if (length(missing_metrics) > 0) {
          stop(paste("Missing metrics in simulation output:",
                     paste(missing_metrics, collapse = ", ")))
        }

        # Add to results
        results_agg <- rbind(results_agg, metrics_run)
      }, error = function(e) {
        stop(paste("Error in simulation function:", e$message))
      })
    }

    J <- J + batch_size

    # Perform tests and estimate power on current J simulations
    p_values_raw <- numeric(K)
    test_stats <- numeric(K)
    mean_diffs <- numeric(K)

    for (k in 1:K) {
      # Extract comparison information
      metric_p <- comparison_pairs[[k]][1]
      metric_b <- comparison_pairs[[k]][2]
      comparison_name <- paste(metric_p, "vs", metric_b)

      # Perform Welch's t-test
      test_result <- welch_t_test(results_agg, metric_p, metric_b)

      # Store results
      p_values_raw[k] <- test_result$p_value
      test_stats[k] <- test_result$t_stat
      mean_diffs[k] <- test_result$mean_p - test_result$mean_b

      # Estimate power if standard deviations are valid
      if (test_result$sd_p > 0 && test_result$sd_b > 0) {
        P_current[k] <- estimate_welch_power(
          J, J,
          test_result$sd_p, test_result$sd_b,
          mdes[k], alpha
        )
      } else {
        P_current[k] <- 0
      }

      # Track power
      power_tracking <- rbind(power_tracking, data.frame(
        simulations = J,
        comparison = comparison_name,
        power = P_current[k]
      ))
    }

    # Adjust p-values for multiple testing
    p_values_adj <- adjust_p_values(p_values_raw, correction_method)

    # Track p-values
    for (k in 1:K) {
      comparison_name <- paste(comparison_pairs[[k]][1], "vs", comparison_pairs[[k]][2])
      pvalue_tracking <- rbind(pvalue_tracking, data.frame(
        simulations = J,
        comparison = comparison_name,
        p_raw = p_values_raw[k],
        p_adj = p_values_adj[k]
      ))
    }

    if (verbose) {
      cat("\\nAfter", J, "simulations:\\n")
      for (k in 1:K) {
        cat(sprintf("Comparison %d: Power = %.4f, p-value = %.4f, adjusted p-value = %.4f\\n",
                    k, P_current[k], p_values_raw[k], p_values_adj[k]))
      }
      cat("Minimum power:", min(P_current), "\\n\\n")
    }

    iteration <- iteration + 1
  }

  # Output Phase
  J_final <- J
  P_raw <- p_values_raw
  P_adj <- p_values_adj
  Stats <- test_stats
  P_achieved <- P_current

  # Create summary table
  summary_table <- data.frame(
    Comparison = character(),
    Proposed_Metric = character(),
    Benchmark_Metric = character(),
    MDE = numeric(),
    Mean_Difference = numeric(),
    Power = numeric(),
    P_Value = numeric(),
    Adjusted_P_Value = numeric(),
    T_Statistic = numeric(),
    stringsAsFactors = FALSE
  )

  for (k in 1:length(comparison_pairs)) {
    summary_table[k, "Comparison"] <- paste(comparison_pairs[[k]][1], "vs", comparison_pairs[[k]][2])
    summary_table[k, "Proposed_Metric"] <- comparison_pairs[[k]][1]
    summary_table[k, "Benchmark_Metric"] <- comparison_pairs[[k]][2]
    summary_table[k, "MDE"] <- mdes[k]
    summary_table[k, "Mean_Difference"] <- mean_diffs[k]
    summary_table[k, "Power"] <- P_achieved[k]
    summary_table[k, "P_Value"] <- P_raw[k]
    summary_table[k, "Adjusted_P_Value"] <- P_adj[k]
    summary_table[k, "T_Statistic"] <- Stats[k]
  }

  if (verbose) {
    cat("\\n=== TISCA COMPLETED ===\\n")
    cat("Final number of simulations required:", J_final, "\\n")
    cat("Final achieved powers:\\n")
    for (k in 1:K) {
      cat(sprintf("Comparison %d (%s vs %s): Power = %.4f, p-value = %.4f, adjusted p-value = %.4f\\n",
                  k, comparison_pairs[[k]][1], comparison_pairs[[k]][2],
                  P_achieved[k], P_raw[k], P_adj[k]))
    }
  }

  # Create visualizations and save results if requested
  if (save_results) {
    # Ensure output directory exists
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }

    # File paths
    results_file <- file.path(output_dir, "tisca_simulation_results.csv")
    summary_file <- file.path(output_dir, "tisca_summary_results.csv")
    power_tracking_file <- file.path(output_dir, "power_tracking.csv")
    pvalue_tracking_file <- file.path(output_dir, "pvalue_tracking.csv")
    power_plot_file <- file.path(output_dir, "power_vs_simulations.png")
    pvalue_plot_file <- file.path(output_dir, "pvalue_comparison.png")

    # Save data to CSV files
    write.csv(results_agg, results_file, row.names = FALSE)
    write.csv(summary_table, summary_file, row.names = FALSE)
    write.csv(power_tracking, power_tracking_file, row.names = FALSE)
    write.csv(pvalue_tracking, pvalue_tracking_file, row.names = FALSE)

    # Create and save visualizations if ggplot2 is available
    if (requireNamespace("ggplot2", quietly = TRUE)) {
      # Plot power vs simulations
      power_plot <- ggplot2::ggplot(power_tracking,
                            ggplot2::aes(x = simulations,
                                         y = power,
                                         color = comparison,
                                         group = comparison)) +
        ggplot2::geom_line(linewidth = 1) +
        ggplot2::geom_point(size = 2) +
        ggplot2::geom_hline(yintercept = target_power, linetype = "dashed", color = "red") +
        ggplot2::labs(
          title = "Power vs Number of Simulations",
          subtitle = paste("Target Power =", target_power),
          x = "Number of Simulations",
          y = "Estimated Power",
          color = "Comparison"
        ) +
        ggplot2::theme_classic() +
        ggplot2::theme(
          legend.position = "bottom",
          plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = ggplot2::element_text(hjust = 0.5)
        )

      # Save power plot
      ggplot2::ggsave(power_plot_file, power_plot, width = 10, height = 6, dpi = 300)

      # Create p-value bar plot
      # Get the final p-values
      final_pvalues <- pvalue_tracking[pvalue_tracking$simulations == max(pvalue_tracking$simulations), ]

      # Transform p-values to -log10 scale for better visualization
      final_pvalues$neg_log10_p_raw <- -log10(final_pvalues$p_raw)
      final_pvalues$neg_log10_p_adj <- -log10(final_pvalues$p_adj)

      # Reshape for plotting
      pvalue_plot_data <- data.frame(
        comparison = rep(final_pvalues$comparison, 2),
        p_type = c(rep("Raw p-value", nrow(final_pvalues)), rep("Adjusted p-value", nrow(final_pvalues))),
        neg_log10_p = c(final_pvalues$neg_log10_p_raw, final_pvalues$neg_log10_p_adj)
      )

      # Create the bar plot
      pvalue_plot <- ggplot2::ggplot(pvalue_plot_data,
                             ggplot2::aes(x = comparison,
                                          y = neg_log10_p,
                                          fill = p_type)) +
        ggplot2::geom_bar(stat = "identity",
                          position = ggplot2::position_dodge(width = 0.9),
                          width = 0.8) +
        ggplot2::geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "red") +
        ggplot2::labs(
          title = "Significance of Comparisons",
          subtitle = paste("After", J_final, "Simulations"),
          x = "Comparison",
          y = "-log10(p-value)",
          fill = "P-value Type"
        ) +
        ggplot2::theme_classic() +
        ggplot2::theme(
          axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
          legend.position = "bottom",
          plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = ggplot2::element_text(hjust = 0.5)
        ) +
        ggplot2::annotate("text", x = 1, y = -log10(0.05) + 0.2,
                          label = "α = 0.05", color = "red")

      # Save p-value plot
      ggplot2::ggsave(pvalue_plot_file, pvalue_plot, width = 10, height = 6, dpi = 300)

      if (verbose) {
        cat("\\nResults saved to CSV files:\\n")
        cat("- ", results_file, "\\n")
        cat("- ", summary_file, "\\n")
        cat("- ", power_tracking_file, "\\n")
        cat("- ", pvalue_tracking_file, "\\n")
        cat("\\nVisualizations saved:\\n")
        cat("- ", power_plot_file, "\\n")
        cat("- ", pvalue_plot_file, "\\n")
      }
    } else {
      if (verbose) {
        cat("\\nResults saved to CSV files, but visualizations could not be created.\\n")
        cat("Install the 'ggplot2' package to enable visualization features.\\n")
      }
    }
  }

  # Return results
  return(list(
    J_final = J_final,
    P_raw = P_raw,
    P_adj = P_adj,
    Stats = Stats,
    P_achieved = P_achieved,
    results = results_agg,
    comparison_pairs = comparison_pairs,
    mdes = mdes,
    summary_table = summary_table,
    power_tracking = power_tracking,
    pvalue_tracking = pvalue_tracking
  ))
}

# Example usage
if (FALSE) {  # Set to TRUE to run the example
  # Define a simple simulation function
  sim_func <- function(seed) {
    set.seed(seed)
    # Simulate data and fit models
    # Return performance metrics as a data frame
    data.frame(
      model_a_rmse = rnorm(1, 0.5, 0.1),
      model_b_rmse = rnorm(1, 0.6, 0.1),
      model_a_coverage = rnorm(1, 0.9, 0.05),
      model_b_coverage = rnorm(1, 0.85, 0.05)
    )
  }

  # Define comparison pairs
  comparison_pairs <- list(
    c("model_a_rmse", "model_b_rmse"),
    c("model_a_coverage", "model_b_coverage")
  )

  # Define minimum detectable effect sizes
  # Negative for RMSE (lower is better), positive for coverage (higher is better)
  mdes <- c(-0.1, 0.05)

  # Run TISCA
  results <- run_tisca(
    sim_func = sim_func,
    comparison_pairs = comparison_pairs,
    mdes = mdes,
    target_power = 0.80,
    alpha = 0.05,
    batch_size = 10,
    initial_count = 20,
    verbose = TRUE,
    save_results = TRUE
  )

  # Access results
  cat("\\nRequired number of simulations:", results$J_final, "\\n")
  print(results$summary_table)
}

3.3 Addressing Multiple Comparisons
Researchers often compare a new model against benchmarks across multiple performance metrics (e.g., PEHE, RMSEAT E, Coverage) or against multiple different benchmarks simultaneously. Performing multiple hypothesis tests increases the probability
of making at least one Type I error (a false positive finding) across the family of tests –
the ”multiple comparisons problem” (Menon, 2019; Streiner & Norman, 2011).
TISCA acknowledges this issue by offering optional p-value adjustment procedures
as an input parameter. The available methods include:
• Bonferroni correction: A simple but often overly conservative method that
controls the Family-Wise Error Rate (FWER) by multiplying each p-value by
the number of tests (or equivalently, dividing α by the number of tests) (Dunn,
1961).
• Holm’s method (Holm-Bonferroni): A step-down procedure that also controls
the FWER but is uniformly more powerful than the standard Bonferroni correction (Holm, 1979).
• Benjamini-Hochberg (BH) procedure: Controls the False Discovery Rate (FDR)
– the expected proportion of rejected null hypotheses that are actually true (Benjamini & Hochberg, 1995). This is generally less conservative and more powerful than FWER-controlling methods, making it suitable when controlling the
proportion of false positives among the significant findings is the primary goal.
The choice of correction method depends on the researcher’s specific goals and tolerance for Type I versus Type II errors. While these methods provide established ways
to handle multiple tests, they are not perfect solutions; FWER methods can be overly
strict, potentially masking true effects, while FDR control allows for some false positives among the declared significant results (Menon, 2019). The default setting of
‘”none”‘ assumes that only a few tests will be performed (say two or three), yet if
more tests are performed, then we advise using a correction method.

Your goal is to assist researchers in structuring their simulation code, choosing appropriate parameters (like MDEs based on pilot data), and debugging issues related to TISCA, based on the information provided above.
You should also consider any R code that the user might paste directly into their message.

Respond to the user message with helpful and informative guidance. Be specific and provide code examples where necessary.
If the user provides R code, you should assume it's related to their `sim_func` or how they intend to call `run_tisca`.
Be mindful of R syntax when suggesting R code, and Python syntax for the Streamlit app itself.
Start the conversation by introducing yourself as the TISCA helper bot.
"""

# --- Model and Chat Initialization ---
model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=TISCA_SYSTEM_PROMPT,
    generation_config=genai.types.GenerationConfig(
        # Controls randomness. Lower for more deterministic, higher for more creative.
        temperature=0.7, # Adjust as needed
        # Add other parameters like top_p, top_k if desired
    )
)

# Initialize chat history in session state
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# --- UI ---
st.set_page_config(layout="wide", page_title="TISCA Helper")

st.title("🧪 TISCA Helper Bot")
st.markdown("""
Welcome! I'm here to help you understand and implement the **Test-Informed Simulation Count Algorithm (TISCA)**
in your research.

**What TISCA Does:**
TISCA provides a statistically rigorous way to determine the number of simulation replications (J) needed for your study.
Instead of picking an arbitrary number, TISCA iteratively runs simulations and performs Welch’s t-tests with power analysis
until a pre-specified power (e.g., 0.8) is achieved for detecting a user-defined minimum detectable effect size (MDE)
at a given significance level (α). This ensures your comparisons are robust and resource-efficient.

**How I Can Help:**
*   **Structuring your R simulation code** to be compatible with TISCA's `sim_func` input.
*   Guidance on choosing appropriate **TISCA parameters** (like MDEs, `comparison_pairs`, `alpha`, `target_power`).
*   Helping you **interpret TISCA outputs**.
*   Assisting with **debugging R code** related to your TISCA implementation.

Simply type your questions or R code snippets below!
*(Note: This tool uses a Large Language Model. While it aims to be helpful, always critically evaluate its advice and refer to the TISCA paper and documentation for definitive guidance.)*
""")
st.markdown("---")

# Display chat history
for message in st.session_state.chat_session.history:
    role = "user" if message.role == "user" else "assistant"
    with st.chat_message(role):
        # Use textwrap for better formatting of assistant messages
        if role == "assistant":
            st.markdown(message.parts[0].text) # Assuming simple text parts for now
        else:
            st.markdown(message.parts[0].text)


# Get user input
user_prompt = st.chat_input("Ask TISCA Helper about your simulation or R code...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    try:
        with st.spinner("TISCA Helper is thinking..."):
            response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(response.text) # Display LLM's response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        # Optionally, you could try to recover or restart the chat session here
        # For now, just show an error.