library(dplyr)
library(rpart)

calc_mae_reduction <- function(y_hat, y_actual) {
  model_error <- mean(abs(y_hat - y_actual))
  baseline <- mean(y_actual, na.rm = TRUE)
  baseline_error <-  mean(abs(baseline - y_actual))
  result <- 1 - model_error/baseline_error
  result <- max(0.0, min(result, 1.0))
  round(100*result, 2)
}
calc_misclass_reduction <- function(y_hat, y_actual) {
  tab <- table(y_hat, y_actual)
  model_error <- 1 - sum(diag(tab))/sum(tab)
  majority_class <- names(which.max(table(y_actual)))
  baseline.preds <- rep(majority_class, length(y_actual))
  baseline_error <- mean(baseline.preds != y_actual)
  result <- 1 - model_error/baseline_error
  result <- max(0.0, min(result, 1.0))
  round(100*result, 2)
}
x2y <- function(x, y) {
  
  results <- list()
  results$x2y <- 0.0
  missing <-  is.na(x) | is.na(y)
  results$perc_of_obs <- round(100* (1 - sum(missing)/length(x)), 2)
  
  x <- x[!missing]
  y <- y[!missing]
  
  
  if (length(unique(x)) > 1 &
      length(unique(y)) > 1) {
    # if y is continuous
    if (is.numeric(y)) {
      preds <- predict(rpart(y ~ x, method = "anova"), type = 'vector')
      results$x2y <- calc_mae_reduction(preds, y)
    }
    # if y is categorical
    else {
      preds <- predict(rpart(y ~ x, method = "class"), type = 'class')
      results$x2y <- calc_misclass_reduction(preds, y)
    }
  }
  results
}
all_pairs_x2y <- function(d) {
  pairs <- combn(ncol(d), 2)
  n <- dim(pairs)[2]
  results <- data.frame(x = c(names(d)[pairs[1,]], names(d)[pairs[2,]]),
                        y = c(names(d)[pairs[2,]], names(d)[pairs[1,]]),
                        x2y = rep(0.00, n*2),
                        perc_of_obs = rep(0.00, n*2))
  for (i in 1:n) {
    x <- d %>% pull(pairs[1,i])
    y <- d %>% pull(pairs[2,i])
    results[i,3:4] <- x2y(x,y)
    results[i+n, 3:4] <- x2y(y,x)
  }
  results <- results %>% arrange(desc(x2y), desc(perc_of_obs))
  results
}
