### Loop Diagnostic Script.R ###

# recursively searching for external pointers, skipping elements that cannot be subset
find_ptr <- function(x, path = "", seen = new.env()) {
  out <- character(0)
  if (typeof(x) == "externalptr") out <- path
  if (is.environment(x)) {
    id <- format(x)
    if (exists(id, envir = seen, inherits = FALSE)) return(out)
    assign(id, TRUE, envir = seen)
    for (n in ls(x, all.names = TRUE)) {
      el <- tryCatch(get(n, envir = x), error = function(e) NULL)
      out <- c(out, find_ptr(el, paste0(path, "$", n), seen))
    }
  } else if (is.list(x) || is.pairlist(x)) {
    for (i in seq_along(x)) {
      el <- tryCatch(x[[i]], error = function(e) NULL)
      out <- c(out, find_ptr(el, paste0(path, "[[", i, "]]"), seen))
    }
  }
  for (a in names(attributes(x))) {
    el <- tryCatch(attr(x, a, exact = TRUE), error = function(e) NULL)
    out <- c(out, find_ptr(el, paste0(path, "@attr:", a), seen))
  }
  out
}

# scanning every global object and printing paths of external pointers
hits <- unlist(lapply(ls(globalenv(), all.names = TRUE), function(n) {
  el <- tryCatch(get(n, envir = globalenv()), error = function(e) NULL)
  find_ptr(el, n)
}))
print(hits)

# listing loaded namespaces that commonly hold background handles
intersect(loadedNamespaces(), c("later", "promises", "httpuv", "processx", "callr", "future", "parallelly", "DBI", "torch", "xgboost", "lightgbm", "reticulate", "plumber", "shiny"))

rm(xgb_fit)
#rm(xgb_fit, tree_fit, bw)
rm(list = ls(pattern = "^p_"))
invisible(gc())

unlist(lapply(ls(globalenv(), all.names = TRUE), function(n) {
  el <- tryCatch(get(n, envir = globalenv()), error = function(e) NULL)
  find_ptr(el, n)
}))

