#' Default server configuration list
#' @return Named list with placeholder / default values.
#' @export
default_server_config <- function() {
  list(
    model_path = "<REPLACE_ME>",
    n_ctx = 4096,
    n_gpu_layers = 50,
    port = 8080
  )
}

#' Write a server.yaml-like file programmatically
#'
#' @param path File path to write.
#' @param overwrite Logical allow overwrite.
#' @param ... Named overrides for default_server_config fields.
#' @export
write_server_config <- function(path = "server.yaml", overwrite = FALSE, ...) {
  if (!requireNamespace("yaml", quietly = TRUE)) stop("Install 'yaml'")
  if (file.exists(path) && !overwrite) stop("File exists: ", path, call. = FALSE)
  cfg <- modifyList(default_server_config(), list(...))
  yaml::write_yaml(cfg, path)
  invisible(path)
}
