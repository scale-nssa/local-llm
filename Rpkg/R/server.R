#' Programmatic launcher for a local llama-server
#'
#' Launches a llama-server subprocess and returns a handle list with utilities.
#' No YAML required; configuration is passed as arguments.
#'
#' @param model_path Path to model .gguf file.
#' @param n_ctx Context length.
#' @param n_gpu_layers Number of GPU layers.
#' @param port Port to bind (default 8080).
#' @param host Host interface (default 127.0.0.1).
#' @param threads Optional thread count.
#' @param http_threads Optional HTTP thread count.
#' @param slots Optional slots.
#' @param cors Optional CORS value.
#' @param log_disable Logical disable logging output at server side.
#' @param log_colors Logical override to disable colors (FALSE -> --log-colors=0).
#' @param quiet Verbosity flag.
#' @param verbose Verbosity flag.
#' @param api_key Optional API key to require.
#' @param extra_args Character vector of extra raw args.
#' @param env Named character vector of environment vars to set.
#' @param stream_logs Logical; stream server output to console.
#' @param health_timeout Seconds to wait for /health.
#' @return A handle list with elements: process (processx object), port, is_alive(), terminate(), kill(), wait().
#' @export
start_server <- function(model_path,
                         n_ctx,
                         n_gpu_layers,
                         port = 8080,
                         host = "127.0.0.1",
                         threads = NULL,
                         http_threads = NULL,
                         slots = NULL,
                         cors = NULL,
                         log_disable = FALSE,
                         log_colors = NULL,
                         quiet = FALSE,
                         verbose = FALSE,
                         api_key = NULL,
                         extra_args = NULL,
                         env = NULL,
                         stream_logs = TRUE,
                         health_timeout = 30) {
  # --- validation ---
  if (!is.character(model_path) || length(model_path) != 1L || !nzchar(model_path))
    stop("model_path must be a non-empty character scalar", call. = FALSE)
  if (!file.exists(model_path))
    stop("model_path does not exist: ", model_path, call. = FALSE)
  scalar_pos_int <- function(x, nm) {
    if (length(x) != 1L || is.na(x) || x <= 0 || as.integer(x) != x)
      stop(nm, " must be a positive integer", call. = FALSE)
  }
  scalar_pos_int(n_ctx, "n_ctx")
  scalar_pos_int(n_gpu_layers, "n_gpu_layers")
  scalar_pos_int(port, "port")
  if (port > 65535) stop("port must be <= 65535", call. = FALSE)
  check_opt <- function(x, nm) {
    if (!is.null(x)) {
      if (length(x) != 1L || is.na(x) || x <= 0 || as.integer(x) != x)
        stop(nm, " must be a positive integer if provided", call. = FALSE)
    }
  }
  check_opt(threads, "threads")
  check_opt(http_threads, "http_threads")
  check_opt(slots, "slots")
  if (!is.null(extra_args) && !is.character(extra_args))
    stop("extra_args must be a character vector if provided", call. = FALSE)
  if (!is.null(env)) {
    if (!is.vector(env) || is.null(names(env)) || any(!nzchar(names(env))))
      stop("env must be a named character vector", call. = FALSE)
  }
  if (!requireNamespace("processx", quietly = TRUE)) stop("Install 'processx'")
  if (!requireNamespace("httr2", quietly = TRUE)) stop("Install 'httr2'")

  bin <- Sys.which("llama-server")
  if (identical(bin, "")) stop("'llama-server' not found in PATH", call. = FALSE)

  if (!is.null(env)) {
    do.call(Sys.setenv, lapply(as.list(env), as.character))
  }
  # macOS Homebrew path convenience
  if (Sys.info()[["sysname"]] == "Darwin") {
    Sys.setenv(PATH = paste("/opt/homebrew/bin", Sys.getenv("PATH", ""), sep=":"))
  }

  args <- c(
    "-m", model_path,
    "-c", as.character(n_ctx),
    "-ngl", as.character(n_gpu_layers),
    "--port", as.character(port),
    "--host", host
  )
  if (!is.null(threads))      args <- c(args, "-t", as.character(threads))
  if (!is.null(http_threads)) args <- c(args, "--threads-http", as.character(http_threads))
  if (!is.null(slots))        args <- c(args, "--slots", as.character(slots))
  if (!is.null(cors) && nzchar(cors)) args <- c(args, "--cors", cors)
  if (isTRUE(log_disable))    args <- c(args, "--log-disable")
  if (identical(log_colors, FALSE)) args <- c(args, "--log-colors=0")
  if (isTRUE(quiet))          args <- c(args, "-q")
  if (isTRUE(verbose))        args <- c(args, "-v")
  if (!is.null(api_key))      args <- c(args, "--api-key", api_key)
  if (!is.null(extra_args) && length(extra_args)) args <- c(args, extra_args)

  proc <- processx::process$new(
    bin, args,
    stdout = "|",
    stderr = "stdout",
    cleanup = TRUE,
    cleanup_tree = TRUE
  )

  is_alive <- function() proc$is_alive()
  kill <- function() { if (proc$is_alive()) try(proc$kill(), silent = TRUE) }
  terminate <- function(timeout = 10) {
    if (!proc$is_alive()) return(invisible())
    try(proc$signal(15L), silent = TRUE) # SIGTERM
    proc$wait(timeout)
    if (proc$is_alive()) kill()
    invisible()
  }
  wait <- function() proc$wait()

  # wait for health
  url <- sprintf("http://127.0.0.1:%s/health", port)
  start <- Sys.time()
  repeat {
    if (!proc$is_alive()) {
      out <- try(rawToChar(proc$read_output()), silent = TRUE)
      stop("Server exited early before health check. Output:\n", out, call. = FALSE)
    }
    ok <- try({
      resp <- httr2::request(url) |>
        httr2::req_timeout(0.5) |>
        httr2::req_perform()
      httr2::resp_status(resp) == 200
    }, silent = TRUE)
    if (isTRUE(ok)) break
    if (as.numeric(difftime(Sys.time(), start, units = "secs")) > health_timeout) {
      terminate()
      stop(sprintf("Server did not become healthy within %ss", health_timeout), call. = FALSE)
    }
    Sys.sleep(0.25)
  }

  if (isTRUE(stream_logs) && !isTRUE(log_disable)) {
    log_thread <- parallel::mcparallel({
      repeat {
        if (!proc$is_alive()) break
        pr <- processx::poll(list(proc), 200)
        if (identical(pr[[1]], "ready")) {
          buf <- proc$read_output()
          if (length(buf)) cat(gsub("\r", "", rawToChar(buf), fixed = TRUE))
        }
      }
    })
  } else {
    log_thread <- NULL
  }

  structure(list(
    process = proc,
    port = port,
    is_alive = is_alive,
    terminate = terminate,
    kill = kill,
    wait = wait,
    log_thread = log_thread
  ), class = "localllm_server")
}
