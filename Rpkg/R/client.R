#' Chat completion against local llama-server
#'
#' @param prompt User prompt string.
#' @param model Model name (default 'local').
#' @param system_prompt Optional system message.
#' @param max_tokens Optional max tokens.
#' @param temperature Optional temperature.
#' @param grammar Optional raw GBNF grammar string.
#' @return Character scalar response.
#' @export
get_response <- function(prompt,
                         model = "local",
                         system_prompt = NULL,
                         max_tokens = NULL,
                         temperature = NULL,
                         grammar = NULL) {
  if (!requireNamespace("httr2", quietly = TRUE)) stop("Install 'httr2'")
  if (!requireNamespace("jsonlite", quietly = TRUE)) stop("Install 'jsonlite'")

  base_url <- Sys.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1")
  api_key  <- Sys.getenv("LLAMA_API_KEY",  "none")

  msgs <- list()
  if (!is.null(system_prompt)) msgs <- append(msgs, list(list(role="system", content=system_prompt)))
  msgs <- append(msgs, list(list(role="user", content=prompt)))

  body <- list(model = model, messages = msgs)
  if (!is.null(max_tokens))  body$max_tokens  <- as.integer(max_tokens)
  if (!is.null(temperature)) body$temperature <- as.numeric(temperature)
  if (!is.null(grammar))     body$grammar     <- as.character(grammar)

  json <- jsonlite::toJSON(body, auto_unbox = TRUE, digits = NA)

  req <- httr2::request(paste0(base_url, "/chat/completions")) |>
    httr2::req_headers(Authorization = paste("Bearer", api_key),
                       "Content-Type" = "application/json") |>
    httr2::req_body_raw(json, type = "application/json")

  resp <- httr2::req_perform(req)
  out  <- httr2::resp_body_json(resp, simplifyVector = TRUE)
  content <- try(out$choices$message$content, silent = TRUE)
  if (inherits(content, "try-error") || is.null(content)) stop("No content in response")
  trimws(content)
}

#' Load a GBNF grammar file
#' @param path Path to .gbnf file.
#' @return Character scalar containing file contents.
#' @export
grammar <- function(path) {
  paste(readLines(path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
}
