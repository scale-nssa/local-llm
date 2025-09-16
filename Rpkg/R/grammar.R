#' Generate a multiple-choice grammar
#'
#' Builds a GBNF grammar whose root must be one of the provided choices.
#' @param choices Character vector of options (non-empty).
#' @param save_dir Directory to create/write grammar file.
#' @param name Rule name (identifier).
#' @param thinking Logical: include thinking envelope rules.
#' @return The grammar text (invisibly writes file).
#' @export
multiple_choice_grammar <- function(choices, save_dir, name, thinking = TRUE) {
  if (length(choices) < 1) stop("choices must be non-empty")
  if (!grepl("^[A-Za-z_][A-Za-z0-9_]*$", name)) stop("name must be a valid rule identifier")

  .esc <- function(s) {
    s <- gsub("\\\\", "\\\\\\\\", s)  # backslash
    s <- gsub("\"", "\\\\\"", s)          # quote
    s <- gsub("\n", "\\\\n", s, fixed = TRUE) # newline -> \n
    s <- gsub("\t", "\\\\t", s, fixed = TRUE) # tab -> \t
    s
  }
  alts <- paste(sprintf('"%s"', vapply(choices, function(x) .esc(as.character(x)), character(1))), collapse = " | ")

  if (isTRUE(thinking)) {
    content <- paste0(
      "root ::=  thinkingBlock ", name, "\n",
      "thinkingBlock ::= thinkingStart anychar* thinkingEnd\n",
      "thinkingStart ::= \"<|channel|>analysis<|message|>\" | \"<think>\"\n",
      "thinkingEnd ::= \"<|end|><|start|>assistant<|channel|>final<|message|>\\n\" | \"</think>\\n\"\n",
      name, " ::= ", alts, "\n",
      "anychar ::= [^<]\n"
    )
    filename <- paste0("thinking_", name, ".gbnf")
  } else {
    content <- paste0(
      "root ::=  ", name, "\n",
      name, " ::= ", alts, "\n"
    )
    filename <- paste0(name, ".gbnf")
  }

  if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(content, file.path(save_dir, filename), useBytes = TRUE)
  content
}
