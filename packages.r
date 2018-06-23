pkgs <- c(
  "devtools",
  "tidyverse",
  "data.table",
  "caret",
  "rjags",
  "e1071",
  "gbm",
  "tableone",
  "tensorflow",
  "keras"
)

for (pkg in pkgs) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) install.packages(pkg)
}
