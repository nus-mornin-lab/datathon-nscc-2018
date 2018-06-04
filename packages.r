pkgs <- c(
  "devtools",
  "tidyverse",
  "data.table",
  "caret",
  "tensorflow",
  "keras"
)

for (pkg in pkgs) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) install.packages(pkg)
}
