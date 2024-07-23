library(tidyverse)
library(rentrez)
library(readxl)
library(openxlsx)
source("r scripts/search_cleaning_function.R")

#searches <- read in PROSPERO data for searches with search text in the "searches" field

searches$pubmed = as.vector(sapply(searches$Searches, search_clean))

# Extra data on resulting string - # ands/ors, length string
count_andsors <- function(sample_str){
  words = strsplit(sample_str, split = " ")[[1]]
  and_or = sapply(words, function(x) return( grepl("AND", toupper(x)) | 
                                               grepl("OR", toupper(x))))
  return(sum(and_or))
} 
searches$num_andsors = as.vector(sapply(searches$pubmed, count_andsors))
searches$len_str = nchar(searches$pubmed)
#likelysearches$count = sapply(likelysearches$pubmed, count)

write.xlsx(searches, file = "searches_cleaned_test.xlsx")

