library(rentrez)
library(stringr)
library(textclean)

sample_str = " A systematic search will be performed in PubMed, Embase, and the Cochrane Library (CENTRAL), using the following search query: ('out-of-hospital cardiac arrest' OR 'OHCA') AND ('MIRACLE 2' OR 'OHCA' OR 'CAHP' OR 'C-GRAPH' OR 'SOFA' OR 'APACHE' OR 'SAPS’ OR ’SWAP’ OR ’TTM’)."

search_clean <- function(sample_str){
  #' Function to isolate the search string from full text by finding longest
  #' parenthetical phrase connected by ands and ors
  #' Note: must have matching parentheses
  
  # Subset into words and mark those with ands/ors and ()s
  words = strsplit(sample_str, split = " ")[[1]]
  start_p = sapply(words, function(x) grepl("\\(", x))
  end_p = sapply(words, function(x) grepl("\\)", x))
  and_or = sapply(words, function(x) return( grepl("AND", toupper(x)) | grepl("OR", toupper(x))))

  # Iterate through words and create potential phrases to return
  pot_phrases = vector(mode="character")
  curr_search = 1
  curr_looking = FALSE
  i = 1
  count = 0
  while (i <= length(words)){
    # Find if still looking for end parenthesis
    count = count+start_p[i]-end_p[i]

    # If started parentheses update current index unless already looking
    if (start_p[i] == TRUE & curr_looking == FALSE){
      curr_looking = TRUE
      curr_search = i
    }

    # If ended parentheses check for and/or to continue or, also check if at end
    if ((count == 0 & curr_looking == TRUE) | 
        (i == length(words) & curr_looking)){

      if (((i+2) <= length(words)) & and_or[i+1] & start_p[i+2]){
        # Continues with and/or
        i = i + 2 # skip and/or
      } else if (((i+1) <= length(words)) & and_or[i+1] & start_p[i+1]){
        # Continues with and or together in one word
        i = i + 1
      } else{
        # Does not continue - end phrase
        pot_phrases = c(pot_phrases, paste(words[curr_search:i], collapse = " "))
        curr_looking = FALSE
        i = i+1
      }
    } else{
      # Continue to next
      i = i + 1
    }
  }

  # Only look at phrases with at least one and/or
  cont_and_or = sapply(pot_phrases, function(x) grepl("AND", toupper(x)) | grepl("OR", toupper(x)))
  pot_phrases = pot_phrases[cont_and_or==TRUE]
  
  # Find longest phrase and characters before first ( and any html
  if (length(pot_phrases) == 0){
    result = NA
  } else{
    result = pot_phrases[which.max(nchar(pot_phrases))]
    result = substr(result, str_locate_all(pattern = "[(]", result)[[1]][1], nchar(result))
    all_end = str_locate_all(pattern = "[)]", result)[[1]]
    if (nrow(all_end) > 0){
      result = substr(result, 1, all_end[nrow(all_end),2])
    }
    result = replace_html(result)
  }
  
  return(result)
}




