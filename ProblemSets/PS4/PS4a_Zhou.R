# PS4a_Zhou.R

# 5a: Download JSON file
system('wget -O dates.json "https://www.vizgr.org/historical-events/search.php?format=json&begin_date=00000101&end_date=20240209&lang=en"')

# 5b: Print file to console
system('cat dates.json')

# 5c: Convert to data frame
library(jsonlite)
library(tidyverse)

mylist <- fromJSON('dates.json')
mydf <- bind_rows(mylist$result[-1])

# 5d: Check object types
print(class(mydf))
print(class(mydf$date))

# 5e: List first 6 rows
print(head(mydf))
