# PS4b_Zhou.R
# Note: sparklyr is not available for R version 4.0.2 on OSCER.
# The following code is written based on the assignment instructions
# but could not be executed due to package unavailability.

library(sparklyr)
library(tidyverse)

# Step 4: Connect to Spark
sc <- spark_connect(master = "local")

# Step 5: Create tibble from iris data
df1 <- as_tibble(iris)

# Step 6: Copy to Spark
df <- copy_to(sc, df1)

# Step 7: Compare classes
print(class(df1))
print(class(df))

# Step 8: Compare column names
print(colnames(df1))
print(colnames(df))

# Step 9: Select operation
df %>% select(Sepal_Length, Species) %>% head %>% print

# Step 10: Filter operation
df %>% filter(Sepal_Length > 5.5) %>% head %>% print

# Step 11: Select + Filter combined
df %>% filter(Sepal_Length > 5.5) %>% select(Sepal_Length, Species) %>% head %>% print

# Step 12: Group_by + summarize
df %>% group_by(Species) %>% summarize(mean = mean(Sepal_Length), count = n()) %>% head %>% print

# Step 13: Arrange
df2 <- df %>% group_by(Species) %>% summarize(mean = mean(Sepal_Length), count = n()) %>% head
df2 %>% arrange(Species) %>% head %>% print
