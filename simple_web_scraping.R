# load the usual packages
library(dplyr)
library(tidyr)
library(ggplot2)

#
# Example 1: Scraping a table of numbers from a Wikipedia page
#

# the URL of the wikipedia page 
url <- "https://en.wikipedia.org/w/index.php?title=2020_coronavirus_outbreak_in_the_United_States&oldid=944107102"

# install the rvest package. This is a one-time operation.
# install.packages("rvest")

# load rvest package
library(rvest)

# read the page by calling the read_html function with the URL of the web page
page <- read_html(url)

page %>% 
  # grab all the tables of class 'wikitable'.
  html_nodes("table.wikitable") %>% 
  # select the 2nd table
  .[[2]] %>% 
  # convert table into a dataframe and save in the variable 'covid_counts'.
  html_table(fill = TRUE) -> covid_counts

names(covid_counts) <- covid_counts[1,]
covid_counts <- covid_counts[-1,]

covid_counts <- covid_counts[,-(56:61)]

# by using the 'head' function with a negative number n
# you can get all but the last n
# this is handy since you don't need to calculate the # of rows
# and subtract n etc.

covid_counts <- head(covid_counts, -5)

covid_counts %>%
  # convert text to legit R date objects
  mutate(Date = as.Date(Date, format = "%b%d")) %>%
  # convert all columns except the 'Date' column to numeric
  mutate_at(.vars = -1, as.numeric) %>%
  # replace blanks with zeros
  replace(., is.na(.), 0.0) -> covid_counts

covid_counts %>%
  pivot_longer(cols = -Date,
               names_to = "State" ,
               values_to = "Counts") %>%
  arrange(State, Date) %>% 
  # group by State so that the cumultive 
  # calculation is for each state
  group_by(State) %>%
  # use the handy 'cumsum' function to
  # "cumulatively sum" :-)
  mutate(Cumulative = cumsum(Counts)) %>%
  # undo the group-by and save it back
  ungroup -> covid_counts

covid_counts %>%
  # select only March data
  filter(Date >= "2020-03-01") %>%
  # remove the days before the first case was reported
  filter(Cumulative > 0) %>%
  # filter for New England states
  filter(State %in% c('MA',
                      'CT',
                      'NH',
                      'RI',
                      'VT',
                      'ME')) %>%
  ggplot(aes(x = Date, y = Cumulative, color = State)) +
  geom_point() +
  geom_line() 


#
# Example 2: Scraping a price from an Amazon product page
#

url <- "https://www.amazon.com/gp/product/B010S6SG3S/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1"

# read in the URL into an R variable
page <- read_html(url)

page %>% 
  # grab tag that contains the price
  html_nodes("span#priceblock_ourprice") %>% 
  # extract the text content from within the tag
  html_text


