getwd()

reddit <- read.csv('reddit.csv')

str(reddit)
View(reddit)
table(reddit$employment.status)
summary(reddit)
levels(reddit$age.range)

install.packages("ggplot2")
library(ggplot2)
qplot(data = reddit, x = age.range)
qplot(data = reddit, x = income.range)

is.factor(reddit$age.range)
age.f <- factor(reddit$age.range, levels = c("Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 or Above"), ordered = T)
age.f
qplot(data = reddit, x = age.f)
age.f <- ordered(age.f, levels = c("Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65 or Above"))
age.f
qplot(data = reddit, x = age.f)

levels(reddit$income.range)
income.f <- factor(reddit$income.range, levels = c("Under $20,000", "$20,000 - $29,999", "$30,000 - $39,999", "$40,000 - $49,999", "$50,000 - $69,999", "$70,000 - $99,999", "$100,000 - $149,999", "$150,000 or more"))
qplot(data = reddit, x = income.f)
