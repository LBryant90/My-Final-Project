# Loading Libraries

library("caret")
library("magrittr")
library("dplyr")
library("tidyr")
library("lmtest")
library("popbio")
library("e1071")
library("ggplot2")


# Loaded in Stack overflow dataset

# Questions: Can we predict the hiring outcome of an applicant based on their
# Education level, Number of Computer Skills and whether they worked with the 
# Most popular Languages

# Our predictor variable Employed has already been recoded into 0 and 1

# Dropping Missing Variables from our data
Hiring <- na.omit(stackoverflow_full)

# Recoding our IV Columns EdLevel and Have Worked with to set up our Logistic Model

Hiring$EdLevel <- factor(Hiring$EdLevel, levels = c("NoHigherEd", "Undergraduate", "Master", "PhD", "Other"))

# Checking the levels of the EducationLevel factor
levels(Hiring$EdLevel)

# Recoding the factor levels to numerical values
Hiring$EducationLevelNum <- as.integer(Hiring$EdLevel)


# Creating a new column to store the recoded age categories
Hiring$Age <- ifelse(Hiring$Age < 35, "<35", ">35")

# Converting the Age Category column to a factor
Hiring$Age <- factor(Hiring$Age, levels = c("<35", ">35"))

# Recoding the factor levels to numerical values
Hiring$AgeFactor <- as.integer(Hiring$Age)


# Splitting the HaveWorkedWith column into a list of programming languages
language_list <- strsplit(Hiring$HaveWorkedWith, ";")

# Getting the unique programming languages from the list for separation
unique_languages <- unique(unlist(language_list))

# Create binary dummy indicator variables for each programming language
for(language in unique_languages) {
  Hiring[[language]] <- as.integer(sapply(language_list, function(x) language %in% x))
}


# Getting the Sum of the binary variables for the languages
language_sums <- colSums(Hiring[, unique_languages])

# Combining the language sums with the Employed column
language_sums <- cbind(language_sums, Employed = Hiring$Employed)


# Sort the programming languages by the sum of hires in descending order
top_languages <- language_sums[order(language_sums[, "Employed"], decreasing = TRUE), ]

# Selecting the top 10 programming languages
top_10_languages <- head(top_languages, 10)

# Printing the top 10 programming languages
print(top_10_languages)

# Appears the Languages the most contributed to an applicant being hired in order are
# C++, Python, Git, PostgreSQL, Bash/Shell, HTML/CSS, Javascript, Node.js,SQL
# and Typescript


top_10_languages <- head(unique_languages, 10)
print(top_10_languages)


### Subsetting my data to include only the columns useful to the analysis


# Subset the dataframe to include only the specified columns
hiring_stats1 <- Hiring[, c('AgeFactor', 'EducationLevelNum', 'Gender', 'MentalHealth', 'YearsCode','PreviousSalary', 'ComputerSkills', 'Employed', 'C++', 'Python', 'Git', 'PostgreSQL', 'Bash/Shell', 'HTML/CSS', 'JavaScript', 'Node.js', 'SQL', 'TypeScript')]

#Combining languages into one column that will display how many of the top 10 languages the
# applicant has worked with

# Assuming your dataframe is named "hiring_stats1"

# Selecting the programming language columns
programming_languages <- hiring_stats1[, c('C++', 'Python', 'Git', 'PostgreSQL', 'Bash/Shell', 'HTML/CSS', 'JavaScript', 'Node.js', 'SQL', 'TypeScript')]

# Creating a new column "HaveWorkedWith10" by totaling the binary indicator values for the programming languages
hiring_stats1$HaveWorkedWith10 <- rowSums(programming_languages)


#Dropping the Individual programming language columns
hiring_stats1 <- hiring_stats1[, -which(names(hiring_stats1) %in% programming_languages)]

# Subset the dataframe to include only the specified columns
hiring_stats2 <- hiring_stats1[, c('AgeFactor', 'EducationLevelNum', 'Gender', 'MentalHealth', 'YearsCode','PreviousSalary', 'ComputerSkills', 'Employed', 'HaveWorkedWith10')]

# Calculating the min and max skills for the applicants to try and reduce it to s similar scale for models
min_skills <- min(hiring_stats2$ComputerSkills)
max_skills <- max(hiring_stats2$ComputerSkills)

# Skills range from 1(min)-107(max)
hiring_stats2$ScaledComputerSkills <- (hiring_stats2$ComputerSkills - min_skills) / (max_skills - min_skills)


#Obtaining a proficency score for the HaveWorkedWith10 Column
min_mastery <- min(hiring_stats2$HaveWorkedWith10)
max_mastery <- max(hiring_stats2$HaveWorkedWith10)

# Proficiency in the TOP 10 programming languages range from 0 (min)-10(max)

hiring_stats2$ProficiencyScore <- (hiring_stats2$HaveWorkedWith10 - min_mastery) / (max_mastery - min_mastery) * 100

# Running Our Basic Logistic Regression Model
mylogitModel <- glm(Employed ~ EducationLevelNum+ ScaledComputerSkills + ProficiencyScore, data=hiring_stats2, family="binomial")
summary(mylogitModel)

# Moving to make predictions
Forecast <- predict(mylogitModel, type = "response")
hiring_stats2$Predicted <- ifelse(Forecast > .5, "pos", "neg")

#Recoding the Predicted Variable Column

hiring_stats2$PredictedR <- NA
hiring_stats2$PredictedR[hiring_stats2$Predicted=='pos'] <- 1
hiring_stats2$PredictedR[hiring_stats2$Predicted=='neg'] <- 0

# Convert Variables to Factors
hiring_stats2$PredictedR <- as.factor(hiring_stats2$PredictedR)
hiring_stats2$Employed <- as.factor(hiring_stats2$Employed)

# Creating a Confusion Matrix
confus_matr <- caret::confusionMatrix(hiring_stats2$PredictedR, hiring_stats2$Employed)
confus_matr

# Our sample size assumption has been met as all these values exceed 5.
# Looks like our model has a 78% accuracy rate which is not too bad and means
# we are getting it right more than we are getting it wrong


# Logit Linearity
hiring_stats3 <- hiring_stats2 %>% 
  dplyr::select_if(is.numeric)

predictors <- colnames(hiring_stats3)

hiring_stats4 <- hiring_stats3 %>%
  mutate(logit=log(Forecast/(1-Forecast))) %>%
  gather(key= "predictors", value="predictor.value", -logit)

# Graph
ggplot(hiring_stats4, aes(logit, predictor.value))+
  geom_point(size=.5, alpha=.5)+
  geom_smooth(method= "loess")+
  theme_bw()+
  facet_wrap(~predictors, scales="free_y")

# Computer Skills has met the assumption but Proficiency Score and Education level have not.
# Will proceed with testing as it is not incredibly paramount to have linearity as
# I believe these variable do have an effect on the outcome it is just not strictly linear.


#MUlticollinearity
cor_matrix <- cor(hiring_stats3[, c("EducationLevelNum", "ScaledComputerSkills", "ProficiencyScore")])

# Printing the correlation matrix
print(cor_matrix)

# There is potential multicollinearity between Computer skills and programming languages
# Which is to be expected but they both contain unique information valuable to
# this project so will proceed with caution. Computer Skills will house all the languages
# an applicant has worked with and HaveWorkedWith10 will have how many of the Top
# 10 languages the applicant had experience with.

# Independent Errors-Graphing
plot(mylogitModel$residuals)

# This is mostly even so I'd say we met this assumption. Will do a Durbin-Watson Test
dwtest(mylogitModel, alternative="two.sided")

# We have met this assumption

# Screening for Outliers
impact <- influence.measures(mylogitModel)
summary(impact)

# Running Logistic Regression and Interpreting the Output
summary(mylogitModel)

#Based on the results I can see that EducationLevelNum, ScaledComputerSkills, and ProficiencyScore 
# have significance in predicting the hiring chances of an applicant. The more
# Computer Skills an individual has it increases their hiring chances by 30%.




















