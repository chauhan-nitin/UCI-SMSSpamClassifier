#------------------------- LOADING THE DATASET ----------------------------#
rm(list=ls(all=TRUE))
getwd()
setwd("C:/Users/obc1/Desktop/Projects")
require(data.table)
spam_data <- fread("spam.csv", header = T, na.strings = "")

# Check data to see if there are missing values.
sapply(spam_data, function(x) sum(is.na(x)))

#length(which(!complete.cases(spam_data)))
#Dropping variables with more than 90% of missing values
spam_data <- spam_data[,1:2]
names(spam_data) <- c("Label","SMS")
str(spam_data)

#------------------------- EXPLORING THE DATASET ----------------------------#
# First, let's take a look at distibution of the class labels (i.e., ham vs. spam).
spam_data$Label <- as.factor(spam_data$Label)
prop.table(table(spam_data$Label))
str(spam_data)

#Feature Engineering
spam_data$SMSLength <- nchar(spam_data$SMS)
summary(spam_data$SMSLength)
length(spam_data$SMSLength)
SMSLength <- spam_data$SMSLength

# Visualize distribution with ggplot2, adding segmentation for ham/spam.
library(ggplot2)

ggplot(spam_data, aes(x = SMSLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y = "Text Count", x = "Length of SMS",
       title = "Distribution of Text Lengths with Class Labels")


#------------------------ PREPROCESSING THE DATASET --------------------------#
spam_data$SMS[1]
library(quanteda)
help(package = "quanteda")
# Tokenize Statement.
train.tokens <- tokens(spam_data$SMS, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE,
                       remove_separators = TRUE, ngrams=1)
# Take a look at a specific SMS message and see how it transforms.
train.tokens[[1]]
# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)
train.tokens[[1]]
#Remove Built in Stopwords in English
train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
train.tokens[[1]]
# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[1]]
# Create our first bag-of-words model.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE, verbose = FALSE)
dim(train.tokens.dfm)
docfreq(train.tokens.dfm[1:10,1:50])
new_dfm <- dfm_trim(train.tokens.dfm, sparsity = 0.98)
dim(new_dfm)

# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(new_dfm)
View(train.tokens.matrix[1:20, 1:30])
dim(train.tokens.matrix)
# Investigate the effects of stemming.
colnames(train.tokens.matrix)[1:59]
train.tokens.df <- cbind(Label = spam_data$Label, data.frame(new_dfm))
# Often, tokenization requires some additional pre-processing
names(train.tokens.df)[c(1,23,14)]
# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))
dim(train.tokens.df)
View(train.tokens.df[1:20, 1:61])
# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}
# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}
# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}
# First step, normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
View(train.tokens.df[1:20, 1:100])

# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)
# Lastly, calculate TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])

# Check for incopmlete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
length(incomplete.cases)
# Fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))

# Make a clean data frame using the same process as before.
spam_data <- cbind(Label = spam_data$Label, data.frame(train.tokens.tfidf))
names(spam_data) <- make.names(names(spam_data))
View(spam_data[1:25, 1:25])
gc()
spam_data$SMSLength <- SMSLength
str(spam_data)

#Splitting the dataset into Train and Test into 70-30% ratio
set.seed(1234)
size_train <- nrow(spam_data)
sample_index <- sample.int(size_train, size = floor(0.3*nrow(spam_data)))
testData <- spam_data[sample_index,]
trainData <- spam_data[-sample_index,]
prop.table(table(trainData$Label))
prop.table(table(testData$Label))
str(trainData)
sapply(trainData, function(x) sum(is.na(x)))



#------------------------- SUPPORT VECTOR MACHINE ----------------------------#
library(e1071)
svmfit1 <- svm (Label ~ ., data = trainData, kernel = "linear") # radial svm,
svmfit2 <- svm (Label ~ ., data = trainData, kernel = "radial") # radial svm, scaling turned OFF
print(svmfit1)
compareTable <- table (testData$Label, predict(svmfit1,testData))
misClassificationError <- mean(testData$Label != predict(svmfit1,testData)) # 4.48% misclassification train error
print(paste("Accuracy of Linear kernel is",1-misClassificationError))

compareTable <- table (testData$Label, predict(svmfit2,testData))
misClassificationError <- mean(testData$Label != predict(svmfit2,testData)) # 4.48% misclassification train error
print(paste("Accuracy of Radial kernel is",1-misClassificationError))

#------------------------ RECURSIVE DECISION TREE ---------------------------#
library(rpart)
Model_rpart= rpart(Label~.,data=trainData, method="class")
Model_rpart
summary(Model_rpart)
#plot(Model_rpart)
#rpart.plot(Model_rpart,type=3,extra=101,fallen.leaves = FALSE)

#Predicting on Train
P1_train_rpart=predict(Model_rpart,trainData,type="class")
table(trainData[,1],predicted=P1_train_rpart)  #Here 1 is column for 'Label'
(3330+420)/(3330+44+107+420) #96.12%

#Predicting on Test
P1_test_rpart=predict(Model_rpart,testData,type="class")
table(testData[,1],predicted=P1_test_rpart) #Here 1 is column for 'Label'
(1425+166)/(1425+26+54+166) #95.21%

# prune the tree 
pfit<- prune(Model_rpart, cp=   Model_rpart$cptable[which.min(Model_rpart$cptable[,"xerror"]),"CP"])

# plot the pruned tree 
plot(pfit, uniform=TRUE, 
     main="Pruned Classification Tree for Label")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)