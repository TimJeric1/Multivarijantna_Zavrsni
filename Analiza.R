install.packages("tidyverse")
install.packages("corrplot")
install.packages("tidyr")
install.packages("caret")
install.packages("MASS")
install.packages("devtools")
install.packages("scales")
install.packages("glmnet")
library(tidyverse)
library(corrplot)
library(tidyr)


# Čitanje dataseta iz csv
current_directory <- getwd()
file_path <- file.path(current_directory, "apple_quality.csv")
data <- read.csv(file_path)

# 1. Istraživačka analiza

# Struktura podataka
str(data)

# Pretvoren stupac "Acidity" u numeričkog tipa
data$Acidity = as.numeric(data$Acidity)

# Izbačen stupac ID
data = data[,-1]

head(data)
summary(data)

# Provjera za nedostajućim vrijednostima
sapply(data, function(x) sum(is.na(x)))

# Izbacivanje svih redova sa nedostajućim vrijednostima (samo 1)
data <- na.omit(data)

# Distribucija ciljne varijable Quality
ggplot(data, aes(x = Quality)) +
  geom_bar() +
  labs(title = "Distribution of Quality")

# Distribucije numeričkih varijabli
data %>%
  select_if(is.numeric) %>%
  gather(key = "feature", value = "value") %>%
  ggplot(aes(x = value)) +
  geom_histogram(binwidth = 1) +
  facet_wrap(~feature, scales = "free") +
  labs(title = "Distribution of Numerical Features")

# Korelacija numeričkih varijabli
cor_matrix <- cor(data[, -8])
corrplot(cor_matrix,)

# 2. Logistička regresija

# Pretvaranje quality character varijable u numericku sa 0 i 1 kao vrijednostima
data$Quality <- as.numeric(data$Quality == "good")

# Razdvajanje skupa na trening i testni
library(caret)

indexes = createDataPartition(data$Quality, p = 0.8, list = F)
data_train = data[indexes, ]
data_test = data[-indexes, ]

# logistička regresija
log_model <- glm(Quality ~ ., data = data_train, family = "binomial")

# predikcije nad modelom logističke regresije
predictions <- predict(log_model, newdata = data_test, type = "response")
predictions <- ifelse(predictions > 0.5, 1, 0)
confusionMatrix(data=as.factor(predictions), reference=as.factor(data_test$Quality))


# 3.a Linearna Diskriminantna analiza (Linearna Diskriminantna analiza nije dobra za ovaj dataset
# Jer ciljna varijabla Quality ima samo dvije kategorije sto znaci da ce LDA outputat samo 1 dimenziju)
library(MASS)

# Razdvajanje skupa na trening i testni
library(caret)
indexes = createDataPartition(data$Quality, p = 0.8, list = F)
data_train = data[indexes, ]
data_test = data[-indexes, ]

# Linearna Diskriminantna Analiza
lda_model <- lda(Quality ~ ., data = data_train)  # Exclude 'id' column

# Predviđanje
predictions <- predict(lda_model, data_test)
confusionMatrix(data=as.factor(predictions$class), reference=as.factor(data_test$Quality))

# 3.b Analiza glavnih komponenti (Napravljena je i Analiza glavnih komponenti jer je LDA loš s ovim datasetom)

# Analiza glavnih komponenti
pca_result <- prcomp(data[-8], center=FALSE, scale=FALSE) # Skup podataka dolazi već skaliran i centriran

# Pregled modela
summary(pca_result)

# Vizualizacija

# Proporcija varijance koju svaka komponenta objašnjava
prop_var <- pca_result$sdev^2 / sum(pca_result$sdev^2)

# Graf kumulativne proporcije varijance koje komponente objašnjavaju
plot(cumsum(prop_var), xlab = "Number of Principal Components", ylab = "Cumulative Proportion of Variance Explained", type = "b")

# Graf za vizualizaciju varijance koju svaka komponenta objašnjava
plot(prop_var, xlab = "Principal Component", ylab = "Proportion of Variance Explained", type = "s")

#Izrada ggbiplota, ali je loš jer prve dvije komponente objašnjavaju samo 42.5% varijance
library(devtools)
install_github('vqv/ggbiplot')
library(ggbiplot)

       
ggbiplot(pca_result,
         obs.scale = 1,
         var.scale = 1,
         groups = as.factor(original_data$Quality))



# 4. Stepwise odabir modela prema naprijed

library(MASS) 
library(caret)

# Razdvajanje skupa na trening i testni
indexes = createDataPartition(data$Quality, p = 0.8, list = F)
data_train = data[indexes, ]
data_test = data[-indexes, ]


# model logističke regresije
full_model <- glm(Quality ~ ., data = data_train, family = "binomial")

# Stepwise odabir modela prema naprijed
stepwise_model <- stepAIC(full_model, direction = "forward", trace = FALSE)

# Pregled modela
summary(stepwise_model)

# Koeficijenti modela
coefficients(stepwise_model)

# Testiranje modela
predictions <- predict(stepwise_model, newdata = data_test, type = "response")
predictions <- ifelse(predictions > 0.5, 1, 0)
confusionMatrix(data=as.factor(predictions), reference=as.factor(data_test$Quality))



# 5. Ridge regresija
library(glmnet)
library(caret)

# Razdvajanje skupa na trening i testni
indexes = createDataPartition(data$Quality, p = 0.8, list = FALSE)
data_train = data[indexes, ]
data_test = data[-indexes, ]

# Pretvaranje podataka u matrice
x_train <- as.matrix(data_train[, -ncol(data_train)])
x_test <- as.matrix(data_test[, -ncol(data_test)])

# Odvajanje ciljne varijable
y_train <- data_train$Quality
y_test <- data_test$Quality

# Izgradnja modela Ridge regresije korištenjem unakrsne validacije
ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)  # alpha = 0 za Ridge regresiju
optimalni_lambda <- ridge_model$lambda.min

# Predviđanja na testnom skupu
predictions <- predict(ridge_model, newx = x_test, s = optimalni_lambda, type = "response")
predictions <- as.numeric(predictions > 0.5)

# Izračunavanje matrice konfuzije
confusionMatrix <- table(Stvarno = y_test, Predviđeno = predictions)

# Izračunavanje točnosti modela
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)

# Ispis rezultata
print(confusionMatrix)
print(paste("Accuracy: ", accuracy))




