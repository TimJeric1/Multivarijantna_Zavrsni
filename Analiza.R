install.packages("tidyverse")
install.packages("corrplot")
install.packages("tidyr")
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
install.packages("caret")
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
install.packages("MASS")
library(MASS)

# Razdvajanje skupa na trening i testni
install.packages("caret")
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
pca_result <- prcomp(data[-8], center=TRUE, scale=TRUE)

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
install.packages('devtools')
library(devtools)
install_github('vqv/ggbiplot')
install.packages("scales")
library(ggbiplot)


ggbiplot(pca_result,
         obs.scale = 1,
         var.scale = 1,
         groups = data$Quality)



# 4. Stepwise odabir modela prema naprijed

install.packages("MASS")
library(MASS) 
install.packages("caret")
library(caret)

# Razdvajanje skupa na trening i testni
indexes = createDataPartition(scaled_data$Quality, p = 0.8, list = F)
data_train = scaled_data[indexes, ]
data_test = scaled_data[-indexes, ]


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
install.packages("glmnet")
library(glmnet)


# Razdvajanje stupaca na zavisne i nezavisne
x <- as.matrix(data[, -8])
y <- as.factor(data$Quality)

# Ridge regresija
ridge_model <- cv.glmnet(x, y, alpha = 0, family = "binomial")

# Pregled modela
plot(ridge_model)

# Extract coefficients for a specific lambda (you can choose based on the plot)
lambda_chosen <- ridge_model$lambda.min
coef_chosen <- coef(ridge_model, s = lambda_chosen)
print(coef_chosen)



