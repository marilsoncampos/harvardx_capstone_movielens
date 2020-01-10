## ----load our packages, echo = TRUE, message= FALSE, warning = FALSE-----
repo_url <- "http://cran.us.r-project.org"
if(!require(kableExtra)) 
  install.packages("kableExtra", repos = repo_url)
if(!require(tidyverse))
  install.packages("tidyverse", repos = repo_url)
if(!require(data.table)) 
  install.packages("data.table", repos = repo_url)
if(!require(caret)) 
  install.packages("caret", repos = repo_url)


## ----preparing the movielens dataset, eval = TRUE, warning = FALSE, cache=TRUE----
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
# Removing unused datasets from memory to avoid out of memory errors.
rm(ratings, movies)


## ----prepare train and validation sets, eval = TRUE, warning = FALSE, cache=TRUE----
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
train <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in train set
validation <- temp %>% 
     semi_join(train, by = "movieId") %>%
     semi_join(train, by = "userId")

# Add rows removed from validation set back into train set
removed <- anti_join(temp, validation)
train <- rbind(train, removed)

rm(ratings, movies, test_index, temp, removed)


## ----count the num records in each set, echo = TRUE----------------------
num_recs <- nrow(movielens)
num_train <- nrow(train)
num_validation <- nrow(validation)
numbers <- data.frame(Dataset=c('All data', 'Train', 'Validate'), 
                      Records=c(
                        format(num_recs, big.mark=","), 
                        format(num_train, big.mark=","), 
                        format(num_validation, big.mark=",")),
                      PercentTotal=c(
                        format(round(100*num_recs/num_recs, digits = 2), big.mark=","), 
                        format(round(100*num_train/num_recs, digits = 2), big.mark=","), 
                        format(round(100*num_validation/num_recs, digits = 2), big.mark=","))
                      )
names(numbers) <- c("Dataset", "Number of records", '% Records')
numbers



## ----unique dataset values-----------------------------------------------
unique_values <- movielens %>% 
  summarize(unique_users = format(n_distinct(userId), big.mark=","),
            unique_movies = format(n_distinct(movieId), big.mark=","))
names(unique_values) <- c("Users", "Movies")
uniques_table <- gather(unique_values, key = "Field", value = "# of unique values")
uniques_table


## ----rating counts-------------------------------------------------------
ratings_counts <- train %>% 
  count(movieId) %>% 
  transmute(ct=n)

min_max_ratings <- ratings_counts %>% 
  summarize(
            min_ratings = format(min(ct), big.mark=","),
            max_ratings = format(max(ct), big.mark=","))
names(min_max_ratings) <- c("Smallest number of ratings for a film", "Largest number of ratings for a film")
min_max_table <- gather(min_max_ratings, key = "Stat", value = "Value")
min_max_table


## ----user counts---------------------------------------------------------
user_counts <- train %>% 
  count(userId) %>% 
  transmute(ct=n)

user_activity_stats <- user_counts %>% 
  summarize(
            min_user_act = format(min(ct), big.mark=","),
            max_user_act = format(max(ct), big.mark=","))
names(user_activity_stats) <- c("Minimum number of times a user rates a movie", 
                                "Maximum number of times a user rates a movie")
user_activity_table <- gather(user_activity_stats, key = "Stat", value = "Value")
user_activity_table




## ----helper functions----------------------------------------------------
cost <- function(observed, predicted){
  sqrt(mean((observed - predicted)^2))
}
draw_rmse_table <- function(ds){
  temp <- ds
  names(temp)  <- c("ML Model", "RMSE")
  print(temp)
}


## ----predicted mean rate-------------------------------------------------
simple_model_predicted <- mean(train$rating)


## ----constant residuals--------------------------------------------------
ml_simple_rmse <- cost(validation$rating, simple_model_predicted)
ml_simple_df <- data.frame(model = "Simple Model (Avg Rating)", rmse = round(ml_simple_rmse,digits=5))
draw_rmse_table(ml_simple_df)


## ----movie averages vector-----------------------------------------------
mu <- mean(train$rating)
movie_averages <- train %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu))


## ----model movie bias, warning = FALSE, message = FALSE------------------
movie_bias  <- validation %>% 
    left_join(movie_averages, by='movieId') %>%
    pull(b_m)
movie_predicted <- mu + movie_bias

ml_movie_rmse <- cost(movie_predicted, validation$rating)
ml_movie_df <- data.frame(model = "Movie Model", rmse = round(ml_movie_rmse,digits=5))
draw_rmse_table(ml_movie_df)


## ----user bias plot, warning = FALSE, message = FALSE--------------------
user_averages_plain <- train %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating))



## ----user averages, warning=FALSE, message=FALSE-------------------------
user_averages <- train %>% 
    left_join(movie_averages, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = mean(rating - mu - b_m))


## ----model movie + user, warning = FALSE, message = FALSE----------------
movie_user_predicted <- validation %>% 
    left_join(movie_averages, by='movieId') %>%
    left_join(user_averages, by='userId') %>%
    mutate(y_hat = mu + b_m + b_u) %>%
    pull(y_hat)

ml_movie_user_rmse <- cost(movie_user_predicted, validation$rating)
ml_movie_user_df <- data.frame(model = "Movie & User Model", rmse = round(ml_movie_user_rmse,digits=5))
draw_rmse_table(ml_movie_user_df)


## ----movie+user with regularization, message=FALSE, warning=FALSE, cache = TRUE----
lambda_grid <- seq(4.5, 5.5, 0.01)

rmse_vector <- sapply(lambda_grid, function(l){
  mu <- mean(train$rating)
  b_m <- train %>% 
    group_by(movieId) %>%
    summarize(b_m = sum(rating - mu)/(n()+l))
  
  b_u <- train %>% 
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - mu)/(n()+l))
  
  predicted_ratings <- 
      validation %>% 
      left_join(b_m, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(y_hat = mu + b_m + b_u) %>%
      pull(y_hat)
  
  return(cost(predicted_ratings, validation$rating))
})


## ----df1, message=FALSE, warning=FALSE-----------------------------------
rmse_lambda <- data.frame(rmse = rmse_vector, lambda_grid = lambda_grid)
best_lambda <- lambda_grid[which.min(rmse_vector)]


## ----regularization, message=FALSE, warning=FALSE------------------------
ml_movie_user_reg_df <- data.frame(model = "Movie & User with regularization", 
                                   rmse = round(min(rmse_vector),digits=5))
draw_rmse_table(ml_movie_user_reg_df)


## ----results, message=FALSE, warning=FALSE-------------------------------
results <- bind_rows(
  ml_simple_df,
  ml_movie_df, 
  ml_movie_user_df,
  ml_movie_user_reg_df)
draw_rmse_table(results)

