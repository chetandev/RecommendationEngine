import numpy as np
import pandas as pd
import matrix_factorization_utilities

# Load user ratings
raw_dataset_df = pd.read_csv('ratings_real.csv')


# training dataset 70%
# test dataset 30 %
raw_dataset_df['split'] = np.random.randn(raw_dataset_df.shape[0], 1)
msk = np.random.rand(len(raw_dataset_df)) <= 0.7
train = raw_dataset_df[msk]
test = raw_dataset_df[~msk]



# Convert the running list of user ratings into a matrix
ratings_training_df = pd.pivot_table(train, index='user_id', columns='movie_id', aggfunc=np.max)
ratings_testing_df = pd.pivot_table(test, index='user_id', columns='movie_id', aggfunc=np.max)

# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_training_df.as_matrix(),
                                                                    num_features=11,
                                                                    regularization_amount=1.1)

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

# Measure RMSE
rmse_training = matrix_factorization_utilities.RMSE(ratings_training_df.as_matrix(),
                                                    predicted_ratings)
rmse_testing = matrix_factorization_utilities.RMSE(ratings_testing_df.as_matrix(),
                                                   predicted_ratings)

print("Training RMSE: {}".format(rmse_training))
print("Testing RMSE: {}".format(rmse_testing))
