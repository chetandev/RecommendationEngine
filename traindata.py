# this module should be run periodically

import numpy as np
import pandas as pd
import pickle
import matrix_factorization_utilities

# Load user ratings
raw_dataset_df = pd.read_csv('ratings_real.csv')

# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(raw_dataset_df, index='userId', columns='movieId', aggfunc=np.max)

# Apply matrix factorization to find the latent features
# longest process is to factor the matrix this should be offloaded from main thread
# factor only once and save the results in file
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.as_matrix(),
                                                                    num_features=15,
                                                                    regularization_amount=0.1)

# Find all predicted ratings by multiplying U and M

predicted_ratings = np.matmul(U, M)

# Save features and predicted ratings to files for later use
pickle.dump(U, open("users_real.dat", "wb"))
pickle.dump(M, open("movies_real.dat", "wb"))
pickle.dump(predicted_ratings, open("predicted_ratings_real.dat", "wb" ))

print("finished")