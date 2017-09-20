import numpy as np
import pandas as pd
import matrix_factorization_utilities


df = pd.read_csv('movie_ratings_data_set.csv')

movies_df = pd.read_csv('movies.csv', index_col='movie_id')
ratings_df = pd.pivot_table(df, index='user_id', columns='movie_id', aggfunc=np.max)

print ("rating dataset {}".format(df))
print ("movies dataset {}".format(movies_df))
print ("ratings dataset {}".format(ratings_df))

U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.as_matrix(),
                                                                    num_features=15,
                                                                    regularization_amount=10)


print ("U matrix {}".format(U))
print ("M matrix {}".format(M))

M = np.transpose(M)

print ("Transposed M matrix {}".format(M))

movie_id = 5

# name of the movie
movie_information = movies_df.loc[movie_id]

print("We are finding movies similar to this movie:")
print("Movie title: {}".format(movie_information.title))
print("Genre: {}".format(movie_information.genre))

current_movie_features = M[movie_id - 1]

print("The attributes for this movie are:")
print(current_movie_features)

difference = M - current_movie_features

absolute_difference = np.abs(difference)

total_difference = np.sum(absolute_difference, axis=1)

print(total_difference.shape)

movies_df['difference_score'] = total_difference

sorted_movie_list = movies_df.sort_values('difference_score')

print("The five most similar movies are:")
print(sorted_movie_list[['title', 'difference_score']][0:5])
