# MF_NN_Recommender_System
Movie Recommendation System using TensorFlow and Movielens 1M Dataset
This GitHub repository contains a Python implementation of a movie recommendation system using TensorFlow, focusing on collaborative filtering. The model is trained and evaluated on the Movielens 1M dataset, a popular dataset in the field of recommendation systems. Collaborative filtering aims to predict user preferences based on their historical interactions with items and leverages the similarity among users or items.

Contents
Data Loading and Preprocessing:
The code begins by loading the Movielens 1M dataset, which includes information about movie ratings, user IDs, movie IDs, and timestamps. The ratings data is merged with movie information, resulting in a unified dataset. User and movie IDs are then mapped to integer indices to facilitate model training.

Model Architecture:
The recommendation model is implemented using TensorFlow's Keras API. It employs an embedding layer for both users and movies to convert the categorical IDs into dense vector representations. These embeddings are flattened and concatenated to create a joint representation. The concatenated vectors are then passed through several fully connected layers with activation functions like ReLU to capture complex interactions between users and movies. The final layer predicts the user's rating for a given movie.

Model Training:
The model is compiled using the mean squared error (MSE) loss function and the Adam optimizer. It is trained using the training data, consisting of user and movie ID pairs, and corresponding ratings. Training involves several epochs with a specified batch size. A validation split is used to monitor the model's performance during training.

Model Evaluation:
After training, the model is evaluated using the test data. Predicted ratings are obtained for user-movie pairs in the test set. The Mean Squared Error (MSE) is calculated to measure the model's overall performance in predicting ratings.

Binary Classification and Metrics:
The code proceeds to convert the predicted ratings to binary values by applying a threshold (e.g., 3.5) to classify whether a user is likely to like a movie or not. Metrics such as precision, recall, and F1-score are computed to assess the model's ability to correctly classify positive and negative instances. Additionally, nDCG (normalized Discounted Cumulative Gain) score is used to evaluate the ranking quality of recommended items.

Printed Results:
The evaluation metrics, including precision, recall, F1-score, and nDCG, are printed to the console, providing insights into the model's performance in predicting user preferences and generating recommendations.

This GitHub repository offers a comprehensive implementation of a movie recommendation system using collaborative filtering techniques and TensorFlow. It provides a solid foundation for understanding, building, and evaluating recommendation models on the Movielens 1M dataset. Users can further explore and enhance the model's architecture, experiment with hyperparameters, and integrate additional features to improve recommendation quality.

SUMMARY of Dataset
================================================================================

These files contain 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

USAGE LICENSE
================================================================================

Neither the University of Minnesota nor any of the researchers involved can guarantee the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set.  The data set may be used for any research purposes under the following conditions:

     * The user may not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group.
     * The user must acknowledge the use of the data set in publications resulting from the use of the data set
       (see below for citation information).
     * The user may not redistribute the data without separate permission.
     * The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota.
If you have any further questions or comments, please contact GroupLens <grouplens-info@cs.umn.edu>. 

CITATION
================================================================================
To acknowledge use of the dataset in publications, please cite the following paper:
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872


FURTHER INFORMATION ABOUT THE GROUPLENS RESEARCH PROJECT
================================================================================
The GroupLens Research Project is a research group in the Department of Computer Science and Engineering at the University of Minnesota. Members of the GroupLens Research Project are involved in many research projects related to the fields of information filtering, collaborative filtering, and recommender systems. The project is lead by professors John Riedl and Joseph Konstan. The project began to explore automated collaborative filtering in 1992, but is most well known for its world wide trial of an automated  collaborative filtering system for Usenet news in 1996. Since then the project  has expanded its scope to research overall information filtering solutions,  integrating in content-based methods as well as improving current collaborative  filtering technology.
Further information on the GroupLens Research project, including research publications, can be found at the following web site:
        
        http://www.grouplens.org/
GroupLens Research currently operates a movie recommender based on  collaborative filtering:

        http://www.movielens.org/

RATINGS FILE DESCRIPTION
================================================================================

All ratings are contained in the file "ratings.dat" and are in the following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040 
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

USERS FILE DESCRIPTION
================================================================================

User information is in the file "users.dat" and is in the following format:

UserID::Gender::Age::Occupation::Zip-code

All demographic information is provided voluntarily by the users and is not checked for accuracy.  Only users who have provided some demographic information are included in this data set.

- Gender is denoted by a "M" for male and "F" for female
- Age is chosen from the following ranges:

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- Occupation is chosen from the following choices:

	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"

MOVIES FILE DESCRIPTION
================================================================================

Movie information is in the file "movies.dat" and is in the following format:

MovieID::Title::Genres

- Titles are identical to titles provided by the IMDB (including year of release)
- Genres are pipe-separated and are selected from the following genres:

	* Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western

- Some MovieIDs do not correspond to a movie due to accidental duplicate entries and/or test entries
- Movies are mostly entered by hand, so errors and inconsistencies may exist
