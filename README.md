## HATE SPEECH DETECTION USING MACHINE LEARNING

The goal of this project is to create a machine learning model that can identify this kind of offensive or abusive content in tweets using machine learning techniques. The provided Python code preprocesses the text data, trains a decision tree classifier, and assesses the classifier's performance after reading in a dataset of labelled tweets.

**REQUIREMENTS:**

1. python 3.x
2. pandas
3. scikit-learn
4. nltk
5. streamlit
   
## TECHNOLOGIES USED 

Libraries:

- pandas:Used for data manipulation and analysis, handling dataset exploration and mapping labels.

- nltk:Utilized for language preprocessing (NLP) tasks like tokenization ,stemming and stopwords removal.

- scikit-learn:This library includes tools for machine learning, including decision trees and vectorization

- streamlit- for app demo

Techniques:

1. Machine Learning: Utilized for text classification of tweets into categories (hate speech, offensive language, or none).
2. Text Preprocessing: Cleans and transforms text data by removing punctuation, converting to lowercase, eliminating URLs, stemming, and removing stopwords.
3. Text Feature Extraction: Converts text data into numerical features using CountVectorizer, creating a matrix representation where rows represent documents and columns represent words, with cell values indicating word frequencies.
4. Decision Tree Classifier: Employed for classification tasks, the model learns rules from training data and applies them to classify new data points.


**INSTALLATION**

Make sure you have the required Libraries intalled. You can install them using pip(for windows).Type the following commands in terminal for installation.

-> pip install pandas

-> pip install scikit-learn

-> pip install nltk

-> pip install streamlit

## Code Walk-through

*Step 1*: Data Loading and Exploration

- The code loads the labeled dataset labeled_data.csv using pandas.
- It checks for missing values in each column.
- Extracts the 'tweet' column and maps the 'class' column to human-readable labels.

*Step 2*: Text Preprocessing

- Defines a function clean() to preprocess the text data by converting to lowercase, removing URLs, punctuation, digits, stopwords, and applying stemming.

*Step 3*: Feature Extraction

- Utilizes CountVectorizer from scikit-learn to convert the preprocessed text data into a matrix of token counts.

*Step 4*: Model Training and Evaluation

- Splits the dataset into training and testing sets.
- Trains a Decision Tree classifier on the training data.
- Evaluates the model's accuracy on the testing data.
- Provides sample predictions using the trained model.

*Step 5*: Performance Evaluation

- Calculates precision scores for each class.
- Generates a classification report including precision, recall, F1-score, and support for each class.

## USAGE

1. Clone this repository.
2. Download the labeled data file named "labeled_data.csv" and place it in the same directory as the Python script
3. To run the streamlit app run the command in the terminal :
   streamlit run app.py



