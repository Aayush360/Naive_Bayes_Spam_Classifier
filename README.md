# Naive_Bayes_Spam_Classifier
Model to classify spam and ham email using Naive Bayes Algorithm.

1.Problem Formulation:

Process the raw email to create a model that classifies if the email is spam or not-spam(ham)
using Naive Bayes Classifier.


2.Gathering Data:

https://spamassassin.apache.org/old/publiccorpus/

opensource anti-spam platform


3. Machine Learning Model:

Uses Naive Bayes Classifier.
Calculates the joint probability of the words in an email or document.
Makes the decision based on the conditional probability and joint probability.
If the condition:
P(Spam|word) > P(Ham|word)
is true, it predicts that the email is spam. otherwise, Non-spam.

Uses Bag of Word approach, that is, every words in the email is treated independently.
Each word is look in isolation.
Does not Give importance to sequence, thus the context is lost - hence the name -Naive.


Frequency of each word will act as feature in our model.

4. Data Cleaning:

a. extracting the body from the entire email.
b. convert the text file to Dataframe.
c. Checking for empty or blank email.
d. Remove rows with bad data

finally, save the dataframe to .json format for further exploration.


5. Data Exploration and Visualization:

i. preprocessing of the data -- (Natural Language Processing):
    a. convert to lower case.
    b. Tokenizing
    c. Removing the stop-words.
    d. Word-stemming
    e. Removing Punctuations

   for tokenizing, removing stop-words and word-stemming, we have used nltk(Natural Language Toolkit) module .

   For cleaning the HTML tags, we have used BeautufulSoup package.

ii. For Visualizing the amount of spam and ham words, we have used pie-chart and Doughnut-chart.
    and this is carried out using Matplotlib and Seaborn.

    Also, we have made WordClouds for visualizing the frequency of words/tokens in the spam and non-spam list.
    Words with higher frequency appears to be larger.
    TO accomplish this , we have used pillow module for image manipulation.


As a feature for the classifier to train on, we have used Sparse matrix which is later converted into FullMatrix.
Fullmatrix is the matrix that has columns for each document_id, category and most common(2500) words from the entire
dataset.
The frequency for each words is inserted into the word column, if the words appears in the document. Else, 0 is inserted.


Finally, seperate the category column from the FullMatrix to form label for the classifier, and the 2500 columns
forms the features.


6. Model Training

Initially, we split the data into 30% test size using sklearn's train-test-split functionality.


Second, we calculate the token Probability for each word.

P(Spam|Token) = P(Token|Spam)*P(Spam) / P(Token)
P(Ham|Token) = P(Token|Ham) * P(Ham) / P(Token)


Finally we calculate the joint Probability.

P(Spam|hello) * P(Spam| how) * P(Spam|are) * P(Spam|you)
                   vs.
P(Ham|hello) * P(Ham| how) * P(Ham|are) * P(Ham|you)


If the above probability is higher, email is classified spam. Else, email is Non-spam.

7. Model Prediction, Evaluation and Deployment.

For Prediction, the joint Probability for email being spam is compared with the Joint Probability for email being Ham.
If former is greater, it outputs True or 1 else, it outputs False or 0.

The same is carried for the test data.

The result is evaluated on test data which gave the following Output:

Accuracy Score : 97.10 %
Recall Score : 93.03%
Precision Score : 98.38%
F1 - score : 95.63%


Lastly, we deployed our model using Flask -API, wt-form- Flexible forms validation and rendering library for Python web development.






Resources:

https://wtforms.readthedocs.io/en/2.3.x/
https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
https://flask.palletsprojects.com/en/1.1.x/
https://stackoverflow.com/
https://spamassassin.apache.org/old/publiccorpus/
