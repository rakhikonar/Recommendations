# SOCIAL MEDIA ANALYSIS AND NETFLIX SHOW RECOMMENDATION SYSTEM USING NLP AND DEEP LEARNING
* ABSTRACT

Recommendation systems are one of the most popular intelligent systems which enable users to discover new and relevant products as per their interest. Traditional approaches such as collaborative filtering and content- based recommender systems have gained popularity over the years. In this project, we propose a deep neural network-based user-item collaborative filtering recommendation system of Netflix shows for twitter users. This is built using sentiment analysis of tweets about Netflix shows and movies. Twitter data is collected using REST API in python and NLP techniques are performed to extract show names. The proposed approach is build solely using implicit feedback of the user and thereby its purely based on user’s social media interaction. Deep learning MLP model produced promising results with decent accuracy and low errorrates.

* INTRODUCTION

In today’s world recommendation system have become integral part of our day-to-day life. We will be provided with n number of results if we are searching for anything online. This massive amount of information is a little overwhelming for the users. Recommender systems provide relevant suggestions to users based on their history of searches or by finding similarity with the history of users with the item they are looking for. Hence recommender systems are used largely by retail industries, service providers as well as entertainment industries and help their users to make decisions much faster.
Twitter is one of the largest microblogging and social networking service where registered users interact via tweets, retweets, likes etc. Users express their interest on different topics via tweets and hence they are excellent source to understand people’s opinion, likes and dislikes. Even though the number of twitter users are not as big as many other mainstream applications such as YouTube or WhatsApp, we can get a lot of opinions, mentions and hashtags about topics which provides a clear understanding about the user’s stand. Moreover, the textual data can be analysed and processed to interpret the underlying sentiment and hence emotions can be classified into positive, negative, and neutral. We leveraged this aspect and attempted to build a recommendation system unlike the conventional systems which uses user ratings to understand the user-item interactions.
In this project, we focus mainly on building a Netflix show recommendation system based on twitter user interests about different shows which they expressed through tweets. Deep learning and NLP (Natural Language Processing) have had immense impact on wide variety of services recently and we made use of these machine learning techniques to perform sentiment analysis of tweets and provide recommendations based on sentiment score.

* DATASET

Our project idea is to build a recommendation system based on twitter data. There are several twitter users who express their interest about Netflix shows, post reviews through their tweets. We have analyzed such tweets to extract their interest and build a collaborative recommendation system using NLP and Deep learning model which suggest shows of the user’s interests. To achieve this, we have collected tweets using Twitter API with different search keywords related to Netflix show interests. In our project, the dataset is created from social media data (Twitter) using REST api to collect data. We have used different keywords for each genre and collected directly into MongoDB. The data contains four different collections for each genre (Comedy, Romance, Horror and Action) and around 160000 records in total from each genre of 40000 records. The data comprises of twitter data such as userid, username, text, created, retweet_count, source, user language and user location. We also collected rating records from IMDB using web scraping of each shows extracted from tweets. In addition, we have collected ratings and show list of all current Netflix shows for using exact names of the shows as Twitter has unstructured data. Each user in our data has at least one tweet about a show or movie on Netflix. There is a total of 3365 users in 160000 records. Users who have not tweeted at least a show / movie are removed.

* DATA AND MAPPING SHOW NAMES

We extracted twitter data with a goal of collecting tweets which is related to Netflix shows. To get a sensible data from twitter and collect the Netflix show names from tweets, we used different search keywords such as “best comedy Netflix”, “best action Netflix”, “best thriller Netflix” etc.

![](Image/1.PNG)

The data gathered from the twitter looks like below,

![](Image/2.PNG)

As we did not acquire significant number of tweets with this, we collected tweets by giving show names as keywords as shown below,

![](Image/3.PNG)

Our next step was to clean the tweets and extract show names from them after which we can get the sentiment score for each tweet. To extract titles/shows, we used regex to extract words which are within quotes, starts with Capital letter and hashtags as we could see most tweets had show names mentioned in these formats. Once such words were extracted, for show mapping we used string matching and fuzzy methods. We did a full string matching where we could get an exact match between extracted words and actual show titles (Collected from Kaggle dataset). For words which did not pass an exact match test, we did a partial string-matching using Fuzzy string match. Fuzzy logic outputs a value (Lower the score, higher the match) based on how close two strings are. After trying few threshold values, we set a threshold score of 10 with which we got most similar matches and mapped actual show titles with score < 10 and selected only those extracted titles/shows.

Since we have used two methods for show extraction, we did not exclude any potential good data but fuzzy string matching which is a partial matching method, there are chances that the tweets can be unrelated to the matched names for few cases. We included such data so that we will have adequate amount of data to perform rest of the project and most importantly build a deep learning model.

As far as the subjectivity of the tweets are concerned, after processing the data using NLP, filtering it using Regex and cleaning it, we removed bad data, but we could not eliminate all bad data from our dataset. We had to compromise the subjectivity and include such data since we are predicting sentiment of the user item interaction which is mainly based on user’s reaction towards a certain topic.

One example of the tweets with show title has shown below,

![](Image/4.PNG)

The text column is showing actual tweet on twitter posted by user_id 202480252. Since the text data has Netflix keyword and movie word, we have extracted the show name using Regex and fuzzy logic. In this case we know that the user is talking about the movie ‘Hush’. Here, the user is talking about the movie subject only as we extracted the tweets using keywords such as Netflix or movies.

Another example, a tweet:
"The Witcher is fantastic!"  this tweet could be about the show, the novels, or the games.

Here, we can see that there is no clear subjectivity as the data collected by matching a list of show names. But our project simply considered it as a tweet about the show and sentiment analysis was performed, because our main idea was to build a recommendation system based on sentiment score.
Overall, we tried to build a Netflix show recommender system for twitter users who express their interest about shows through tweets. Since there are above mentioned inconsistencies(partial string matching) involved in the data, as NLP is a vast field, we believe this could be improved by collecting tweets which are more specific about shows. Since Twitter contains many user specific unstructured data, we believe that the veracity of this time series data extracted from tweets can be further analyzed to get better accuracy which we considered as future scope.


