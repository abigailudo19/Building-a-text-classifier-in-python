# Building-a-text-classifier-in-python
#Building a text classifier in python
#1 Include the following lines in a new Python file to add datasets
from sklearn.datasets import fetch_20newsgroups 
category_mapping = {'misc.forsale': 'Sellings', 'rec.motorcycles': 'Motorbikes', 
        'rec.sport.baseball': 'Baseball', 'sci.crypt': 'Cryptography', 
        'sci.space': 'OuterSpace'} 
 
training_content = fetch_20newsgroups(subset='train', 
categories=category_mapping.keys(), shuffle=True, random_state=7) 

#2 Perform feature extraction to extract the main words from the text:
from sklearn.feature_extraction.text import CountVectorizer 
 
vectorizing = CountVectorizer() 
train_counts = vectorizing.fit_transform(training_content.data) 
print("nDimensions of training data:", train_counts.shape )

#3Train the classifier:
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfTransformer 
 
input_content = [ 
    "The curveballs of right handed pitchers tend to curve to the left", 
    "Caesar cipher is an ancient form of encryption", 
    "This two-wheeler is really good on slippery roads" 
] 
 
tfidf_transformer = TfidfTransformer() 
train_tfidf = tfidf_transformer.fit_transform(train_counts) 

#4Implement the Multinomial Naive Bayes classifier:
classifier = MultinomialNB().fit(train_tfidf, training_content.target) 
input_counts = vectorizing.transform(input_content) 
input_tfidf = tfidf_transformer.transform(input_counts) 

#5Predict the output categories:
categories_prediction = classifier.predict(input_tfidf) 

#6Print the output:
for sentence, category in zip(input_content, categories_prediction): 
    print ('nInput:', sentence, 'nPredicted category:',  
            category_mapping[training_content.target_names[category]] )
            
########
#The below would let you know were to save the folders on your computer if you dont want to  get it from sk learn
#import os
#cwd = os.getcwd()
#cwd
#The below would quickly tell you if its saved 
#os.path.exists('xxxxxxxxxx')
