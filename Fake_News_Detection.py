#!/usr/bin/env python
# coding: utf-8

# # ***Fake News Detection Project***
# ## **Name : T Divyasri**

# ### This project aims to build a fake news detection system using datasets from two different sources: an international news dataset and an Indian news dataset. The goal is to combine these datasets and prepare them for further analysis and model training. Combining and preparing these datasets will provide a comprehensive foundation for further analysis and model training, ultimately contributing to the fight against the spread of false information.
# 
# ---
# 
# 
# 
# 
# 
# 
# 
# **Reference:** <br>
# International News Dataset : [International real and fake News]( https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) <br>
# Indian News Dataset : [Indian fake and real news](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset)
# 
# 
# ---

# # Problem Statement : The goal is to develop an automated system to accurately classify news articles as "Real" or "Fake." In recent years, the widespread dissemination of fake news has led to confusion, eroded public trust in media, and influenced opinions on critical societal issues. Given the vast amount of information available online, manual verification is time-consuming and impractical
# 
# 
# 
# #Approach used to solve: We aim to build a machine learning model that uses natural language processing (NLP) to analyze the text of news articles and predict their authenticity. This solution seeks to aid in combatting misinformation by providing an efficient, reliable method for identifying fake news.

# ## **Step I. Data Loading, Preprocessing, Merging, and Final Dataset Preparation**
# 
# 

# #### We start by importing the pandas library, which we'll use for data manipulation and analysis.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB

import warnings
warnings.filterwarnings('ignore')

# ## <u>**Loading and preprocessing the International News Datasets**</u>

# ### Here, we load two CSV files from the international news dataset:
# 
# #### - 'True.csv': Contains true news articles
# 
# #### - 'Fake.csv': Contains fake news articles
# 
# ### Let's examine the structure of these dataframes:

# In[ ]:


df1 = pd.read_csv('True.csv')
df2 = pd.read_csv('Fake.csv')


# In[ ]:


df2.head()


# ### As we can see, both dataframes contain columns for **'title', 'text', 'subject', and 'date'**. We'll need to preprocess this data for our analysis.
# 

# 
# # Data Preprocessing for International News

# ### **Adding Labels**
# 
# ### We'll add a 'Label' column to distinguish between true and fake news:
# 
# #### - 1 for true news
# 
# #### - 0 for fake news
# 

# In[ ]:


df1['Label'] = 1
df2['Label'] = 0


# ### **Combining Title and Text**
# 
# #### To create a single text field for analysis, we'll combine the 'title' and 'text' columns:

# In[ ]:


df1['Text'] = df1['title'] + " " + df1['text']
df2['Text'] = df2['title'] + " " + df2['text']


# ### **Removing Unnecessary Columns**
# 
# #### We are dropping the columns 'subject', 'date', 'text', and 'title' from both datasets as they do not contribute to predicting whether the news is real or fake. The 'title' column has already been combined with the 'text' column to form a single 'Text' column, which will be used for analysis. This step simplifies the dataset and ensures that only relevant features are considered in the model training process.
# 

# In[ ]:


df1.drop(columns=['subject','date','text','title'],inplace=True)
df2.drop(columns=['subject','date','text','title'],inplace=True)


# In[ ]:


df1.head()


# In[ ]:


df2.head()


# ## **Text Cleaning**

# ### This function performs the following cleaning operations:
# 
# #### 1. Removes non-alphabetic characters (except spaces)
# 
# #### 2. Converts text to lowercase
# 
# #### 3. Removes extra whitespace
# 

# In[ ]:


import re
def clean_text(text):
    # Ensure the input is a string
    text = str(text)
    # Remove non-alphabetic characters (excluding spaces)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


# ### *This function is necessary to standardize and clean the text data, ensuring consistency and improving the accuracy of the model by removing irrelevant characters, normalizing case, and eliminating redundant spaces.*
# 

# In[ ]:


df1['Text'] = df1['Text'].apply(clean_text)
df2['Text'] = df2['Text'].apply(clean_text)


# ## **<u>Loading and Preprocessing the Indian News Dataset</u>**

# In[ ]:


indian_df = pd.read_csv("/data/IFND.csv")


# In[ ]:


indian_df.head()


# ### The Indian news dataset has a different structure, with 'Label' and 'Statement' columns. We'll rename 'Statement' to 'Text' for consistency with our other dataset.
# 

# In[ ]:


indian_df = indian_df.rename(columns={"Statement":"Text"})


# In[ ]:


indian_df.head()


# ### **Mapping Labels**
# 
# ### We'll map the labels to match our international dataset:
# 
# ##### - 'TRUE' to 1
# 
# ##### - 'Fake' to 0
# 

# In[ ]:


label_map = {"TRUE":1,"Fake":0}


# In[ ]:


indian_df["Label"] = indian_df["Label"].replace(label_map)


# In[ ]:


indian_df.head()


# ### **Cleaning Text**
# 
# ### We'll apply the same text cleaning function to the Indian dataset:
# 

# In[ ]:


indian_df["Text"] = indian_df["Text"].apply(clean_text)


# ## **Combining Datasets**
# 
# ### Now that we have preprocessed both the international and Indian news datasets, we can combine them into a single dataset for our fake news detection project:

# In[ ]:


final_df = pd.concat([df1,df2,indian_df], ignore_index=True)


# ### **Saving the Final Dataset**
# 
# ## Finally, we save our combined and preprocessed dataset to a CSV file for future use in our fake news detection model:
# 

# In[ ]:


final_df.to_csv("Final_Prepared_Dataset.csv",index=False)


# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# ---
# 
# 

# # **Step II. Model Building and Evaluation**

# # Importing the necessary Libraries

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ## *The final dataset contains preprocessed text data from both international and Indian news sources, with labels indicating whether each article is true (1) or fake (0)*.
# 

# In[ ]:


df = pd.read_csv("Final_Prepared_Dataset.csv")


# ## **Feature Extraction and Data Splitting**
# ### We separate our features (text content) and target variable (labels):

# In[ ]:


x = df["Text"]
y = df["Label"]


# ## Next, we split our data into training and testing sets. We use 80% of the data for training and 20% for testing:
# 
# ### *Setting random_state=42 ensures reproducibility of our results*
# 

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# ## **Text Vectorization**
# 
# ### To convert our text data into a format suitable for machine learning models, we use TF-IDF (Term Frequency-Inverse Document Frequency) vectorization:
# 

# In[ ]:


vectorization = TfidfVectorizer()


# ### **TF-IDF vectorization helps to:**
# 
# ### 1. *Convert text data into numerical features*
# 
# ### 2. *Capture the importance of words in the context of the entire corpus*
# 
# ### 3. *Reduce the impact of common words that may not be as informative for classification*

# ### We fit the vectorizer on the training data and then transform both training and test data. This ensures that our model isn't exposed to any information from the test set during training.
# 

# In[ ]:


x_train_tfidf = vectorization.fit_transform(x_train)
x_test_tfidf = vectorization.transform(x_test)


# ### Difference between fit_transform and transform : <br>
# #### **fit_transform**: Learns the vocabulary from the training data and transforms the training data into a numerical representation.
# 
# #### **transform** : Transforms the test data into the same numerical representation as the training data, using the vocabulary learned from the training data.

# # **<u>Model Selection and Hyperparameter Tuning</u>**

# ## For our fake news detection task, we'll use Logistic Regression as our baseline model. Logistic Regression is a good choice for binary classification problems like this one.
# 

# # **1. Logisitic Regression**
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()


# ### To optimize our model's performance, we'll use GridSearchCV for hyperparameter tuning:
# 

# ## **Here's what we're tuning**:
# 
# ### **C**: The inverse of regularization strength. Smaller values specify stronger regularization.
# 
# ### **solver**: The algorithm to use in the optimization problem.
# 
# ### We **set max_iter=1000** to allow the model more iterations to converge.
# 
# ### We use 5-fold cross-validation (**cv=5**) to ensure robust results.
# 

# In[ ]:


param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['newton-cg', 'liblinear', 'saga']
}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, refit=True, verbose=2, cv=5)


# ### Now, we fit the grid search to our training data:
# 

# In[ ]:


grid.fit(x_train_tfidf, y_train)


# ### After fitting, we can see the best parameters and score:
# 

# In[ ]:


print("Best parameters found: ", grid.best_params_)
print("Best score: ", grid.best_score_)


# ### With our best model selected, we can now evaluate its performance on the test set:

# In[ ]:


best_model = grid.best_estimator_
y_pred_lr = best_model.predict(x_test_tfidf)
print("Logistic Regression Classification Report after Hyperparameter Tuning:")
print(classification_report(y_test, y_pred_lr))


# In[ ]:


accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy: {accuracy_lr:.2f}")


# ## **Accuracy of Logistic Regression Model : 0.97**

# ## **Confusion Matrix For Logistic Regression**

# In[ ]:


cm = confusion_matrix(y_test,y_pred_lr)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# ### **Accuracy of Logistic Regression Model : 0.97**

# 
# 
# ---
# 
# 
# 
# ---
# 
# 

# # **2. Support Vector Machine (SVM) Model**

# ### After exploring Logistic Regression, we'll now implement a Support Vector Machine (SVM) classifier. SVMs are particularly effective for text classification tasks due to their ability to handle high-dimensional data well.
# 
# ## Optimizing SVM Performance
# 
# ## *To address the computational intensity of SVM, especially with large datasets, we're using Intel's optimized scikit-learn extension*
# 

# In[ ]:





# In[ ]:



from sklearnex import patch_sklearn
patch_sklearn()


# ## We import and initialize our SVM classifier:
# 

# In[ ]:


from sklearn.svm import SVC
svc = SVC()


# ### Then, we fit the model to our training data:

# In[ ]:


svc.fit(x_train_tfidf,y_train)


# ## **Model Evaluation**
# 
# ### After training, we use the model to make predictions on our test set:
# 

# In[ ]:


y_pred_svc = svc.predict(x_test_tfidf)


# In[ ]:


print(classification_report(y_test, y_pred_svc))


# In[ ]:


accuracy = accuracy_score(y_test, y_pred_svc)
print(f"Accuracy: {accuracy:.2f}")


# ## **Accuracy of SVC model : 0.97**

# ## **Confusion Matrix for SVC**

# In[ ]:


cm = confusion_matrix(y_test,y_pred_lr)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# 
# 
# ---
# 
# 
# 
# ---
# 

# # **3. XGBClassifier**

# 
# ### **First, we need to import the necessary library for XGBClassifier**

# In[ ]:


from xgboost import XGBClassifier


# ### **Hyperparameter Tuning for XGBoost Classifier**

# ### **We initialize the XGBoost classifier.**

# In[ ]:


xgb_classifier = XGBClassifier()


# ### **We fit the model with the training data**

# In[ ]:


xgb_classifier.fit(x_train_tfidf,y_train)


# In[ ]:


y_pred_xgb = xgb_classifier.predict(x_test_tfidf)


# In[ ]:


print(classification_report(y_test, y_pred_xgb))


# In[ ]:


accuracy = accuracy_score(y_test, y_pred_xgb)
print(f'Accuracy: {accuracy}')


# ### **Accuracy of XGBClassifier : 0.97**

# 
# 
# ---
# 
# 
# 
# ---

# # **4. Naive Nayes**

# ### **First, we need to import the necessary library for Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# ### **We initialize the Naive bayes clasifier.**

# In[ ]:


nb = MultinomialNB()


# ### **We fit the model with the training data**

# In[ ]:


nb.fit(x_train_tfidf,y_train)


# In[ ]:


y_pred_nb = nb.predict(x_test_tfidf)


# ### **Classification report**

# In[ ]:


print(classification_report(y_test, y_pred_nb))


# In[ ]:


accuracy = accuracy_score(y_test, y_pred_nb)
print(f'Accuracy: {accuracy}')


# ### **Accuracy of Naive Bayes Classifier : 0.93**

# ## **Confusion Matrix For Naive Bayes**

# In[ ]:


cm = confusion_matrix(y_test,y_pred_nb)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 

# 
# ## **<u>Conclusion</u>**
# ### **In this project, we developed a fake news detection system using multiple machine learning models. We utilized logistic regression, support vector classifier (SVC), XGBoost classifier, and Naive Bayes classifier. After extensive evaluation, the performance of each model was as follows:**

# ![export (1).png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABLAAAALcCAYAAAD3x0nUAAAAAXNSR0IArs4c6QAAIABJREFUeF7s3Qe8XNV5L+xXEkIFVSQkQPQiRAcDptgEGQKYUEzvxYAx2Bg75bu5cXK/3MQ3uclNrhPbBNvY9N5NNR0k0w2YZnoRQoAKHYQKIJ3vWyPmMPXMzDlr1M6zf/Ev4syed6/97HXQ5r/XXqvPG9Pf6ggbAQIECBAgQIBAVoGFCxdM7du339pZiypGgAABAgQIEOhlAh0dHdGnT5/o8/qbszqmzZo/9aU3577aywycLgECBAgQIECgLQK7bj1ylw9mf/rBoy/OfrwtB1CUAAECmQT69OnbEdERHR0dfYp/juhTqN7RsfDznxX+KYo/r/5z+rzP5/v36Vi0X3GcxKI/L6q/6LNFdb/Yr/TYxT+X7tO9/dN51W93V/UXfa+83V+0P7ksOpfS/1/4Rp++HZVtra61aJ9a+9c7z9Lr0F2X1NZFx2yu3fWuSSvnWdpPqs+htsPi6SeL+kYz/bteuyv7ck/aXft3pfz3svq6N9O/69fo6pjlfb+8TzfTv0trV++/6N8ttf998sW/hyrPd8Nxg9dec8yAdfqkEVh3Pfbe5B9d8Poumf4dqAwBAgQIECBAoFcLTPrPzePcW2bG+bfO6tUOTp4AAQIECBAg0FOB4/YcE8d/fWwIsHoq6fsECBAgQIAAgQoBAZYuQYAAAQIECBDIIyDAyuOoCgECBAgQIECgSkCApVMQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QMECBAgQIAAgTYJCLDaBKssAQIECBAgQMAILH2AAAECBAgQIJBHQICVx1EVAgQIECBAgECVgABLpyBAgAABAgQI5BEQYOVxVIUAAQIECBAgIMDSBwgQIECAAAECbRIQYLUJVlkCBAgQIECAgBFY+gABAgQIECBAII+AACuPoyoECBAgQIAAgSoBAZZOQYAAAQIECBDIIyDAyuOoCgECBAgQIEBAgKUPECBAgAABAgTaJCDAahOssgQIECBAgAABI7D0AQIECBAgQIBAHgEBVh5HVQgQIECAAAECVQICLJ2CAAECBAgQIJBHQICVx1EVAgQIECBAgIAASx8gQIAAAQIECLRJQIDVJlhlCRAgQIAAAQJGYOkDBAgQIECAAIE8AgKsPI6qECBAgAABAgSqBARYOgUBAgQIECBAII+AACuPoyoECBAgQIAAAQGWPkCAAAECBAgQaJOAAKtNsMoSIECAAAECBIzA0gcIECBAgAABAnkEBFh5HFUhQIAAAQIECFQJCLB0CgIECBAgQIBAHgEBVh5HVQgQIECAAAECAix9gAABAgQIECDQJgEBVptglSVAgAABAgQIGIGlDxAgQIAAAQIE8ggIsPI4qkKAAAECBAgQqBIQYOkUBAgQIECAAIE8AgKsPI6qECBAgAABAgQEWPoAAQIECBAgQKBNAgKsNsEqS4AAAQIECBAwAksfIECAAAECBAjkERBg5XFUhQABAgQIECBQJSDA0ikIECBAgAABAnkEBFh5HFUhQIAAAQIECAiw9AECBAgQIECAQJsEBFhtglWWAAECBAgQIGAElj5AgAABAgQIEMgjIMDK46gKAQIECBAgQKBKQIClUxAgQIAAAQIE8ggIsPI4qkKAAAECBAgQEGDpAwQIECBAgACBNgkIsNoEqywBAgQIECBAwAgsfYAAAQIECBAgkEdAgJXHURUCBAgQIECAQJWAAEunIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIHCBAgQIAAAQJtEhBgtQlWWQIECBAgQICAEVj6AAECBAgQIEAgj4AAK4+jKgQIECBAgACBKgEBlk5BgAABAgQIEMgjIMDK46gKAQIECBAgQECApQ8QIECAAAECBNokIMBqE6yyBAgQIECAAAEjsPQBAgQIECBAgEAeAQFWHkdVCBAgQIAAAQJVAgIsnYIAAQIECBAgkEdAgJXHURUCBAgQIECAgABLHyBAgAABAgQItElAgNUmWGUJECBAgAABAkZg6QMECBAgQIAAgTwCAqw8jqoQIECAAAECBKoEBFg6BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZY+QIAAAQIECBBok4AAq02wyhIgQIAAAQIEjMDSBwgQIECAAAECeQQEWHkcVSFAgAABAgQIVAkIsHQKAgQIECBAgEAeAQFWHkdVCBAgQIAAAQICLH2AAAECBAgQINAmAQFWm2CVJUCAAAECBAgYgaUPECBAgAABAgTyCAiw8jiqQoAAAQIECBCoEhBg6RQECBAgQIAAgTwCAqw8jqoQIECAAAECBARY+gABAgQIECBAoE0CAqw2wSpLgAABAgQIEDACSx8gQIAAAQIECOQR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DXdcAtUCAiy9ggABAgQIECDQJgEBVptglSVAgAABAgR6nYAAq9ddcidMgAABAgQILC4BAdbiknYcAgQIECBAYHkXEGAt71fY+REgQIAAAQJLTECAtcToHZgAAQIECBBYzgQEWMvZBXU6BAgQIECAwNIjIMBaeq6FlhAgQIAAAQLLtoAAa9m+flpPgAABAgQILMUCAqyl+OJoGgECBAgQILBMCQiwlqnLpbEECBAgQIDAsiQgwFqWrpa2EiBAgAABAkuzgABrab462kaAAAECBAgs0wICrGX68mk8AQIECBAgsBQJCLCWoouhKQQIECBAgMDyJSDAWr6up7MhQIAAAQIElpyAAGvJ2TsyAQIECBAgsJwLCLCW8wvs9AgQIECAAIHFJiDAWmzUDkSAAAECBAj0NgEBVm+74s6XAAECBAgQaJeAAKtdsuoSIECAAAECvV5AgNXruwAAAgQIECBAIJOAACsTpDIECBAgQIAAgUoBAZY+QYAAAQIECBDIIyDAyuOoCgECBAgQIECgSkCApVMQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QMECBAgQIAAgTYJCLDaBKssAQIECBAgQMAILH2AAAECBAgQIJBHQICVx1EVAgQIECBAgECVgABLpyBAgAABAgQI5BEQYOVxVIUAAQIECBAgIMDSBwgQIECAAAECbRIQYLUJVlkCBAgQIECAgBFY+gABAgQIECBAII+AACuPoyoECBAgQIAAgSoBAZZOQYAAAQIECBDIIyDAyuOoCgECBAgQIEBAgKUPECBAgAABAgTaJCDAahOssgQIECBAgAABI7D0AQIECBAgQIBAHgEBVh5HVQgQIECAAAECVQICLJ2CAAECBAgQIJBHQICVx1EVAgQIECBAgIAASx8gQIAAAQIECLRJQIDVJlhlCRAgQIAAAQJGYOkDBAgQIECAAIE8AgKsPI6qECBAgAABAgSqBARYOgUBAgQIECBAII+AACuPoyoECBAgQIAAAQGWPkCAAAECBAgQaJOAAKtNsMoSIECAAAECBIzA0gcIECBAgAABAnkEBFh5HFUhQIAAAQIECFQJCLB0CgIECBAgQIBAHgEBVh5HVQgQIECAAAECAix9gAABAgQIECDQJgEBVptglSVAgAABAgQIGIGlDxAgQIAAAQIE8ggIsPI4qkKAAAECBAgQqBIQYOkU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AZY+QIAAAQIECBBok4AAq02wyhIgQIAAAQIEjMDSBwgQIECAAAECeQQEWHkcVSFAgAABAgQIVAkIsHQKAgQIECBAgEAegbIAyyqEeVBVIUCAAAECBAgkAasQ6gcECBAgQIAAgTwCZasQCrDyoKpCgAABAgQIEBBg6QMECBAgQIAAgXwCZQHWgAH9J48aOXyXfOVVIkCAAAECBAj0XgGvEPbea+/MCRAgQIAAgbwC5sDK66kaAQIECBAgQKBTQIClMxAgQIAAAQIE8ggIsPI4qkKAAAECBAgQqBIQYOkUBAgQIECAAIE8AgKsPI6qECBAgAABAgQEWPoAAQIECBAgQKBNAlYhbBOssgQIECBAgAABqxDqAwQIECBAgACBPAJWIczjqAoBAgQIECBAoEpAgKVTECBAgAABAgTyCFiFMI+jKgQIECBAgACBmgHWkJUGxbChK9EhQIAAAQIECBDogYA5sHqA56sECBAgQIAAga4ETOKufxAgQIAAAQIE8ggIsPI4qkKAAAECBAgQqBIQYOkUBAgQIECAAIE8AgKsPI6qECBAgAABAgQEWPoAAQIECBAgQKBNAgKsNsEqS4AAAQIECBAwAksfIECAAAECBAjkESgLsAYNHDBp5IihE/OUVoUAAQIECBAg0LsFrELYu6+/sydAgAABAgTyCZStQijAygerEgECBAgQIEBAgKUPECBAgAABAgTyCJQFWAMG9J88auTwXfKUVoUAAQIECBAg0LsFvELYu6+/sydAgAABAgTyCZgDK5+lSgQIECBAgACBMgEBlg5BgAABAgQIEMgjIMDK46gKAQIECBAgQKBKQIClUxAgQIAAAQIE8ggIsPI4qkKAAAECBAgQEGDpAwQIECBAgACBNgkIsNoEqywBAgQIECBAwAgsfYAAAQIECBAgkEegLMCyCmEeVFUIECBAgAABAknAKoT6AQECBAgQIEAgj4BVCPM4qkKAAAECBAgQqBIwAkunIECAAAECBAjkEfAKYR5HVQgQIECAAAECAix9gAABAgQIECDQJgEBVptglSVAgAABAgQIGIGlDxAgQIAAAQIE8ggIsPI4qkKAAAECBAgQqBIQYOkUBAgQIECAAIE8AgKsPI6qECBAgAABAgQEWPoAAQIECBAgQKBNAlYhbBOssgQIECBAgAABqxDqAwQIECBAgACBPAJWIczjqAoBAgQIECBAoErAK4Q6BQECBAgQIEAgj4BXCPM4qkKAAAECBAgQEGDpAwQIECBAgACBNgkIsNoEqywBAgQIECBAwAgsfYAAAQIECBAgkEdAgJXHURUCBAgQIECAQJWAAEunIECAAAECBAjkETCJex5HVQgQIECAAAECNQOsoUMGR/qfjQABAgQIECBAoPsCJnHvvp1vEiBAgAABAgS6FDACSwchQIAAAQIECOQR8AphHkdVCBAgQIAAAQJVAgIsnYIAAQIECBAgkEdAgJXHURUCBAgQIECAgABLHyBAgAABAgQItElAgNUmWGUJECBAgAABAkZg6QMECBAgQIAAgTwCAqw8jqoQIECAAAECBKoEBFg6BQECBAgQIEAgj4BVCPM4qkKAAAECBAgQqBlgWYVQxyBAgAABAgQI9FygbBXCQQMHTBo5YujEnpdVgQABAgQIECBAII3AEmDpBwQIECBAgACBnguUBVgDBvSfPGrk8F16XlYFAgQIECBAgAABrxDqAwQIECBAgACBPALmwMrjqAoBAgQIECBAoEpAgKVTECBAgAABAgTyCAiw8jiqQoAAAQIECBAQYOkDBAgQIECAAIE2CQiw2gSrLAECBAgQIEDACCx9gAABAgQIECCQR0CAlcdRFQIECBAgQIBAlYAAS6cgQIAAAQIECOQRKAuwrEKYB1UVAgQIECBAgEASsAqhfkCAAAECBAgQyCNgFcI8jqoQIECAAAECBKoEjMDSKQgQIECAAAECeQS8QpjHURUCBAgQIECAgABLHyBAgAABAgQItElAgNUmWGUJECBAgAABAkZg6QMECBAgQIAAgTwCAqw8jqoQIECAAAECBKoEBFg6BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZY+QIAAAQIECBBok4BVCNsEqywBAgQIECBAwCqE+gABAgQIECBAII9A2SqEgwYOmDRyxNCJeUqrQoAAAQIECBDo3QICrN59/Z09AQIECBAgkE+gLMAaMKD/5FEjh++Sr7xKBAgQIECAAIHeK2AOrN577Z05AQIECBAgkFfAHFh5PVUjQIAAAQIECHQKCLB0BgIECBAgQIBAHgEBVh5HVQgQIECAAAECVQICLJ2CAAECBAgQIJBHQICVx1EVAgQioqMj4s0ZM+Oe+x6O5154Jd7/4MP47LMFBZs+ffrEkCGDY7111owdt/9S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gAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII+AACuPoyoECBAgQIAAgSYBAZamIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIDBAgQIECAAIGSBARYJcEqS4AAAQIECBAwAksPECBAgAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII+AACuPoyoECBAgQIAAgSYBAZamIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIDBAgQIECAAIGSBARYJcEqS4AAAQIECBAwAksPECBAgAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII+AACuPoyoECBAgQIAAgSYBAZamIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIDBAgQIECAAIGSBARYJcEqS4AAAQIECBAwAksPECBAgAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII+AACuPoyoECBAgQIAAgSYBAZamIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIDBAgQIECAAIGSBARYJcEqS4AAAQIECBAwAksPECBAgAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII+AACuPoyoECBAgQIAAgSYBAZamIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIDBAgQIECAAIGSBARYJcEqS4AAAQIECBAwAksPECBAgAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII+AACuPoyoECBAgQIAAgSYBAZamIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIDBAgQIECAAIGSBARYJcEqS4AAAQIECBAwAksPECBAgAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII+AACuPoyoECBAgQIAAgSYBAZamIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIDBAgQIECAAIGSBARYJcEqS4AAAQIECBAwAksPECBAgAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII+AACuPoyoECBAgQIAAgSYBAZamIECAAAECBAjkERBg5XFUhQABAgQIECAgwNIDBAgQIECAAIGSBARYJcEqS4AAAQIECBAwAksPECBAgAABAgTyCAiw8jiqQoAAAQIECBBoEhBgaQoCBAgQIECAQB4BAVYeR1UIECBAgAABAgIsPUCAAAECBAgQKElAgFUSrLIECBAgQIAAASOw9AABAgQIECBAII9AXYCVp6QqBAgQIECAAAECFYHBgwelr79eAYQAAQIECBAgQCCDwKAlS994efDgweOHDxv6RoZ6ShAgQKBMgfg/wUHffEHl15X/O4w/b+fX8fFWdRr/vN392qnV3T65j7uda9DdMfXl/GvPpzd1ujr+lXXcjdekse9qj3FVHVMcQ3d9v6qOu/a6t9PfvTXs7efa6cuujruVa92xfPrZ5xNWrEjfGjF86Gvt/IW0DwECBFaRwIpBgwanFSu+Lv6bMmjQ4BXx6/iz4l+Ubf76Xx+u1uns18V/s9rZ75vj6NP+bZxD2/Vr7h27eipR8Wu7bk/Os+Z8elW/4Z6htt06O+5qPzRc3x6dZ+1xx/nW9lVD31XuaXp1fj3pq8bj6KbXi/NtcdyVPl9px13796ed/q78fW7372ft3/9vrk2rv8vtXp86n8r16eq4W/VD5XOfff7F+OI8li57e0VHx1p3jRs7Zuoq+pemryVAgAABAgQI9CsBrxD2q8vpZAgQIECAAIFVKGAOrFWI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980ECBAgQIBAPxcQYPXzC+z0CBAgQIAAgZUmIMBaadS+iAABAgQIEBhoAgKsgXbFnS8BAgQIECBQloAAqyxZdQkQIECAAIEBLyDAGvAtAIAAAQIECBDIJCDAygSpDAECBAgQIECgUUCApScIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkEBFh5HFUhQIAAAQIECDQJCLA0BQECBAgQIEAgj4AAK4+jKgQIECBAgAABAZYeIECAAAECBAiUJCDAKglWWQIECBAgQICAEVh6gAABAgQIECCQR0CAlcdRFQIECBAgQIBAk4AAS1MQIECAAAECBPIICLDyOKpCgAABAgQIEBBg6QECBAgQIECAQEkCAqySYJUlQIAAAQIECBiBpQcIECBAgAABAnkE6gKsPCVVIUCAAAECBAgQqAgMHjwoff31CiAECBAgQIAAAQIZBAYtWfrGy0OGfGvDYUPXfjNDPSUIECBQpkD8n+Cgb76g8uvK/x3Gn7fz6/h4qzqNf97ufu3U6m6f3MfdzjXo7pj6cv6159ObOl0d/8o67sZr0th3tce4qo4pjqG7vl9Vx1173dvp794a9vZz7fRlV8fdyrXuWD797PMJg9KgIcOGdSxt5y+kfQgQILCKBFYMGjQ4rVjxdfHflEGDBq+IX8ef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CBAoSUCAVRKssgQIECBAgAABI7D0AAECBAgQIEAgj4AAK4+jKgQIECBAgACBJgEBlqYgQIAAAQIECOQREGDlcVSFAAECBAgQICDA0gMECBAgQIAAgZIEBFglwSpLgAABAgQIEDACSw8QIECAAAECBPIICLDyOKpCgAABAgQIEGgSEGBpCgIECBAgQIBAHgEBVh5HVQgQIECAAAECAiw9QIAAAQIECBAoSUCAVRKssgQIECBAgAABI7D0AAECBAgQIEAgj4AAK4+jKgQIECBAgACBJgEBlqYgQIAAAQIECOQREGDlcVSFAAECBAgQICDA0gMECBAgQIAAgZIEBFglwSpLgAABAgQIEDACSw8QIECAAAECBPIICLDyOKpCgAABAgQIEGgSEGBpCgIECBAgQIBAHgEBVh5HVQgQIECAAAECAiw9QIAAAQIECBAoSUCAVRKssgQIECBAgAABI7D0AAECBAgQIEAgj4AAK4+jKgQIECBAgACBJgEBlqYgQIAAAQIECOQREGDlcVSFAAECBAgQICDA0gMECBAgQIAAgZIEBFglwSpLgAABAgQIEDACSw8QIECAAAECBPIICLDyOKpCgAABAgQIEGgSEGBpCgIECBAgQIBAHgEBVh5HVQgQIECAAAECAiw9QIAAAQIECBAoSUCAVRKssgQIECBAgAABI7D0AAECBAgQIEAgj4AAK4+jKgQIECBAgACBJgEBlqYgQIAAAQIECOQREGDlcVSFAAECBAgQICDA0gMECBAgQIAAgZIEBFglwSpLgAABAgQIEDACSw8QIECAAAECBPIICLDyOKpCgAABAgQIEGgSEGBpCgIECBAgQIBAHgEBVh5HVQgQIECAAAECAiw9QIAAAQIECBAoSUCAVRKssgQIECBAgAABI7D0AAECBAgQIEAgj4AAK4+jKgQIECBAgACBJgEBlqYgQIAAAQIECOQREGDlcVSFAAECBAgQICDA0gMECBAgQIAAgZIEBFglwSpLgAABAgQIEDACSw8QIECAAAECBPIICLDyOKpCgAABAgQIEGgSEGBpCgIECBAgQIBAHgEBVh5HVQgQIECAAAECAiw9QIAAAQIECBAoSUCAVRKssgQIECBAgAABI7D0AAECBAgQIEAgj4AAK4+jKgQIECBAgACBJgEBlqYgQIAAAQIECOQREGDlcVSFAAECBAgQICDA0gMECBAgQIAAgZIEBFglwSpLgAABAgQIEDACSw8QIECAAAECBPIICLDyOKpCgAABAgQIEGgSEGBpCgIECBAgQIBAHgEBVh5HVQgQIECAAAECAiw9QIAAAQIECBAoSUCAVRKssgQIECBAgAABI7D0AAECBAgQIEAgj4AAK4+jKgQIECBAgACBJgEBlqYgQIAAAQIECOQREGDlcVSFAAECBAgQICDA0gMECBAgQIAAgZIEBFglwSpLgAABAgQIEDACSw8QIECAAAECBPIIVAKs/x/OKppzDJLTdAAAAABJRU5ErkJggg==)

# ## **The logistic regression, SVC, and XGBoost classifier all achieved a high accuracy of 97%, indicating their effectiveness in detecting fake news. The Naive Bayes classifier, while slightly less accurate, still performed admirably with a 93% accuracy.**

# #### ***Overall, the results demonstrate that machine learning models, particularly logistic regression, SVC, and XGBoost, are highly capable of distinguishing between real and fake news. This project showcases the potential of these models to aid in combating the spread of misinformation, providing a reliable tool for fake news detection.***
