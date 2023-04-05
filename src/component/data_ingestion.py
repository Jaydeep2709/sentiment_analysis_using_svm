import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd 
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationConfig

# from src.component.model_trainer import ModelTrainerConfig
# from src.component.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def data_processing(tweet):
        tweet = tweet.lower()
        tweet= re.sub(r"https\S+|www\S+https\S+", '',tweet, flags=re.MULTILINE)
        tweet= re.sub(r'\@w+|\#','',tweet)
        tweet= re.sub(r'[^\w\s]','',tweet)
        tweet_tokens = word_tokenize(tweet)
        filtered_text = [w for w in tweet_tokens if not w in stop_words]
        return " ".join(filtered_text)
    
    
    stemmer = PorterStemmer()
    def stemming(data):
        text = [DataIngestion.stemmer.stem(word) for word in data]
        return data
    
    def polarity(Text):
        return TextBlob(Text).sentiment.polarity
    
    def sentiment(label):
        if label <0:
            return "Negative"
        elif label ==0:
            return "Neutral"
        elif label>0:
            return "Positive"
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\\data\\tweets_sentiment2.csv')
            logging.info('Read the dataset as dataframe')
            df['tweet'] = df['tweet'].apply(DataIngestion.data_processing)
            logging.info("Data processing finished successfully.")
            df['tweet'] = df['tweet'].apply(lambda x: DataIngestion.stemming(x))
            df['polarity'] = df['tweet'].apply(DataIngestion.polarity)
            logging.info("Polarity is assigned to data successfully.")
            df['sentiment'] = df['polarity'].apply(DataIngestion.sentiment)
            logging.info("Sentiment add successfully.")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # modeltrainer=ModelTrainer()
    # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


