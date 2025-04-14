import streamlit as st
import pandas as pd
import requests
import json 
if __name__ == "__main__":
    import sys
    sys.path.append('code')
    from apicalls import get_google_place_details, get_azure_sentiment, get_azure_named_entity_recognition
else:
    from code.apicalls import get_google_place_details, get_azure_sentiment, get_azure_named_entity_recognition

PLACE_IDS_SOURCE_FILE = "code/solutions/cache/place_ids.csv"
CACHE_REVIEWS_FILE = "code/solutions/cache/reviews.csv"
CACHE_SENTIMENT_FILE = "code/solutions/cache/reviews_sentiment_by_sentence.csv"
CACHE_ENTITIES_FILE = "code/solutions/cache/reviews_sentiment_by_sentence_with_entities.csv"


def reviews_step(place_ids: str|pd.DataFrame) -> pd.DataFrame:
    '''
      1. place_ids --> reviews_step --> reviews: place_id, name (of place), author_name, rating, text 
    '''

    #Decides if the input is a string for the file name, or the df itself to create the ids df
    if isinstance(place_ids, str):
      places_ids_df = pd.read_csv(place_ids)
    else:
      places_ids_df = place_ids

    google_ids = []

    #Iterates over all the rows in the dataframe created above
    for index, row in places_ids_df.iterrows():
        
        #For each row, calling the get_google_place_details function made in previous file
        place = get_google_place_details(row['Google Place ID'])

        #Appending the results of the API call to a list
        google_ids.append(place['result'])

    #Creates a new DF with the google review ids, location name, and review
    reviews_df = pd.json_normalize(google_ids, record_path= "reviews", meta= ["place_id", "name"])

    #Filtering DF to just the columns we want
    review_df = reviews_df[['place_id', 'name', 'author_name', 'rating', 'text']]

    #Caching review df created in this function and returning it
    review_df.to_csv(CACHE_REVIEWS_FILE, index = False)
    return review_df


def sentiment_step(reviews: str|pd.DataFrame) -> pd.DataFrame:
    '''
      2. reviews --> sentiment_step --> review_sentiment_by_sentence
    '''

    #Decides if the input is a string for the file name, or the df itself to create the reviews df
    if isinstance(reviews, str):
      reviews_df = pd.read_csv(reviews)
    else:
      reviews_df = reviews

    sentiments = []
    for index, row in reviews_df.iterrows():
        sentiment = get_azure_sentiment(row['text'])
        sentiment_item = sentiment['results']['documents'][0]
        sentiment_item['place_id'] = row['place_id']
        sentiment_item['name'] = row['name']
        sentiment_item['author_name'] = row['author_name']
        sentiment_item['rating'] = row['rating']
        sentiments.append(sentiment_item)

    #Creates a DF of the sentiments list which is a list of dictionaries
    sentiment_df = pd.json_normalize(sentiments, record_path="sentences", meta=["place_id", 'name', 'author_name', 'rating'])
    
    #Renames the sentiment column
    sentiment_df.rename(columns={'text': 'sentence_text'}, inplace=True)
    sentiment_df.rename(columns={'sentiment': 'sentence_sentiment'}, inplace=True)

    #Filtering down to the columns we want 
    sentiment_df = sentiment_df[['place_id', 'name', 'author_name', 'rating', 'sentence_text', 'sentence_sentiment', 'confidenceScores.positive', 'confidenceScores.neutral', 'confidenceScores.negative']]

    #Outputting file to the cache and returning it
    sentiment_df.to_csv(CACHE_SENTIMENT_FILE, index=False)
    return sentiment_df




def entity_extraction_step(sentiment: str|pd.DataFrame) -> pd.DataFrame:
    '''
      3. review_sentiment_by_sentence --> entity_extraction_step --> review_sentiment_entities_by_sentence
    '''
    #Decides if the input is a string for the file name, or the df itself to create the sentiment df
    if isinstance(sentiment, str):
      sentiment_df = pd.read_csv(sentiment)
    else:
      sentiment_df = sentiment

    entities = []

    #Iterates through each row of the sentiment_df to get the entities of the text
    #Creates dictionary of entities with each row 
    for index, row in sentiment_df.iterrows():
        entity= get_azure_named_entity_recognition(row['sentence_text'])
        entity_item = entity['results']['documents'][0]
        for col in sentiment_df.columns:
            entity_item[col] = row[col]    
        entities.append(entity_item)

    #Creates df of the entites list which is a list of dictionaries
    entities_df = pd.json_normalize(entities, record_path="entities", meta=list(sentiment_df.columns))

    #Renames columns as specified in read me
    entities_df.rename(columns={'text': 'entity_text'}, inplace=True)
    entities_df.rename(columns={'category': 'entity_category'}, inplace=True)
    entities_df.rename(columns={'subcategory': 'entity_subcategory'}, inplace=True)
    entities_df.rename(columns={'confidenceScore': 'confidenceScores.entity'}, inplace=True) 

    #Filters to columns specificd in directions
    entities_df = entities_df[['place_id', 'name', 'author_name', 'rating', 'sentence_text', 
                               'sentence_sentiment', 'confidenceScores.positive', 'confidenceScores.neutral', 'confidenceScores.negative', 
                               'entity_text', 'entity_category', 'entity_subcategory', 'confidenceScores.entity']]  
    
    #Saves df to cache and returns is
    entities_df.to_csv(CACHE_ENTITIES_FILE, index=False)
    return entities_df

if __name__ == '__main__':
    # helpful for debugging as you can view your dataframes and json outputs
    import streamlit as st 
    st.write("What do you want to debug?")

    reviews_step(PLACE_IDS_SOURCE_FILE)
    sentiment_step(CACHE_REVIEWS_FILE)
    entities_df = entity_extraction_step(CACHE_SENTIMENT_FILE)
    st.write(entities_df)

   