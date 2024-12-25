import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn



def call_polisci() -> None:
    '''

    '''

    ### Story-Data
    import pandas as pd

    def compute_story_data() -> pd.DataFrame:

        # Load the data from 'docs.csv'

        story_data_result = pd.read_csv('docs.csv')

    

        # Remove any rows with NA values

        story_data_result.dropna(inplace=True)

    

        # Ensure the DataFrame has the required columns

        required_columns = ['doc_id', 'doc_text', 'KILL', 'ARREST', 'ANY_ACTION', 'FAIL', 'FORCE']

        story_data_result = story_data_result[required_columns]

    

        # Assertions to check requirements

        assert isinstance(story_data_result, pd.DataFrame), "story_data_result should be a pandas DataFrame."

        assert not story_data_result.isna().any().any(), "story_data_result should not contain any NA values."

        assert list(story_data_result.columns) == required_columns, "story_data_result should have the correct columns."

    

        return story_data_result

    ### Sentiment-Chart
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    def compute_sentiment_chart(story_data_result: pd.DataFrame) -> None:
        # Define the sentiment categories
        sentiment_categories = ['KILL', 'ARREST', 'ANY_ACTION', 'FAIL', 'FORCE']

        # Count the occurrences of each sentiment category
        sentiment_counts = {category: max(story_data_result[category].sum(), 0) for category in sentiment_categories}

        # Assertions to check the requirements
        assert all(category in sentiment_counts for category in sentiment_categories), "Not all sentiment categories are present in the sentiment_counts dictionary."
        assert all(count >= 0 for count in sentiment_counts.values()), "Some sentiment counts are negative."

        # Create a bar plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()))
        plt.title('Sentiment Counts')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()
        return None

    ### News-Sentiment
    import pandas as pd
    from textblob import TextBlob
    import re
 
    def compute_news_sentiment(story_data_result: pd.DataFrame) -> pd.DataFrame:
        # Preprocess text data
        def preprocess_text(text: str) -> str:
            # Convert to lowercase
            text = text.lower()
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            return text
    
        # Determine sentiment
        def determine_sentiment(text: str) -> str:
            analysis = TextBlob(text)
            # Classify as 'happy' if polarity is positive, otherwise 'sad'
            return 'happy' if analysis.sentiment.polarity > 0 else 'sad'
    
        # Apply preprocessing and sentiment analysis
        story_data_result['cleaned_text'] = story_data_result['doc_text'].apply(preprocess_text)
        story_data_result['sentiment'] = story_data_result['cleaned_text'].apply(determine_sentiment)
    
        # Assertions to check requirements
        assert 'doc_text' in story_data_result.columns, "Input DataFrame must contain 'doc_text' column"
        assert 'sentiment' in story_data_result.columns, "Output DataFrame must contain 'sentiment' column"
        assert all(story_data_result['sentiment'].isin(['happy', 'sad'])), "Sentiment values must be 'happy' or 'sad'"
    
        # Return DataFrame with 'doc_text' and 'sentiment' columns
        return story_data_result[['doc_text', 'sentiment']]

    ### Correlation-Matrix
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    def compute_correlation_matrix(sentiment_chart_result: None, story_data_result: pd.DataFrame) -> None:
        # Ensure no NA values
        story_data_result = story_data_result.dropna()
        # Extract relevant columns
        sentiment_columns = ['KILL', 'ARREST', 'ANY_ACTION', 'FAIL', 'FORCE']
        sentiment_data = story_data_result[sentiment_columns]
        # Calculate correlation matrix
        correlation_matrix = sentiment_data.corr()
        # Ensure correlation values are between -1 and 1
        assert correlation_matrix.min().min() >= -1 and correlation_matrix.max().max() <= 1, 'Correlation values should be between -1 and 1'
        # Plot heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix for Sentiments')
        plt.draw()
        return None

    ### Article-Count
    from typing import Dict

    import pandas as pd


    def compute_article_count(news_sentiment_result: pd.DataFrame, story_data_result: pd.DataFrame) -> Dict[str, int]:

        # Initialize a dictionary to store counts of 'happy' and 'sad' articles

        sentiment_counts = {'happy': 0, 'sad': 0}


        # Iterate over the 'sentiment' column in the DataFrame

        for sentiment in news_sentiment_result['sentiment']:

            if sentiment == 'happy':

                sentiment_counts['happy'] += 1

            elif sentiment == 'sad':

                sentiment_counts['sad'] += 1


        # Assertions to ensure the requirements are met

        assert isinstance(sentiment_counts, dict), "Output should be a dictionary."

        assert 'happy' in sentiment_counts and 'sad' in sentiment_counts, "Dictionary should have keys 'happy' and 'sad'."

        assert sentiment_counts['happy'] >= 0 and sentiment_counts['sad'] >= 0, "Counts should be non-negative."

        assert sentiment_counts['happy'] + sentiment_counts['sad'] == len(news_sentiment_result), "Total count should match the number of articles."


        return sentiment_counts


    ### fd2b9193-2421-4490-8b7c-63ebc9987dcf
    story_data_result = compute_story_data()


    ### Step-1
    sentiment_chart_result = compute_sentiment_chart(story_data_result)


    ### Step-3
    news_sentiment_result = compute_news_sentiment(story_data_result)


    ### Step-2
    correlation_matrix_result = compute_correlation_matrix(sentiment_chart_result,story_data_result)


    ### Step-4
    article_count_result = compute_article_count(news_sentiment_result,story_data_result)


    return None
