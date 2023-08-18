from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext

def analyze_text_sentiment(text):
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 2)
    polarity = round(blob.sentiment.polarity, 2)
    subjectivity = round(blob.sentiment.subjectivity, 2)
    return polarity, subjectivity

def analyze_tweet_score(x):
    blob = TextBlob(x)
    return blob.sentiment.polarity

def analyze_sentiment_category(x):
    if x >= 0.5:
        return 'Positive'
    elif x <= -0.5:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    st.header('Sentiment Analysis')

    with st.expander('Analyze Text'):
        text = st.text_input('Text here:')
        if text:
            polarity, subjectivity = analyze_text_sentiment(text)
            st.write('Polarity:', polarity)
            st.write('Subjectivity:', subjectivity)

        pre = st.text_input('Clean Text:')
        if pre:
            cleaned_text = cleantext.clean(
                pre, clean_all=False, extra_spaces=True,
                stopwords=True, lowercase=True, numbers=True, punct=True)
            st.write(cleaned_text)

    with st.expander('Analyze CSV'):
        upl = st.file_uploader('Upload file')

        if upl:
            df = pd.read_excel(upl)
            if 'Unnamed: 0' in df.columns:
                del df['Unnamed: 0']
            df['score'] = df['tweets'].apply(analyze_tweet_score)
            df['analysis'] = df['score'].apply(analyze_sentiment_category)
            st.write(df.head(10))

            @st.cache
            def convert_df_to_csv(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df_to_csv(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )

if __name__ == '__main__':
    main()
