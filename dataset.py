# import pandas as pd
# import numpy as np

# df1=pd.read_csv("train.csv", names=['sentiment_label','review_text'])
# df2=pd.read_csv('twitter_training.csv', names=['sentiment_label','review_text'])

# # df2_subset=df2.head(15000)  it takes first n rows
# df2_subset=df2.sample(30000) #it takes randomly
# combined_df=pd.concat([df1,df2_subset],ignore_index=True)
# combined_df.to_csv('final_combined.csv', index=False)

# df=pd.read_csv("final_combined.csv", names=['sentiment_label','review_text'])
# df1=pd.DataFrame(df)
# # print(df1)

# print(df1.describe())
# print(df1.info())

# print(df1.isnull().sum())
# df1=df1.dropna()
# df1=df1.drop_duplicates()


# import re

# def easy_clean(text):
#     # 1. Lowercase everything
#     text = str(text).lower()
    
#     # 2. Remove URLs and Special Characters
#     # This regex removes http/https and anything that isn't a letter or space
#     text = re.sub(r'@[^\s]+', '[USER]', text) # Neutralize usernames
#     text = re.sub(r"http\S+|[^a-z\s]", "", text)
#     text = re.sub(r"[^a-z\s\[\]]", "", text)
   
#     # 3. Clean up extra spaces
#     return " ".join(text.split())


# # Capitalize the first letter only
# df1['sentiment_label'] = df1['sentiment_label'].str.capitalize()
# # Apply it to your column
# df1['review_text'] = df1['review_text'].apply(easy_clean)

# # 1. Ensure the column is string type
# df1['review_text'] = df1['review_text'].astype(str)
# df1['review_text']=df1['review_text'].str.lower()

# # 1. Strip leading whitespace to ensure the '@' check is accurate
# df1['review_text'] = df1['review_text'].str.strip()
# df1['review_text']=df1['review_text'].str.rstrip('.')


# # 2. Filter by Label (Exact match is case-sensitive!)
# # df1 = df1[df1['sentiment_label'] == 'Irrelevant'].reset_index(drop=True)
# df1 = df1[df1['sentiment_label'].isin(['Positive', 'Neutral', 'Negative'])].reset_index(drop=True)

# # 3. Filter by Content (More than 1 word and NOT starting with @)
# df1=df1[df1['review_text'].str.split().str.len()>1].reset_index(drop=True)

# df1=df1[~df1['review_text'].str.startswith('@')].reset_index(drop=True)

# # 4. Verify and Save
# print(df1['sentiment_label'].value_counts())
# df1.to_csv('cleaned_twitter_data.csv', index=False)


# # 4. Verify
# print(f"Remaining rows: {len(df1)}")
# print(df1.head())
# print(df1['sentiment_label'].value_counts())
# df1.to_csv('cleaned_twitter_data.csv',index=False)



import pandas as pd
import numpy as np
import re

# 1. Loading and Combining
df1 = pd.read_csv("train.csv", names=['sentiment_label','review_text'],encoding='utf-8', encoding_errors='ignore')
df2 = pd.read_csv('twitter_training.csv', names=['sentiment_label','review_text'],encoding='utf-8', encoding_errors='ignore')

df2_subset = df2.sample(15000, random_state=42) 
df1 = pd.concat([df1, df2_subset], ignore_index=True)

# 2. Initial Cleaning
df1 = df1.dropna().drop_duplicates()
df1['sentiment_label'] = df1['sentiment_label'].str.capitalize()
df1['review_text'] = df1['review_text'].str.strip()

# 3. Filtering to 3 classes
df1 = df1[df1['sentiment_label'].isin(['Positive', 'Neutral', 'Negative'])].reset_index(drop=True)

# 4. Improved Cleaning Function
def improved_clean(text):
    text = str(text).lower()
    # Replace handles and links with placeholders GloVe might recognize or ignore safely
    text = re.sub(r'@[^\s]+', 'user', text) 
    text = re.sub(r"http\S+", 'url', text)
    # Remove special chars but keep spaces
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(text.split())

df1['review_text'] = df1['review_text'].apply(improved_clean)

# 5. Final Filtering
# Remove empty or single-word tweets
df1 = df1[df1['review_text'].str.split().str.len() > 1].reset_index(drop=True)

# 1. Separate the classes
df_pos = df1[df1['sentiment_label'] == 'Positive']
df_neg = df1[df1['sentiment_label'] == 'Negative']
df_neu = df1[df1['sentiment_label'] == 'Neutral']

# 2. Determine your target size for Neutral
# For example, make it 70% of the size of the smallest other class
target_size = int(min(len(df_pos), len(df_neg)) * 0.85)

# 3. Randomly sample the Neutral class down to that size
df_neu_downsampled = df_neu.sample(target_size, random_state=42)

# 4. Combine back together and shuffle
df1 = pd.concat([df_pos, df_neg, df_neu_downsampled])
df1 = df1.sample(frac=1, random_state=42).reset_index(drop=True)

# 6. Save for Training
df1.to_csv('cleaned_twitter_data.csv', index=False)
print("Cleaning Complete. Class counts:")
print(df1['sentiment_label'].value_counts())