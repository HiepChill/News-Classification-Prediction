import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

df = pd.read_csv("bbc_data.csv")

# Convert to lower case
df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

# Remove newline characters
df = df.apply(lambda x: x.str.replace("\n", " ") if x.dtype == "object" else x)

# Remove non-word and non-whitespace characters
df = df.replace(to_replace=r"[^\w\s]", value="", regex=True)

# Remove digits
df = df.replace(to_replace=r"\d", value="", regex=True)

# Tokenization
df["content"] = df["content"].apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words("english"))
df["content"] = df["content"].apply(
    lambda x: [word for word in x if word not in stop_words]
)

# Lemmatization
# initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# define function to lemmatize tokens
def lemmatize_tokens(tokens):
    # convert POS tag to WordNet format
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        return tag_dict.get(tag, wordnet.NOUN)

    # lemmatize tokens
    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

    # return lemmatized tokens as a list
    return lemmas


# apply lemmatization function to column of dataframe
df["lemmatized_content"] = df["content"].apply(lemmatize_tokens)
df["final_content"] = df["lemmatized_content"].apply(lambda x: " ".join(x))

print(df.head())
print(df.tail())

df.to_csv("bbc_clean.csv", index=False)