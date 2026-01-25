import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

def load_and_preprocess(path):
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["text", "emotion"],
        engine="python"
    )
    df["clean_text"] = df["text"].apply(clean_text)
    return df

if __name__ == "__main__":
    train_df = load_and_preprocess("data/train.txt")
    test_df = load_and_preprocess("data/test.txt")
    val_df = load_and_preprocess("data/val.txt")
    print(train_df.head())
    print(train_df.shape)
    print(train_df["emotion"].value_counts())

    print(test_df.head())
    print(test_df.shape)
    print(test_df["emotion"].value_counts())

    print(val_df.head())
    print(val_df.shape)
    print(val_df["emotion"].value_counts())