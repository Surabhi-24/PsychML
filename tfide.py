from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv("train.txt", sep="\t", names=["text", "label"])

train_df = pd.read_csv("train.txt", sep=" ", names=["text", "label"])
val_df   = pd.read_csv("val.txt", sep=" ", names=["text", "label"])
test_df  = pd.read_csv("test.txt", sep=" ", names=["text", "label"])



vectorizer = TfidfVectorizer(max_features=5000)
X=vectorizer.fit_transform(texts).toarray()
y=labels

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))