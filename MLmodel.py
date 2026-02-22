import pandas as pd
train_df = pd.read_csv("data/train.txt",sep=";",names=["text", "emotion"])
test_df = pd.read_csv("data/test.txt", sep=";", names=["text", "emotion"])
val_df  = pd.read_csv("data/val.txt", sep=";", names=["text", "emotion"])

from preprocess import load_and_preprocess

train_df = load_and_preprocess("data/train.txt")
test_df = load_and_preprocess("data/test.txt")

X_train = train_df["clean_text"]
y_train = train_df["emotion"]

X_test = test_df["clean_text"]
y_test = test_df["emotion"]

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from sklearn.svm import LinearSVC

model_svm = LinearSVC(max_iter=1000, class_weight="balanced")
model_svm.fit(X_train_tfidf, y_train)

y_pred_svm = model_svm.predict(X_test_tfidf)
print("Accuracy (SVM):", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

models={
    "logistic_regression": model,
    "svm": model_svm
}
for name, mdl in models.items():
    y_pred = mdl.predict(X_test_tfidf)
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))



import pickle 
import os
os.makedirs("models", exist_ok=True)
with open("models/logistic_regression.pkl", "wb") as f: pickle.dump(model, f)
with open("models/svm.pkl", "wb") as f: pickle.dump(model_svm, f)
with open("models/tfidf_vectorizer.pkl", "wb") as f: pickle.dump(tfidf, f)  
print("Models and vectorizer saved successfully.")