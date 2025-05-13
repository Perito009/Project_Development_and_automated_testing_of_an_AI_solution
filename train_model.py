import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Example dataset (expand with your real data)
texts = [
    "Congratulations! You've won a free ticket. Call now!",
    "Win a free iPhone by clicking this link!",
    "Hey, are we still meeting for lunch today?",
    "Don't forget to bring your notebook to the meeting.",
    "URGENT! You have won a 1 week FREE membership in our prize draw!",
    "Can you send me the report by tomorrow?",
    "Free entry in 2 a weekly competition to win FA Cup final tickets",
    "Are you coming to the party tonight?",
]
labels = [
    "spam",
    "spam",
    "ham",
    "ham",
    "spam",
    "ham",
    "spam",
    "ham",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = RandomForestClassifier(random_state=42)
model.fit(X, labels)

with open('spam_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
