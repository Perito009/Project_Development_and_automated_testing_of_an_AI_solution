import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle

# Load the CSV file
df = pd.read_csv('spam.csv', encoding='latin-1')

# Remove missing values
df_clean = df.dropna()

# Utilise seulement les colonnes nécessaires
X_raw = df_clean['v2']  # Texte
y = df_clean['v1']      # Label

# Vectorisation du texte
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_raw)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîne le modèle
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Évaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Sauvegarde du modèle et du vectorizer
with open('spam_model.pkl', 'wb') as f:
    pickle.dump({'model': clf, 'vectorizer': vectorizer}, f)

# Optionally, save the cleaned data
df_clean.to_csv('spam_clean.csv', index=False)

print("Loaded and cleaned data. Shape:", df_clean.shape)
