import pickle

# Charger le modèle et le vectorizer
with open('spam_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    vectorizer = data['vectorizer']

# Lire une entrée utilisateur
text = input("Entrez le message à classer : ")

# Transformer et prédire
X = vectorizer.transform([text])
prediction = model.predict(X)[0]

print(f"Prédiction : {prediction}")
