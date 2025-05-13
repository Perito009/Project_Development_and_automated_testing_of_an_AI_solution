import unittest
import pickle
import os

MODEL_PATH = 'spam_model.pkl'

class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
            cls.model = data['model']
            cls.vectorizer = data['vectorizer']

    def predict(self, text):
        X = self.vectorizer.transform([text])
        return self.model.predict(X)[0]

    def test_spam_message(self):
        # Positive case: typical spam
        text = "Congratulations! You've won a free ticket. Call now!"
        pred = self.predict(text)
        self.assertEqual(pred.lower(), "spam")

    def test_ham_message(self):
        # Positive case: typical ham
        text = "Hey, are we still meeting for lunch today?"
        pred = self.predict(text)
        self.assertEqual(pred.lower(), "ham")

    def test_empty_message(self):
        # Edge case: empty input
        text = ""
        pred = self.predict(text)
        self.assertIn(pred.lower(), ["ham", "spam"])  # Should not crash

    def test_special_characters(self):
        # Edge case: only special characters
        text = "!@#$%^&*()"
        pred = self.predict(text)
        self.assertIn(pred.lower(), ["ham", "spam"])  # Should not crash

    def test_long_message(self):
        # Edge case: very long message
        text = "hello " * 1000
        pred = self.predict(text)
        self.assertIn(pred.lower(), ["ham", "spam"])  # Should not crash

if __name__ == '__main__':
    unittest.main()
