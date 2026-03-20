# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
texts = [
    "This product is amazing and works perfectly",
    "Very good service and genuine seller",
    "Highly recommended and trustworthy",
    "Great quality and fast delivery",
    "Buy now!!! Limited offer!!!",
    "Click this link to win money",
    "This is fake and scam",
    "Don't trust this seller",
    "Free money offer click now",
    "Suspicious link do not open"
]

# Labels: 1 = trustworthy, 0 = suspicious
labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

# Convert text into numerical form
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Take user input
user_input = input("Enter text to check: ")

# Convert input text
input_data = vectorizer.transform([user_input])

# Predict probability
proba = model.predict_proba(input_data)

# Calculate trust score
trust_score = proba[0][1] * 100

# Output result
print(f"\nTrust Score: {trust_score:.2f}%")

if trust_score > 50:
    print("✅ This seems TRUSTWORTHY")
else:
    print("⚠️ This seems SUSPICIOUS")