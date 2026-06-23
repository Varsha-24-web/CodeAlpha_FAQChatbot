from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "What is AI?",
    "What is Machine Learning?",
    "What is Python?",
    "What is NLP?"
]

answers = [
    "AI stands for Artificial Intelligence.",
    "Machine Learning is a subset of AI.",
    "Python is a programming language.",
    "NLP stands for Natural Language Processing."
]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

print("FAQ Chatbot Started!")
print("Type 'exit' to quit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, faq_vectors)

    index = similarity.argmax()

    print("Bot:", answers[index])