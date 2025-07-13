from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def check_plagiarism(text1, text2):
    # Create TF-IDF vectors for both texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100  # Convert to percentage

# Example usage
def main():
    print("📄 Plagiarism Checker\n")

    # Input text manually or read from files
    text1 = input("Enter the first text:\n")
    print("\n---\n")
    text2 = input("Enter the second text:\n")

    similarity_score = check_plagiarism(text1, text2)
    print(f"\n🔍 Similarity Score: {similarity_score:.2f}%")

    if similarity_score > 80:
        print("⚠️ High similarity detected. Possible plagiarism.")
    elif similarity_score > 50:
        print("🧐 Moderate similarity. Review recommended.")
    else:
        print("✅ Low similarity. Likely original.")

if __name__ == "__main__":
    main()

