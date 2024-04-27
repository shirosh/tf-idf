import docx
import pandas as pd
""" import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download() """
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Read law book document
def read_docx(file_path):
    doc = docx.Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

book_text = read_docx('book.docx')

# Step 2: Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization
    stop_words = set(stopwords.words('dutch'))
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]  # Stemming
    return ' '.join(stemmed_tokens)

book_text_processed = preprocess_text(book_text)

# Step 3: Load and parse taxonomy
taxonomy_df = pd.read_excel('taxonomy.xlsx')
taxonomy_hierarchy = {}  # Assuming taxonomy is hierarchical

# Iterate through each row in the DataFrame
for index, row in taxonomy_df.iterrows():
    # Extract keywords from columns 'Trefwoord 1', 'Trefwoord 2', and 'Trefwoord 3'
    keyword1 = str(row['Trefwoord 1'])
    keyword2 = str(row['Trefwoord 2'])
    keyword3 = str(row['Trefwoord 3'])
    
    # Build the hierarchy using nested dictionaries
    if keyword1 not in taxonomy_hierarchy:
        taxonomy_hierarchy[keyword1] = {}
    if keyword2 not in taxonomy_hierarchy[keyword1]:
        taxonomy_hierarchy[keyword1][keyword2] = []
    taxonomy_hierarchy[keyword1][keyword2].append(keyword3)

# Step 4: Matching Algorithm
# Vectorize law book sections and taxonomy entries using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
book_tfidf = tfidf_vectorizer.fit_transform([book_text_processed])
taxonomy_entries = list(taxonomy_hierarchy.keys())
taxonomy_tfidf = tfidf_vectorizer.transform(taxonomy_entries)

# Calculate cosine similarity between law book sections and taxonomy entries
similarity_scores = cosine_similarity(book_tfidf, taxonomy_tfidf)

# Step 5: Assignment Process
section_taxonomy_assignments = {}
for i, section in enumerate(similarity_scores):
    max_similarity_index = section.argmax()
    max_similarity_score = section.max()
    taxonomy_assignment = taxonomy_entries[max_similarity_index]
    section_taxonomy_assignments[i] = taxonomy_assignment, max_similarity_score

# Print section-taxonomy assignments
for section_idx, (taxonomy_assignment, similarity_score) in section_taxonomy_assignments.items():
    print(f"Section {section_idx + 1}: Assigned to '{taxonomy_assignment}' with similarity score {similarity_score:.2f}")
