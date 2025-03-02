import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = {
    "book_id": [1, 2, 3, 4, 5],
    "title": ["The Great Gatsby", "To Kill a Mockingbird", "1984", "Moby Dick", "War and Peace"],
    "description": [
        "The Great Gatsby, by F. Scott Fitzgerald, is a novel of the 1920s American novelist, African-American author, and literary critic. The novel tells the story of Jay Gatsby, a wealthy, middle-class man living in the South during the Great Depression, struggling to find meaning in his life and his love for Daisy Buchanan. The story explores themes of self-discovery, ambition, and the dangers of materialism.",
        "To Kill a Mockingbird, by Harper Lee, is a 1960 American novel written by Harper Lee. The novel tells the story of a black man named Tom Robinson, who is wrongfully accused of raping a white woman and is sentenced to death by a brutal murderer. The novel explores themes of racial injustice, inequality, and the human cost of injustice.",
        "1984, by George Orwell, is a dystopian novel published in 1949. The novel tells the story of World War II-era British resistance against Nazi Germany. The novel explores themes of surveillance, communism, and the human condition in a society where the press is controlled by the government and the media is used to oppress the people.",
        "Moby Dick, by Herman Melville, is a 1851 novel published in 1851. The novel tells the story of Captain Ahab, a whaleman who encounters various unusual and dangerous sea creatures. The novel explores themes of courage, duty, and the human capacity for adventure.",
        "War and Peace, by Leo Tolstoy, is a 1869 novel published in 1869. The novel tells the story of the Russian protagonist, Napoleon Bonaparte, and his struggle to reclaim Russia from the Ottoman Empire. The novel explores themes of power, loyalty, and the human capacity for self-discovery and growth."
    ]
}

df = pd.DataFrame(data)

vertorizer = TfidfVectorizer()
tfidf_matrix = vertorizer.fit_transform(df["description"])

cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_book(title, n=3):
    idx=df[df['title'] == title].index[0]
    sim_score = list(enumerate(cosine_sim[idx]))
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = sim_score[1:n+1]
    recommended_indices = [i[0] for i in sim_score]
    return df.iloc[recommended_indices][['title', 'description']]

print(recommend_book("The Great Gatsby"))