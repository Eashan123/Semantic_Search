from semantic_search.dataset import Dataset
from semantic_search.sentence_similarity import SentenceSimilarity

data = Dataset('../data/final_description_data.txt')
sentence_sim = SentenceSimilarity(data)

most_similar = sentence_sim.get_most_similar(
    'Facing hardware problem')

print(' '.join(most_similar))
