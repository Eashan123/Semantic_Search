import re
import string
import time
import scipy
import logging
import tqdm
import numpy as np
import joblib
import nltk
nltk.download('stopwords')

from sentence_transformers import SentenceTransformer
from typing import List, AnyStr
from semantic_search.dataset import Dataset
from nltk.corpus import stopwords
from collections import Counter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stop_words = stopwords.words('english')
stop_words_dict = Counter(stop_words)

class SentenceSimilarity:
    def __init__(
        self,
        dataset: Dataset,
        model: SentenceTransformer = None,
        n_docs: int = -1,
        device: str = None,
        batch_size: int = 64,
        show_progress_bar: bool = True,
        num_workers: int = 2,
    ):

        self.dataset = dataset
        self.model = (
            model if model else SentenceTransformer(
                "distilbert-base-nli-mean-tokens")
        )
        # bert-base-nli-stsb-mean-tokens
        # bert-large-nli-mean-tokens

        self.sentences = []
        self.doc_id_to_sentence_ids = {}

        self.sentence_pattern = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?])\s")
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.num_workers = num_workers
        self.device = device

        for d in tqdm.tqdm(dataset.get_documents(n=n_docs), total=len(dataset)):
            doc_id = d.get("id")
            text = d.get("text", None)

            sentence_ids = []
            if text:
                text = ' '.join(text.split())
                text = ' '.join([word for word in text.split() if word not in stop_words_dict])
                text = re.sub(r'[^ ]+\.[^ ]+', '', text)
                text = re.sub(r'\S*@\S*\s?', '', text)
                text = re.sub(r'\w*\d\w*', '', text)
                text = re.sub(r'[^\W\d]*$', '', text)
                text = re.sub(r"\s+", " ", text)
                text = re.sub(r"\w*\d\w*", "", text)
                text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
                text = text.encode('ascii', 'replace').decode()
                text = re.sub(r"  +", " ", text)
                text = ' '.join([word for word in text.split() if len(word) > 3])

                sentences = re.split(self.sentence_pattern, text)
                for s in sentences:
                    sentence_ids.append(len(self.sentences))
                    self.sentences.append(s)

            # Map from document to all its sentences (One-to-Many)
            self.doc_id_to_sentence_ids[doc_id] = sentence_ids

        logger.debug(f"doc_to_sentence_ids: {self.doc_id_to_sentence_ids}")

        # Map from sentence to the document it came from (Many-to-One)
        self.sentence_id_to_doc_id = {}
        for doc_id, sentence_ids in self.doc_id_to_sentence_ids.items():
            for s_id in sentence_ids:
                self.sentence_id_to_doc_id[s_id] = doc_id

        logger.debug(f"sentence_id_to_doc_id: {self.sentence_id_to_doc_id}")
        # Embedd extracted sentences using SentenceTransformer model.
        start = time.time()
        self.embedded_sentences = joblib.load(
            "semantic_search/embeddings/emb.pkl")
        logger.info(
            f"It took {round(time.time() - start, 3)} s to embedd "
            f"{len(self.sentences)} sentences."
        )

    def get_most_similar(
        self, query: AnyStr, threshold: float = 0.15, limit: int = 10
    ) -> List[str]:

        query_sentences = re.split(self.sentence_pattern, query)
        query_embeddings = self.model.encode(query_sentences)

        logger.info(f"Extracted {len(query_sentences)} sentences from query")
        logger.debug(f"Sentences: {' -- '.join(query_sentences)}")

        # Calculate cosine distance between requested sentences and all sentences

        cosine_dist = scipy.spatial.distance.cdist(
            query_embeddings, self.embedded_sentences, "cosine"
        )

        # Extract column values where distance is below threshold
        below_threshold = cosine_dist < threshold

        doc_ids, matched_column_ids = np.where(below_threshold)

        # Extract x (input sentence id), y (dataset sentence id) and distance between these.
        x_y_dist = []
        for x, y in zip(doc_ids, matched_column_ids):
            x_y_dist.append([x, y, cosine_dist[x][y]])

        # Sort list based on distance and remove duplicates, keeping the one with lowest distance.
        sorted_x_y_dist = sorted(x_y_dist, key=lambda z: z[2])

        sorted_sentence_ids = [doc[1] for doc in sorted_x_y_dist]
        sorted_doc_ids = [
            self.sentence_id_to_doc_id[sent_id] for sent_id in sorted_sentence_ids
        ]

        result = self.dataset.get_documents_by_id(
            list(dict.fromkeys(sorted_doc_ids).keys())[:limit]
        )

        result_ids = zip(sorted_doc_ids, result)
        final_responses = list()
        ids = list()
        for idx, sent in result_ids:
            sent_id = str(idx)
            sent_text = sent
            final_sentence = sent_text
            ids.append(idx)
            final_responses.append(final_sentence)

        logger.info(
            f"Distances for top documents: "
            f"{[round(x[2], 3) for x in sorted_x_y_dist[:limit]]}"
        )
        return ids, final_responses
