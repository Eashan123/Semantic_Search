import logging
import time
import waitress

from flask import Flask, render_template, request, jsonify
from semantic_search.utils import timer
from semantic_search.dataset import Dataset
from semantic_search.sentence_similarity import SentenceSimilarity

app = Flask(__name__)

logging.basicConfig(format="%(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)

dataset = timer(Dataset, "data/final_description_data.txt")

data = set([i["text"] for i in dataset.data])

sentence_sim = timer(SentenceSimilarity, dataset=dataset)


@app.route("/")
def home():
    return render_template("search.html")


@app.route("/auto_complete", methods=["POST"])
def auto_completion():
    question = request.form["input"]
    s = time.time()
    hits = [doc[:100] for doc in data if question.lower() in doc][:5]
    print(f" * Took: {round((time.time() - s) * 1000)} ms")
    res = {"hits": hits}
    # print(f"hits:{res}")
    return jsonify(hits=res["hits"])


@app.route("/search", methods=["POST"])
def search_request():
    query = request.form["input"]
    ids, most_sim_docs = sentence_sim.get_most_similar(query)

    hits = [{"body": doc} for doc in most_sim_docs]
    res = {"total": len(most_sim_docs), "hits": hits}

    print(f"result: {res['total']}")
    return render_template(
        "results.html", 
        question=query, 
        total=len(ids), 
        response=zip(res['hits'], ids)
        )

    return res


if __name__ == "__main__":
    # waitress.serve(app, listen="0.0.0.0:7001", threads=4)
    waitress.serve(app, host="0.0.0.0", port=5000, threads=4)
