from logging import getLogger
from logging.config import fileConfig as logConfig
from datasets import load_dataset
import chromadb

logConfig("./logging.conf", disable_existing_loggers=False)
logger = getLogger(__name__)

def hello() -> str:
    logger.info("Hello")
    return "Hello"

def testchroma() -> str:
    dataset = load_dataset("sciq", split="train")
    dataset = dataset.filter(lambda x: x["support"] != "")
    client = chromadb.Client()
    collection = client.create_collection("sciq_supports")
    # Embed and store the first 100 supports for this demo
    collection.add(
        ids=[str(i) for i in range(0, 100)],  # IDs are just strings
        documents=dataset["support"][:100],
        metadatas=[{"type": "support"} for _ in range(0, 100)
        ],
    )

    results = collection.query(
        query_texts=dataset["question"][:10],
        n_results=1)

    for i, q in enumerate(dataset['question'][:10]):
        print(f"Question: {q}")
        print(f"Retrieved support: {results['documents'][i][0]}")
        print()

    print("Number of questions with support: ", len(dataset))
    return "done"

if __name__ == "__main__":  # pragma: no cover
    # print(hello())
    print(testchroma())