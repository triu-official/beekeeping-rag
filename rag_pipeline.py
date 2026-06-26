import re
import json
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGPipeline:
    """
    A simple Retrieval-Augmented Generation (RAG) pipeline.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the RAG pipeline with a specified SentenceTransformer model.

        Args:
            model_name: The HuggingFace model string.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.document = ""
        self.chunks: List[str] = []
        self.chunk_embeddings: np.ndarray = np.array([])

    def load_document(self, text: str):
        """
        Loads the document into the pipeline.

        Args:
            text: The full string document.
        """
        self.document = text

    def create_chunks(self, window_size: int = 2, overlap: int = 1) -> List[str]:
        """
        Splits the document into overlapping chunks of sentences.

        Args:
            window_size: Number of sentences per chunk.
            overlap: Number of sentences overlapping between consecutive chunks.

        Returns:
            A list of chunk strings.
        """
        sentences = re.split(r'(?<=[.!?])\s+', self.document.strip())
        sentences = [s for s in sentences if s]

        chunks = []
        step = window_size - overlap
        if step <= 0:
            step = 1

        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i : i + window_size]
            if not chunk_sentences:
                break
            chunks.append(" ".join(chunk_sentences))

            if i + window_size >= len(sentences):
                break

        self.chunks = chunks
        return self.chunks

    def embed_chunks(self):
        """
        Computes the embeddings for the created chunks and normalizes them.
        """
        if not self.chunks:
            return

        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.chunk_embeddings = embeddings / np.where(norms == 0, 1e-10, norms)

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        Retrieves the top-K chunks for a given query based on cosine similarity.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / np.where(query_norm == 0, 1e-10, query_norm)

        similarities = np.dot(self.chunk_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.chunks[i], float(similarities[i])) for i in top_indices]

    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
        """
        Extracts the best matching sentence from the retrieved chunks as the answer.
        """
        if not retrieved_chunks:
            return ""

        # Combine chunks and split back into sentences to find the best single sentence
        context = " ".join([chunk for chunk, _ in retrieved_chunks])
        sentences = re.split(r'(?<=[.!?])\s+', context.strip())
        # Preserve original order while making unique
        seen = set()
        unique_sentences = []
        for s in sentences:
            if s and s not in seen:
                seen.add(s)
                unique_sentences.append(s)

        if not unique_sentences:
            return ""

        sentence_embeddings = self.model.encode(unique_sentences, convert_to_numpy=True)
        sentence_norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        sentence_embeddings = sentence_embeddings / np.where(sentence_norms == 0, 1e-10, sentence_norms)

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding / np.where(query_norm == 0, 1e-10, query_norm)

        # We compute similarity for sentence chunks using cosine similarity
        similarities = np.dot(sentence_embeddings, query_embedding.T).flatten()

        # To improve extraction, we can boost sentences that contain question keywords or numbers
        # since expected answers often contain specific facts from the text.
        query_words = set(re.findall(r'\w+', query.lower()))
        for i, sent in enumerate(unique_sentences):
            sent_words = set(re.findall(r'\w+', sent.lower()))
            overlap = len(query_words.intersection(sent_words))
            # Boost score based on word overlap to counteract embedding biases (e.g. favoring generic statements)
            similarities[i] += overlap * 0.05

        best_idx = np.argmax(similarities)
        return unique_sentences[best_idx]


if __name__ == "__main__":
    try:
        with open("dataset.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: dataset.json not found. Please ensure it exists.")
        exit(1)

    document = data.get("document", "")
    qa_pairs = data.get("qa_pairs", [])

    pipeline = RAGPipeline()
    print(f"Model used: {pipeline.model_name}")

    pipeline.load_document(document)
    pipeline.create_chunks(window_size=2, overlap=1)
    print(f"Chunk count: {len(pipeline.chunks)}")

    pipeline.embed_chunks()

    for qa in qa_pairs:
        query = qa["question"]
        print(f"\nQuery: {query}")

        retrieved = pipeline.retrieve(query, top_k=2)
        print("Retrieved chunks and scores:")
        for chunk, score in retrieved:
            print(f"  - Score: {score:.4f} | Chunk: {chunk}")

        answer = pipeline.generate_answer(query, retrieved)
        print(f"Generated Answer: {answer}")
        print("-" * 50)
