import os
import re
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DEFAULT_KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledgebase", "info.md")
COLLECTION_NAME = "ai_agent_xcelerator_kb"

class RAGEngine:
    def __init__(self, persist_dir: str = CHROMA_DATA_DIR):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Use Chroma's default sentence-transformers ONNX embedding function
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"description": "AI Agent Xcelerator Program 2026 Knowledge Base"}
        )

    def load_and_chunk_markdown(self, filepath: str = DEFAULT_KB_PATH) -> List[Dict[str, Any]]:
        """
        Parses the markdown file into semantically meaningful chunks
        by preserving section hierarchy and key table details.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Knowledge base file not found at {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        chunks: List[Dict[str, Any]] = []
        
        # Split into main sections by H1 / H2 / H3 headers
        # Use regex to keep the headers with the text
        sections = re.split(r'\n(?=#{1,3}\s+)', raw_text)

        chunk_id = 0
        current_super_header = "AI Agent Xcelerator Program"

        for section in sections:
            section_clean = section.strip()
            if not section_clean:
                continue

            lines = section_clean.split("\n")
            first_line = lines[0].strip()

            # Detect main header
            if first_line.startswith("# "):
                current_super_header = first_line.replace("#", "").strip()
            
            section_title = first_line.replace("#", "").strip()

            # If section is very large (e.g. over 800 chars), split by sub-blocks or paragraphs
            if len(section_clean) > 800:
                paragraphs = section_clean.split("\n\n")
                sub_chunk = ""
                for p in paragraphs:
                    p_clean = p.strip()
                    if not p_clean:
                        continue
                    if len(sub_chunk) + len(p_clean) < 700:
                        sub_chunk += "\n\n" + p_clean if sub_chunk else p_clean
                    else:
                        if sub_chunk:
                            chunk_id += 1
                            chunks.append({
                                "id": f"chunk_{chunk_id}",
                                "text": f"Context: {current_super_header} > {section_title}\n\n{sub_chunk}",
                                "title": section_title,
                                "category": current_super_header
                            })
                        sub_chunk = p_clean
                if sub_chunk:
                    chunk_id += 1
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": f"Context: {current_super_header} > {section_title}\n\n{sub_chunk}",
                        "title": section_title,
                        "category": current_super_header
                    })
            else:
                chunk_id += 1
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": f"Context: {current_super_header}\n\n{section_clean}",
                    "title": section_title,
                    "category": current_super_header
                })

        return chunks

    def ingest_knowledge_base(self, filepath: str = DEFAULT_KB_PATH, force_reload: bool = False) -> int:
        """
        Ingests the markdown knowledge base into the ChromaDB collection.
        If force_reload is True or collection is empty, re-embeds all chunks.
        """
        current_count = self.collection.count()
        if current_count > 0 and not force_reload:
            return current_count

        if force_reload and current_count > 0:
            # Delete existing items in collection
            existing_ids = self.collection.get()["ids"]
            if existing_ids:
                self.collection.delete(ids=existing_ids)

        chunks = self.load_and_chunk_markdown(filepath)
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metadatas = [{"title": c["title"], "category": c["category"]} for c in chunks]

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas
        )
        return len(chunks)

    def retrieve(self, query: str, n_results: int = 4) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search against the indexed knowledge base.
        """
        if self.collection.count() == 0:
            self.ingest_knowledge_base()

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0] if "ids" in results else [""] * len(docs)

            for i in range(len(docs)):
                retrieved.append({
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i]
                })

        return retrieved

    def get_context_string(self, query: str, n_results: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves context and formats it as a clean text string for the LLM.
        Also returns the raw retrieved items for display in the UI.
        """
        retrieved_items = self.retrieve(query, n_results=n_results)
        if not retrieved_items:
            return "", []

        context_blocks = []
        for idx, item in enumerate(retrieved_items, start=1):
            title = item.get("metadata", {}).get("title", f"Source {idx}")
            context_blocks.append(f"--- [Source {idx}: {title}] ---\n{item['document']}")

        return "\n\n".join(context_blocks), retrieved_items

    def get_stats(self) -> Dict[str, Any]:
        """Returns stats about the ChromaDB knowledge store."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "persist_dir": self.persist_dir
        }

if __name__ == "__main__":
    print("Testing RAG Engine...")
    engine = RAGEngine()
    count = engine.ingest_knowledge_base(force_reload=True)
    print(f"Ingested {count} chunks.")
    stats = engine.get_stats()
    print("Stats:", stats)
    
    test_query = "What is the duration and cohort size of the program?"
    context, sources = engine.get_context_string(test_query)
    print("\n--- Test Query:", test_query)
    print("--- Retrieved Context Preview:\n", context[:300], "...")
