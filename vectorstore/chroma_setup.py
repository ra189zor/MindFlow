# vectorstore/chroma_setup.py
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

# Load environment variables specifically for this module if needed
load_dotenv()

# Note: This setup seems unused by the agent logic provided so far.
# Ensure agents have a tool or mechanism to interact with this if needed.
def get_collection():
    """
    Initialize ChromaDB client and return a collection 'research_docs'.
    Populates with dummy documents if empty.
    Returns None if initialization fails.
    """
    chroma_openai_api_key = os.getenv("OPENAI_API_KEY")
    if not chroma_openai_api_key:
        print("Warning: OPENAI_API_KEY not found for ChromaDB embedding function.")
        # Decide error handling: raise error, return None, or use default key if applicable
        raise ValueError("Missing OPENAI_API_KEY required for ChromaDB embeddings.")

    try:
        # Define OpenAI embedding function within the function scope
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=chroma_openai_api_key,
            model_name="text-embedding-ada-002"
        )

        # Initialize ChromaDB client with persistent storage
        # Ensure the directory exists or ChromaDB can create it
        client = chromadb.PersistentClient(path="./chroma_data")

        # Get or create a collection
        collection = client.get_or_create_collection(
            name="research_docs",
            embedding_function=openai_ef
            # Consider adding metadata={"hnsw:space": "cosine"} for cosine similarity
        )

        # If the collection is empty, add dummy documents
        if collection.count() == 0:
            print("ChromaDB collection 'research_docs' is empty. Adding dummy documents.")
            documents = [
                "Search Engine Optimization (SEO) is crucial for improving website visibility and ranking on search engines like Google. Keywords are fundamental.",
                "Content marketing focuses on creating and distributing valuable, relevant, and consistent content to attract and retain a clearly defined audience.",
                "Social media marketing utilizes platforms like Facebook, Instagram, and Twitter to build brand awareness, engage customers, and drive website traffic."
            ]
            # Ensure metadata and IDs match the number of documents
            metadatas = [{"source": f"dummy_doc_{i+1}"} for i in range(len(documents))]
            ids = [f"dummy_doc_{i+1}" for i in range(len(documents))]

            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"{len(documents)} dummy documents added to ChromaDB.")

        return collection

    except Exception as e:
        # Catch potential errors during client init, collection creation, or adding docs
        print(f"Error initializing ChromaDB or adding documents: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None