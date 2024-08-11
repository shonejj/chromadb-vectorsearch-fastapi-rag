from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Query
from pydantic import BaseModel
from typing import Optional
import fitz  # PyMuPDF for PDF text extraction
import os
import uuid
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# Initialize FastAPI app
app = FastAPI()

# Define the path for persistent storage
persistent_storage_path = './chroma_db'
os.makedirs(persistent_storage_path, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize ChromaDB Persistent Client and collection
client = PersistentClient(persistent_storage_path)
collection_name = 'pdf_collection'
embedding_function = embedding_functions.DefaultEmbeddingFunction()

collection = client.get_collection(collection_name)

def get_token(x_access_token: str = Header(...)):
    # Replace this with your actual token validation logic
    if x_access_token != "testtoken":
        raise HTTPException(status_code=403, detail="Invalid access token")
    return x_access_token

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file."""
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")
    return text

def chunk_text(text, chunk_size=1000, overlap=100):
    """Chunks text into manageable parts with overlap."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def create_documents_with_metadata(chunks, filename):
    """Creates documents with metadata, omitting the unique ID."""
    documents = []
    for i, chunk in enumerate(chunks):
        documents.append({
            "text": chunk,
            "metadata": {
                "filename": filename,
                "chunk_index": i
            }
        })
    return documents

def add_documents_to_collection(documents):
    """Adds documents to ChromaDB collection with dynamically generated IDs."""
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    # Generate unique IDs for each document
    ids = [str(uuid.uuid4()) for _ in documents]
    
    # Perform the add operation
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

@app.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Query(1000, le=5000, ge=100),  # Default value, max 5000, min 100
    overlap: Optional[int] = Query(100, le=1000, ge=0),        # Default value, max 1000, min 0
    token: str = Depends(get_token)
):
    """Upload and process a PDF file."""
    pdf_path = f'./temp_{uuid.uuid4()}.pdf'
    try:
        with open(pdf_path, "wb") as buffer:
            buffer.write(await file.read())

        text = extract_text_from_pdf(pdf_path)
        print(text)
        chunks = chunk_text(text, chunk_size, overlap)
        print(chunks)
        documents = create_documents_with_metadata(chunks, filename=file.filename)
        print(documents)
        add_documents_to_collection(documents)
        
        return {"message": f"Processed {len(documents)} documents from {file.filename}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)  # Clean up the temporary file

@app.post("/search/")
async def search(query: SearchQuery, token: str = Depends(get_token)):
    """Search for similar documents in the collection."""
    # Perform the search using the query text
    results = collection.query(
        query_texts=[query.query],  # ChromaDB will embed this for you
        n_results=query.top_k       # Number of results to return
    )
    
    return {"results": results}
