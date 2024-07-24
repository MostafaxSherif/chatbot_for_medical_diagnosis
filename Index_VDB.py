""""
This file will index the pdf medical documents into a Vector DB that will be using RAG to generate response
"""
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings


print("Reading PDF  documents.........")
# Read PDF documents
pdf_files = DirectoryLoader("Docs/",glob='*.pdf',loader_cls=PyPDFLoader)
documents = pdf_files.load()


print("Splitting documents........")
# Split text on predefined characters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=40)
doc_splits = text_splitter.split_documents(documents)

# Import SentenceTransformer embeding model (all-mpnet-base-v2 )
# All available models can be found here (https://www.sbert.net/docs/pretrained_models.html)
print("Creating  embeddings ........")

embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2',
                                       model_kwargs={'device': 'cpu'})

print("Saving  embeddings to VDB ........")

# Save embedding output to Vector DB
VDB = Chroma.from_documents(
    documents=doc_splits,
    embedding=embeddings_model,
    persist_directory="VDB/medicalDocs/"
)
print("Done!")


