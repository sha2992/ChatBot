from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# loading files
def load_pdf_files(data):
    loader = DirectoryLoader(data
                             ,glob="*.pdf"
                             ,loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

Data_path = "data/"

documents = load_pdf_files(Data_path)
print("Pages number in the document is: ",len(documents))

### Creating Chunks

def create_chunks(extracted_documents):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=800
                                            , chunk_overlap=100)
    
    text_chunks = text_spliter.split_documents(extracted_documents)
    return text_chunks

text_chunks = create_chunks(documents)
print("length of text chunks: ", len(text_chunks))

#Vectore Embeddings

def embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = embedding_model()

#Store embedding vector
DB_FAISS_path = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_path)