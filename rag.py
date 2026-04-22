import os
import warnings
from transformers import pipeline
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Suppress huggingface_hub FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")


# ==============================
# STEP 1: SETUP DATA
# ==============================
DATA_PATH = "data/data.txt"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("data/data.txt not found. Create it and add some content.")


# ==============================
# STEP 2: LOAD DOCUMENT
# ==============================
loader = TextLoader(DATA_PATH)
docs = loader.load()


# ==============================
# STEP 3: SPLIT DOCUMENT
# ==============================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)


# ==============================
# STEP 4: EMBEDDINGS
# ==============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==============================
# STEP 5: VECTOR DB (FAISS)
# ==============================
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})


# ==============================
# STEP 6: LLM SETUP
# ==============================
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=100
)

llm = HuggingFacePipeline(pipeline=pipe)


# ==============================
# STEP 7: RAG FUNCTION
# ==============================
def rag_chain(question: str) -> str:
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in retrieved_docs])

    prompt = f"""Answer ONLY using the context below.
If not found, say: I don't know.
Keep answer short (1-2 lines). No repetition.

Context:
{context}

Question: {question}

Answer:"""

    raw = llm.invoke(prompt)

    answer = raw.replace(prompt, "").strip()
    answer = answer.split("\n")[0]

    return raw 


# ==============================
# STEP 8: RUN
# ==============================
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question about TCS (or type 'exit'): ")

        if query.lower() == "exit":
            break

        response = rag_chain(query)
        print("Answer:", response)