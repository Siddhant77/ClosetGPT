import os
import pathlib
import json
import torch
from dotenv import load_dotenv
from typing import List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

from langchain.docstore.in_memory import InMemoryDocstore
import faiss
import openai
import clip

import numpy as np
import re

from .data.datasets import polyvore
from .data.datatypes import Outfit

# Setup
load_dotenv()
SRC_DIR = pathlib.Path(__file__).parent.parent.absolute()
OUTFITS_PATH = SRC_DIR / 'datasets' / 'outfits_v1.json'

# Load outfits
# with open(OUTFITS_PATH, "r") as f:
#     outfits = [Outfit.from_dict(d) for d in json.load(f)]
# outfits = outfits[:8]

# CLIP wrapper for LangChain
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class CLIPImageEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ensure each string is within the token limit by truncating
        max_tokens = 77  # CLIP's limit
        truncated_texts = [t[:300] for t in texts]  # 300 chars is usually safe
        inputs = clip_processor(text=truncated_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
        return outputs.tolist()

    def embed_query(self, text: str) -> List[float]:
        max_tokens = 77
        text = text[:300]
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
        return outputs[0].tolist()

def build_query(outfits: List[Outfit], weather: str, occasion: str) -> tuple[list[str], str]:
    descriptions = [f"index {i}: {o.description}" for i, o in enumerate(outfits)]
    query = f"""
            Given the following outfit descriptions:

            {json.dumps(descriptions, indent=2)}

            Please return a list of outfit indices ranked by their suitability for the weather: "{weather}" and the occasion: "{occasion}".

            Respond in this exact format:
            {{
            "ranking": [index_1, index_2, ...]
            }}
            """
    return descriptions, query


def extract_ranking(response: str) -> List[int]:
    try:
        # Strip markdown backticks if present
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?|```$", "", response, flags=re.IGNORECASE | re.MULTILINE).strip()
        
        response_json = json.loads(response)
        return response_json["ranking"]
    except Exception as e:
        print("Failed to parse response:", response)
        raise e

class CLIPTextAdapter(Embeddings):
    """Converts text queries to 3072-dim pseudo-embeddings compatible with your outfit vectors"""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ensure each string is within the token limit by truncating
        max_tokens = 77  # CLIP's limit
        truncated_texts = [t[:300] for t in texts]  # 300 chars is usually safe
        inputs = clip_processor(text=truncated_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
        return outputs.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        # max_tokens = 77
        # text = text[:300]
        # inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        # with torch.no_grad():
        #     outputs = clip_model.get_text_features(**inputs)
        # return outputs[0].tolist()

        with torch.no_grad():
            # Get CLIP text embedding (512-dim)
            text_tokens = clip.tokenize([text]).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)[0].cpu().numpy()
        
        # Repeat to 3072-dim (512 * 6 = 3072)
        expanded_embedding = np.tile(text_embedding, 6)
        return expanded_embedding.tolist()

# -----------------------------
# Build Vector Store from Precomputed Embeddings
# -----------------------------

def create_vector_store(outfits: list[Outfit]):
    vectors = []
    documents = []
    # clip_embedder = CLIPImageEmbeddings()


    for outfit in outfits:
        if outfit.embedding is not None:
            # Flatten the embedding from (3, 1024) to (3072,) and append it
            flattened_embedding = outfit.embedding.astype("float32").flatten()
            vectors.append(flattened_embedding)
            documents.append(Document(
                page_content=outfit.description,
                metadata={
                    "outfit_id": outfit.outfit_id,
                    "score": outfit.score,
                    "outfit_description": outfit.description
                }
            ))

    # Create FAISS index manually
    embedding_dim = vectors[0].shape[0]  # Get the size of each flattened vector
    print("Embedding shape", vectors[0].shape)
    
    # Convert vectors to a 2D numpy array (shape: (total_vectors, embedding_dim))
    vectors = np.array(vectors, dtype=np.float32)
    
    # Create the FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(vectors)  # Add vectors to the index

    # Wrap in LangChain FAISS object
    vector_store = FAISS(
        index=index,
        docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)}),
        index_to_docstore_id={i: str(i) for i in range(len(documents))},
        embedding_function=CLIPTextAdapter()  # Not needed since we already have embeddings
    )

    return vector_store

def re_rank_outfits(
        outfits: List[Outfit], 
        weather="Rainy", 
        occasion="Work Meeting"
        ) -> List[Outfit]:

    print(__file__, weather, len(weather), occasion, len(occasion))

    descriptions, query = build_query(outfits, weather, occasion)

    clip_embedder = CLIPImageEmbeddings()

    print("QUERY =", query)

    #
    vector_store = FAISS.from_texts(descriptions, clip_embedder)
    #
    # vector_store = create_vector_store(outfits)
    retriever = vector_store.as_retriever()

    # llm_prompt = "Which outfit should i wear on a sunny day where I have to walk around a lot?"
    # response = openai.ChatCompletion.create(
    # model="gpt-4",
    # messages=[{"role": "user", "content": llm_prompt}],
    # temperature=0
    # )   


    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = qa_chain.invoke({"query": query})
    print("RESULT = ", result)
    ranked_indices = extract_ranking(result['result'])

    # print(ranked)

    return [outfits[i] for i in ranked_indices]

# def rerank_from_ui(
#         outfits: list[Outfit], 
#         weather: str, 
#         occasion: str) -> list[int]:

#     return re_rank_outfits(outfits, weather, occasion)

# Run
# ranked = re_rank_outfits(outfits)
# for i, o in enumerate(ranked):
#     print(f"{i}: {o.description}")


