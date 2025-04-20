from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch
import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import pathlib
import json

from .data.datasets import polyvore
from .data.datatypes import Outfit


SRC_DIR = pathlib.Path(__file__).parent.parent.absolute()
OUTFITS_PATH = SRC_DIR / 'datasets' / 'outfits_v1.json'

load_dotenv() 

# change when we have database
# image_folder = 'clothingimages'
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# # creates a caption/description of the image with BLIP
# def caption_image(image_path):
#     image = Image.open(image_path).convert('RGB')
#     inputs = processor(image, return_tensors="pt")
#     out = model.generate(**inputs, max_new_tokens=50)
#     caption = processor.decode(out[0], skip_special_tokens=True)
#     # here we could possibly add fashion score to each clothing item, not sure how well it'll work
#     # caption = caption + "Fashion Score: " + score
#     return caption

with open(OUTFITS_PATH, "r") as f:
    data = json.load(f)
    outfits = [Outfit.from_dict(d) for d in data]

outfits = outfits[:8]

# captions = []
# metadata = []
# # filling data structures
# for filename in os.listdir(image_folder):
#     if filename.lower().endswith((".jpg", ".png", ".jpeg")):
#         path = os.path.join(image_folder, filename)
#         cap = caption_image(path)
#         captions.append(cap)
#         # or add score to the metadata and use it to rank
#         metadata.append({"filename": filename})

# print(captions)



# load CLIP for embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

captions = []
metadata = []
for outfit in outfits:
    captions.append(outfit.description)

# CLIP embedding wrapper
class CLIPImageEmbeddings(Embeddings):
    def embed_documents(self, texts):
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
        return outputs.tolist()

    def embed_query(self, text):
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
        return outputs[0].tolist()

clip_embedder = CLIPImageEmbeddings()
# creating embeddings from captions
clip_vectors = clip_embedder.embed_documents(captions)
vector_store = FAISS.from_texts(captions, clip_embedder) #, metadatas=metadata)
retriever = vector_store.as_retriever()

# dev_key = "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"

qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI, retriever=retriever)

def get_recommendation(query="What outfit should I wear?"):
    # prompt can be changed to add weather/schedule data automatically. Ask for a full outfit. Ask for higher scored items.
    answer = qa_chain.invoke({"query": query})
    print(answer)
    return answer

get_recommendation("interpret this data please.")
# client = OpenAI(
#   api_key=dev_key
# )

# completion = client.chat.completions.create(
#   model="gpt-4o-mini",
#   store=True,
#   messages=[
#     {"role": "user", "content": "write a haiku about ai"}
#   ]
# )

# print(completion.choices[0].message);

