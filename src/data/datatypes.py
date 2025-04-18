from typing import List, Optional, TypedDict, Union
from PIL import Image
import copy
from pydantic import BaseModel, Field
import numpy as np
import io
import base64


class FashionItem(BaseModel):
    item_id: Optional[int] = Field(
        default=None,
        description="Unique ID of the item, mapped to `id` in the ItemLoader"
    )
    category: Optional[str] = Field(
        default="",
        description="Category of the item"
    )
    image: Optional[Image.Image] = Field(
        default=None,
        description="Image of the item"
    )
    description: Optional[str] = Field(
        default="",
        description="Description of the item"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata for the item"
    )
    embedding: Optional[np.ndarray] = Field(
        default=None,
        description="Embedding of the item"
    )

    def to_dict(self):

        data = {
            "item_id": self.item_id,
            "category": self.category,
            "description": self.description,
            "metadata": self.metadata
        }

        # Handle image (store as base64 string or path reference)
        if self.image:
            buffered = io.BytesIO()
            self.image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data["image"] = img_str

        # Handle embedding (convert to list)
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        
        return data
    
    @classmethod
    def from_dict(cls, data: dict):
        # Handle image (base64 back to PIL)
        img_data = data.get("image")
        image = None
        if img_data:
            try:
                image = Image.open(io.BytesIO(base64.b64decode(img_data)))
            except Exception as e:
                print(f"Error decoding image: {e}")

        # Handle embedding
        embedding = data.get("embedding")
        if embedding is not None:
            embedding = np.array(embedding)

        return cls(
            item_id=data.get("item_id"),
            category=data.get("category"),
            image=image,
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            embedding=embedding
        )

    class Config:
        arbitrary_types_allowed = True

class Outfit(BaseModel):
    outfit_id: str = Field(default=None, description="Unique identifier for the outfit")
    description: str = Field(default="", description="Description or name of the outfit")
    score: float = Field(default=None, description="A score or rating associated with the outfit")
    fashion_items: List[FashionItem] = Field(default_factory=list, description="List of fashion items in the outfit")

    def __init__(self, fashion_items: List[FashionItem], score: float = None, outfit_id: str = None, description: str = None):
        # Generate outfit_id by joining item_ids
        if not outfit_id:
            item_ids = [str(item.item_id) for item in fashion_items if item.item_id is not None]
            outfit_id = "_".join(item_ids) if item_ids else "outfit_undefined"

        # Generate description from item categories or names
        if not description:
            description = ", ".join([item.category + ":" + item.description or "?" for item in fashion_items])  # or use item.name if you prefer

        # Call BaseModel's constructor
        super().__init__(
            outfit_id=outfit_id,
            description=description,
            score=score,
            fashion_items=fashion_items
        )

    def to_dict(self):
        return {
            "outfit_id": self.outfit_id,
            "description": self.description,
            "score": self.score,
            "fashion_items": [item.to_dict() for item in self.fashion_items]
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            outfit_id=data.get("outfit_id"),
            description=data.get("description", ""),
            score=data.get("score"),
            fashion_items=[FashionItem.from_dict(d) for d in data.get("fashion_items", [])]
        )

    
class FashionCompatibilityQuery(BaseModel):
    outfit: List[Union[FashionItem, int]] = Field(
        default_factory=list,
        description="List of fashion items"
    )

class FashionComplementaryQuery(BaseModel):
    outfit: List[Union[FashionItem, int]] = Field(
        default_factory=list,
        description="List of fashion items"
    )
    category: str = Field(
        default="",
        description="Category of the target outfit"
    )
    
    
class FashionCompatibilityData(TypedDict):
    label: Union[
        int, 
        List[int]
    ]
    query: Union[
        FashionCompatibilityQuery, 
        List[FashionCompatibilityQuery]
    ]
    
    
class FashionFillInTheBlankData(TypedDict):
    query: Union[
        FashionComplementaryQuery,
        List[FashionComplementaryQuery]
    ]
    label: Union[
        int,
        List[int]
    ]
    candidates: Union[
        List[FashionItem],
        List[List[FashionItem]]
    ]
    
    
class FashionTripletData(TypedDict):
    query: Union[
        FashionComplementaryQuery,
        List[FashionComplementaryQuery]
    ]
    answer: Union[
        FashionItem,
        List[FashionItem]
    ]