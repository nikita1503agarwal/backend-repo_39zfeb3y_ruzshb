"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Example schemas (retain for reference)
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# App-specific schemas
class ChatMessage(BaseModel):
    session_id: str = Field(..., description="Chat session identifier")
    role: str = Field(..., pattern="^(user|assistant|system)$", description="Who sent the message")
    content: str = Field(..., description="Message content")
    language: str = Field("auto", description="ISO language code, e.g., en, es, fr, de, it, hi")

class ChatSession(BaseModel):
    session_id: str
    user_agent: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class Diagnosis(BaseModel):
    session_id: str = Field(..., description="Diagnosis session identifier")
    plant_type: Optional[str] = Field(None, description="Type of plant if known")
    predicted_disease: str = Field(..., description="Predicted disease label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    recommendations: List[str] = Field(default_factory=list, description="Actionable next steps")
    source: str = Field(..., description="'huggingface', 'openai_vision', or 'heuristic'")
