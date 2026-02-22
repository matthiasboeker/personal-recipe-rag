from pydantic import BaseModel
from typing import List

class AskRequest(BaseModel):
    question: str

class RecipeResponse(BaseModel):
    dish: str
    ingredients: List[str]
    steps: List[str]
    
    def to_embedding_text(self) -> str:
        return f"{self.dish}. Ingredients: {', '.join(self.ingredients)}. Steps: {', '.join(self.steps)}"