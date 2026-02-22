from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import instructor
import asyncpg
import httpx
import os

from api.datamodels import RecipeResponse, AskRequest

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db_pool
    client = instructor.from_provider(
        "ollama/phi3:mini",
        async_client=True
    )
    # Database pool
    db_pool = await asyncpg.create_pool(
        os.getenv("DATABASE_URL")
    )
    
    yield
    
    # Cleanup on shutdown
    await db_pool.close()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Hello Oslo!"}

async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{os.getenv('OLLAMA_URL')}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            },
            timeout=120
        )
    return response.json()["embedding"]

@app.post("/recipes/embed")
async def embed_recipes():
    async with db_pool.acquire() as conn:
        recipes = await conn.fetch("SELECT id, name, ingredients, cuisine FROM recipes WHERE embedding IS NULL")
        
        for recipe in recipes:
            text = f"{recipe['name']}. Ingredients: {', '.join(recipe['ingredients'])}. Cuisine: {recipe['cuisine']}"
            embedding = await get_embedding(text)
            
            await conn.execute(
                "UPDATE recipes SET embedding = $1 WHERE id = $2",
                str(embedding), recipe['id']
            )
    
    return {"message": f"Embedded {len(recipes)} recipes"}

async def find_similar_recipes(question: str, limit: int = 2) -> list:
    query_embedding = await get_embedding(question)
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT name, ingredients, steps, cuisine
            FROM recipes
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1
            LIMIT $2
        """, str(query_embedding), limit)
    
    return [dict(row) for row in rows]

@app.post("/ask", response_model=RecipeResponse)
async def ask(request: AskRequest) -> RecipeResponse:
    example = RecipeResponse(
        dish="Omelette",
        ingredients=["3 eggs", "butter", "salt"],
        steps=["Beat eggs", "Heat pan", "Cook until set"]
    )
    similar_recipes = await find_similar_recipes(request.question)
    context = "\n".join([
        f"Recipe: {r['name']}\nIngredients: {', '.join(r['ingredients'])}\nSteps: {', '.join(r['steps'])}\nCuisine: {r['cuisine']}"
        for r in similar_recipes
    ])
    
    return await client.create(
        response_model=RecipeResponse,
        max_retries=3,
        messages=[
            {
                "role": "system",
                "content": f"""You are a recipe assistant.
                Answer using ONLY the recipes provided below. 
                Do not invent ingredients or steps that aren't listed.
                Always respond in this exact JSON format:
                {example.model_dump()}
                
                Available recipes:
                {context}"""
            },
            {"role": "user", "content": request.question}
        ]
    )

@app.get("/recipes")
async def get_recipes():
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, name, cuisine FROM recipes")
        return [dict(row) for row in rows]