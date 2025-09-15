from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Optional

# Create FastAPI instance
app = FastAPI(title="User Management API", version="1.0.0")


# Your corrected User model
class User(BaseModel):
    name: str
    age: int
    email: EmailStr


# In-memory storage
users_db = []


# Root endpoint
@app.get("/")
async def root():
    return {"message": "User Management API", "version": "1.0.0"}


# Create a new user
@app.post("/users/", response_model=User)
async def create_user(user: User):
    # Check if email already exists
    if any(u.email == user.email for u in users_db):
        raise HTTPException(status_code=400, detail="Email already registered")

    users_db.append(user)
    return user


# Get all users
@app.get("/users/", response_model=List[User])
async def get_users():
    return users_db


# Get user by email
@app.get("/users/{email}", response_model=User)
async def get_user_by_email(email: str):
    user = next((u for u in users_db if u.email == email), None)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# Update user by email
@app.put("/users/{email}", response_model=User)
async def update_user(email: str, user_update: User):
    for i, user in enumerate(users_db):
        if user.email == email:
            # Update the user
            users_db[i] = user_update
            return user_update
    raise HTTPException(status_code=404, detail="User not found")


# Delete user by email
@app.delete("/users/{email}")
async def delete_user(email: str):
    global users_db
    initial_length = len(users_db)
    users_db = [u for u in users_db if u.email != email]

    if len(users_db) == initial_length:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User deleted successfully"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "user_count": len(users_db)}