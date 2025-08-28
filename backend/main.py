from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from .agents.data_quality_agent import DataQualityAgent

app = FastAPI(title="Data Quality Rule Recommendation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the data quality agent
agent = DataQualityAgent()

# Include routers
app.include_router(router)

# Make agent available to routes
app.state.agent = agent

# To run: uvicorn backend.main:app --reload
