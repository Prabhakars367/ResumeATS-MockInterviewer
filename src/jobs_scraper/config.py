import os
from dotenv import load_dotenv

# Load variables from .env if it exists
load_dotenv()

class Config:
    # Use absolute path for SQLite
    _default_db = "sqlite:///jobs.db"
    if os.name != 'nt': # If not Windows (likely Render/Linux)
        _default_db = "sqlite:////opt/render/project/src/jobs.db"
    
    DATABASE_URL = os.getenv("JOBS_DATABASE_URL", _default_db)
    # Note: We use JOBS_DATABASE_URL to avoid conflict with potential other DBs

settings = Config()
