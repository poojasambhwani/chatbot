import os
from pymongo import MongoClient

def get_mongo_client() -> MongoClient:
    """
    Returns a MongoClient connected to MongoDB Atlas using the URI from the MONGODB_ATLAS_URI environment variable.
    """
    uri = os.getenv("MONGODB_ATLAS_URI")
    if not uri:
        raise ValueError(
            "MONGODB_ATLAS_URI environment variable not set. "
            "Please set it to your MongoDB Atlas connection string."
        )
    return MongoClient(uri)
