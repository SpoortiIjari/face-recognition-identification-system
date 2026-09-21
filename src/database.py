import json
import os
import numpy as np


DATABASE_PATH = "database/faces.json"


def load_database():
    """
    Load all enrolled face embeddings from the JSON database.
    """

    if not os.path.exists(DATABASE_PATH):
        return {}

    with open(DATABASE_PATH, "r") as file:
        data = json.load(file)

    return data


def save_embedding(name, embedding):
    """
    Save a person's face embedding to the database.
    """

    data = load_database()

    # Convert NumPy array into a normal Python list
    embedding_list = embedding.flatten().tolist()

    # Create person entry if they don't exist
    if name not in data:
        data[name] = []

    # Add the new embedding
    data[name].append(embedding_list)

    # Save updated database
    with open(DATABASE_PATH, "w") as file:
        json.dump(data, file, indent=4)

    print(f"✓ Embedding saved for {name}")
