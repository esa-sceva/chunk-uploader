#!/usr/bin/env python3
"""
Script to recreate Qdrant collection with correct dimensions.
"""

from qdrant_client import QdrantClient, models
import yaml

def recreate_collection():
    """Recreate the Qdrant collection with correct dimensions."""
    
    # Load config
    with open("config_qwen.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Connection details
    qdrant_url = "https://your-qdrant-instance.cloud.qdrant.io:6333"
    qdrant_api_key = "your-qdrant-api-key-here"
    collection_name = config["database"]["collection_name"]
    vector_size = config["upload_params"]["vector_size"]
    
    print(f"Recreating collection: {collection_name}")
    print(f"Vector dimensions: {vector_size}")
    
    # Initialize client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    try:
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name)
            print(f"Current collection info:")
            print(f"   Vectors count: {collection_info.vectors_count}")
            print(f"   Vector size: {collection_info.config.params.vectors.size}")
            
            # Ask for confirmation
            response = input(f"\nCollection '{collection_name}' exists with {collection_info.vectors_count} vectors. Delete and recreate? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted by user")
                return False
            
            # Delete existing collection
            print(f"Deleting existing collection...")
            client.delete_collection(collection_name)
            print(f"Collection deleted")
            
        except Exception as e:
            print(f"Collection doesn't exist or error checking: {e}")
        
        # Create new collection
        print(f"Creating new collection...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
        
        # Verify creation
        new_collection_info = client.get_collection(collection_name)
        print(f"Collection created successfully!")
        print(f"   Name: {collection_name}")
        print(f"   Vector size: {new_collection_info.config.params.vectors.size}")
        print(f"   Distance metric: {new_collection_info.config.params.vectors.distance}")
        
        return True
        
    except Exception as e:
        print(f"Error recreating collection: {e}")
        return False

if __name__ == "__main__":
    print("Qdrant Collection Recreator")
    print("=" * 40)
    
    success = recreate_collection()
    
    if success:
        print("\nCollection recreated successfully!")
        print("You can now run your upload script.")
    else:
        print("\nFailed to recreate collection.")
        print("Check your Qdrant credentials and network connectivity.")