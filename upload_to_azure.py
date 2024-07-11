#upload_to_azure.py

import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Connection string and container name
connect_str = ""
container_name = "stockdata"

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Get a reference to the container
container_client = blob_service_client.get_container_client(container_name)

# Path to your local file
local_file_name = "stock_data_365days.csv"  # Updated to new file name
blob_name = "stock_data_365days.csv"  # Updated blob name

# Create a blob client using the local file name as the name for the blob
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# Upload the created file
try:
    with open(local_file_name, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {local_file_name} to Azure Blob Storage")
except FileNotFoundError:
    print(f"Error: The file {local_file_name} was not found. Please make sure it exists in the same directory as this script.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
