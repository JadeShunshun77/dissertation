import pandas as pd
from dune_client.client import DuneClient
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DUNE_API_KEY") 
QUERY_ID = 7792182 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_filename = os.path.join(BASE_DIR, "raw_edges_new.csv")

if not API_KEY:
    raise ValueError("DUNE_API_KEY environment variable is not set. Please set it in your .env file or system environment variables.")

print("Starting the download of edges from Dune Analytics...")

try:
    dune = DuneClient(API_KEY)
    df = dune.get_latest_result_dataframe(QUERY_ID)

    df.to_csv(output_filename, index=False)
    print(f"Download completed successfully! A total of {len(df)} rows of data have been saved to {output_filename}.")

except Exception as e:
    print(f"Error occurred while running the script. Detailed information: {e}")