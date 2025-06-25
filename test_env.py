# test_env.py
from dotenv import load_dotenv
import os
load_dotenv()
print("OPENAI_API_KEY:", "Set" if os.environ.get(
    "OPENAI_API_KEY") else "Not set")
