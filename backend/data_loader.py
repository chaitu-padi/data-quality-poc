import json
import pandas as pd
from ydata_profiling import ProfileReport

def load_csv(filepath):
    return pd.read_csv(filepath)

def load_json_file(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def generate_profile(df, title):
    return ProfileReport(df, title=title, minimal=True)
