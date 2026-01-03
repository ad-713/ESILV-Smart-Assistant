import json
import os
import logging
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LEADS_FILE = os.path.join("data", "leads.json")
file_lock = threading.Lock()

def ensure_data_dir():
    """Ensure the data directory exists."""
    os.makedirs(os.path.dirname(LEADS_FILE), exist_ok=True)

def load_leads() -> List[Dict]:
    """Load leads from the JSON file."""
    if not os.path.exists(LEADS_FILE):
        return []
    
    try:
        with open(LEADS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {LEADS_FILE}. Returning empty list.")
        return []

def save_lead(name: str, email: str, topic: Optional[str] = None) -> str:
    """
    Save a new lead to the JSON file.
    
    Args:
        name: Name of the interested user.
        email: Email of the interested user.
        topic: Context or topic of interest (optional).
        
    Returns:
        Status message.
    """
    logger.info(f"Attempting to save lead: name='{name}', email='{email}', topic='{topic}'")
    ensure_data_dir()
    
    with file_lock:
        leads = load_leads()
        
        new_lead = {
            "name": name,
            "email": email,
            "topic": topic or "General Inquiry",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        leads.append(new_lead)
        
        try:
            with open(LEADS_FILE, "w", encoding="utf-8") as f:
                json.dump(leads, f, indent=4, ensure_ascii=False)
            logger.info("Lead saved successfully.")
            return "Lead saved successfully."
        except Exception as e:
            logger.error(f"Error saving lead to file: {str(e)}", exc_info=True)
            return f"Error saving lead: {str(e)}"

def get_leads_dataframe():
    """Returns leads as a pandas DataFrame for easy display."""
    import pandas as pd
    leads = load_leads()
    if not leads:
        return pd.DataFrame(columns=["Timestamp", "Name", "Email", "Topic"])
    
    df = pd.DataFrame(leads)
    # Reorder columns if keys exist, handle missing keys gracefully
    columns = ["timestamp", "name", "email", "topic"]
    # Filter to only existing columns in case schema changes
    available_cols = [c for c in columns if c in df.columns]
    df = df[available_cols]
    
    # Rename for display
    df.rename(columns={
        "timestamp": "Timestamp", 
        "name": "Name", 
        "email": "Email", 
        "topic": "Topic"
    }, inplace=True)
    
    return df

def clear_leads() -> bool:
    """
    Clears all leads data.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info("Attempting to clear all leads.")
    ensure_data_dir()
    
    with file_lock:
        try:
            # Overwrite with empty list
            with open(LEADS_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4, ensure_ascii=False)
            logger.info("All leads cleared successfully.")
            return True
        except Exception as e:
            logger.error(f"Error clearing leads: {str(e)}", exc_info=True)
            return False
