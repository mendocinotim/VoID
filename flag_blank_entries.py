from src.voiceprint.database import VoiceprintDatabase
import os

def flag_blank_entries():
    """Flag all blank speaker entries as ignored in the database."""
    db = VoiceprintDatabase()
    
    # Get all blank speaker entries
    if '' in db.metadata:
        # Get all indices for blank speaker
        indices = list(range(len(db.metadata[''])))
        
        # Mark them as ignored
        db.mark_as_ignored('', indices)
        
        print(f"Flagged {len(indices)} blank speaker entries as ignored")
        
        # Verify the flagging
        voiceprints = db.get_voiceprints(include_ignored=True)
        blank_count = len(voiceprints.get('', []))
        print(f"Total blank entries (including ignored): {blank_count}")
        
        voiceprints = db.get_voiceprints(include_ignored=False)
        blank_count = len(voiceprints.get('', []))
        print(f"Active blank entries (excluding ignored): {blank_count}")
    else:
        print("No blank speaker entries found")

if __name__ == "__main__":
    flag_blank_entries() 