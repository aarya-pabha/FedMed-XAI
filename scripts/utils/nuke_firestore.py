import argparse
import sys
from google.cloud import firestore

def nuke_rounds(force=False):
    if not force:
        print("[ERROR] Must use --force to nuke Firestore.")
        sys.exit(1)
        
    db = firestore.Client(project='healthcare-fl-diagnostics')
    print("Nuking Rounds 1-20 (including virtual docs)...")
    
    for r_id in range(1, 21):
        r_ref = db.collection('rounds').document(str(r_id))
        
        # Delete sub-shards
        shards = r_ref.collection('shards').stream()
        for s in shards:
            # Delete uploads
            uploads = s.reference.collection('uploads').stream()
            for u in uploads:
                u.reference.delete()
            s.reference.delete()
            print(f"  -> Deleted Shard {s.id} for Round {r_id}")
            
        r_ref.delete()
        
    print("Nuke Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Destructive Firestore Nuke")
    parser.add_argument("--force", action="store_true", help="Force the nuke operation")
    args = parser.parse_args()
    nuke_rounds(args.force)
