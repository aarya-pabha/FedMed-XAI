from google.cloud import firestore

def force_purge_firestore():
    db = firestore.Client(project='healthcare-fl-diagnostics')
    
    # Force delete all documents in 'rounds' and its subcollections
    print("🔥 Starting Force Purge of Firestore 'rounds' collection...")
    rounds_ref = db.collection('rounds')
    rounds = rounds_ref.stream()
    
    count = 0
    for r in rounds:
        # Delete subcollections first
        shards = r.reference.collection('shards').stream()
        for s in shards:
            uploads = s.reference.collection('uploads').stream()
            for u in uploads:
                u.reference.delete()
            s.reference.delete()
        r.reference.delete()
        count += 1
        print(f"  -> Deleted Round {r.id}")
        
    print(f"✅ Force Purge Complete. Removed {count} round documents.")

if __name__ == "__main__":
    force_purge_firestore()
