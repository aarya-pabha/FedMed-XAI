from google.cloud import firestore

def inspect_firestore():
    db = firestore.Client(project='healthcare-fl-diagnostics')
    
    print("--- FIRESTORE INSPECTION ---")
    
    # Check specifically for Round 1, Shard 0
    ref = db.collection('rounds').document('1').collection('shards').document('0')
    doc = ref.get()
    
    if doc.exists:
        print(f"✅ round_1/shard_0 found.")
        print(f"   Status: {doc.get('status')}")
        
        # Check uploads
        uploads = list(ref.collection('uploads').stream())
        print(f"   Uploads count: {len(uploads)}")
        for u in uploads:
            print(f"     -> Client {u.id}")
    else:
        print("❌ round_1/shard_0 NOT found.")
        
    # List all rounds
    rounds = list(db.collection('rounds').stream())
    print(f"\nTotal Rounds in root: {len(rounds)}")
    for r in rounds:
        print(f"  - Round {r.id}")

if __name__ == "__main__":
    inspect_firestore()
