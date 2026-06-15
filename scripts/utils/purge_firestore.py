from google.cloud import firestore

def purge_firestore():
    db = firestore.Client(project='healthcare-fl-diagnostics')
    rounds = db.collection('rounds').stream()
    count = 0
    for r in rounds:
        print(f"Purging Round {r.id}...")
        # Delete uploads
        shards = r.reference.collection('shards').stream()
        for s in shards:
            uploads = s.reference.collection('uploads').stream()
            for u in uploads:
                u.reference.delete()
            s.reference.delete()
        r.reference.delete()
        count += 1
    print(f"Successfully purged {count} rounds from Firestore.")

if __name__ == "__main__":
    purge_firestore()
