
import subprocess
import os
import time
from pathlib import Path

def main():
    root_dir = Path(__file__).resolve().parent.parent
    embeddings_path = root_dir / 'models' / 'candidate_embeddings.npy'
    
    print("Waiting for embeddings...")
    while not os.path.exists(embeddings_path):
        time.sleep(30)
        print(".", end="", flush=True)
    
    # Wait a bit more for the file to be fully written
    time.sleep(10)
    
    print("\nEmbeddings ready! Running ranker...")
    start = time.time()
    rank_script = root_dir / 'src' / 'rank.py'
    subprocess.run(["python", str(rank_script)], check=True)
    end = time.time()
    print(f"Ranking took {end - start:.2f} seconds")
    
    print("Validating submission...")
    validate_script = root_dir / 'src' / 'validate_submission.py'
    submission_path = root_dir / 'outputs' / 'submission.csv'
    subprocess.run(["python", str(validate_script), str(submission_path)], check=True)
    
    print("All done!")

if __name__ == "__main__":
    main()
