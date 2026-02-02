import os
from pipeline import run_pipeline

def main():
    folder_path = r"your_link_to_folder"

    df = run_pipeline(folder_path)
    df['label'] = 0 # if adls dataset and = 1 if fall dataset
  
    # Data-Storage
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("your_link_to_save.csv", index=False)

    print(" Pipeline finished.")

if __name__ == "__main__":
    main()
