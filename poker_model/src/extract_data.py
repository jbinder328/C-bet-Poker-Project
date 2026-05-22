import zipfile
import glob
import shutil
import os

def extract_cash_game_files(zip_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        # Only extract files from the 1:2 Cash Game folder (note: colon, not slash)
        cash_files = [
            f for f in z.namelist()
            if "1:2 Cash Game" in f and f.endswith(".txt")
        ]

        print(f"Found {len(cash_files)} cash game txt files to extract...")

        for f in cash_files:
            z.extract(f, "temp_extract/")

    # Move all txt files to flat output directory
    txt_files = glob.glob("temp_extract/**/*.txt", recursive=True)
    for f in txt_files:
        shutil.copy(f, output_dir)

    # Clean up temp
    shutil.rmtree("temp_extract/", ignore_errors=True)

    print(f"Extracted {len(txt_files)} cash game files to {output_dir}")
    return txt_files

if __name__ == "__main__":
    zip_path = "/Users/joshbinder/Downloads/Poker Cash and Tourney.zip"
    output_dir = "/Users/joshbinder/EBK Project/poker_model/data/raw/"

    if not os.path.exists(zip_path):
        print(f"Error: Zip file not found at {zip_path}")
        exit(1)

    files = extract_cash_game_files(zip_path, output_dir)
    print(f"\nSuccess! Extracted {len(files)} files.")
