import re
from pathlib import Path 

# 1. Define Constants
TIMESTAMP_REGEX = re.compile(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}')
CURRENT_FOLDER  = Path("Transcript")
OUTPUT_FOLDER   = Path("Clean_Transcript")

def clean_transcript(transcript:str) -> str:
    lines = transcript.splitlines()
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue 
        if TIMESTAMP_REGEX.match(stripped):
            continue 
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)



def load_cleaned_transcript(original_dir : Path | str,
                            new_dir : Path | str):
  
    # Create the output folder safely
    new_dir.mkdir(parents=True, exist_ok=True)

    # Check if input folder exists to avoid crashing
    if not original_dir.exists():
        print(f"Error: The input directory '{original_dir}' does not exist.")
        return

    # Loop through every file in the input directory
    # Use '*.vtt' if you only want VTT files, or '*' for all files
    for file_path in original_dir.glob(pattern="*.txt"):
        
        # Read the raw file content
        raw_text = file_path.read_text(encoding="utf-8")
        
        # Strip out timestamps
        final_text = clean_transcript(raw_text)

        # save to the new folder with the SAME name
        output_file_path = new_dir / file_path.name
        output_file_path.write_text(final_text, encoding="utf-8")
        
        print(f"Processed: {file_path.name} -> {output_file_path.name}")

# Run the automation script
if __name__ == "__main__":
    load_cleaned_transcript(original_dir=CURRENT_FOLDER,
                            new_dir=OUTPUT_FOLDER)
