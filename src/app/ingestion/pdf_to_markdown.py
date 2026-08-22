
import os  
import pymupdf4llm
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv()

def PDFToMarkdownConverter(input_dir : str, output_dir : str) -> list[Path]:
    """
    Convert all the .pdf files in this directory to markdown format
   
    Args :
    input_dir : Directory that contains input pdf files
    output_dir : Directory that hold converted markdown files

    Return :
    List of generated mamrkdown file paths

   """     
   
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # All PDF files
    pdf_files = [input_path] if input_path.is_file() else list(input_path.glob("*.pdf"))


    if not pdf_files:
         raise FileNotFoundError(f"PDF files not found : {input_path}")

    # Create output path if not created already
    output_path.mkdir(exist_ok = True, parents = True)

    markdown_files = []

    # For each .pdf file in input folder
    for file in pdf_files:

        # Convert each file to markdown and it takes string as input so str(file)
        markdown_content = pymupdf4llm.to_markdown(str(file))

        # Here .stem will remove last extension of pdf file like msft.pdf will become msft and then add .md to it
        markdown_file = output_path / f"{file.stem}.md"

        # Write the md content to output folder as text
        markdown_file.write_text(
                                data = markdown_content,
                                encoding = "utf-8"
                                )
        markdown_files.append(markdown_file)
        
   

if __name__ == "__main__":

    # Root Directory Name
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()

    # Input Dir and Output Dir paths
    input_dir  = ROOT_DIR / "data" / "raw"
    output_dir = ROOT_DIR / "data" / "processed"

    PDFToMarkdownConverter(str(input_dir), str(output_dir))

 
