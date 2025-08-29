import PyPDF2
import os

def parse_pdf_resume(file_path):
    """
    Parses a PDF file to extract all text content.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at {file_path}")

    full_text = ""
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                full_text += page.extract_text() or ""
    except Exception as e:
        print(f"❌ An error occurred while reading the PDF: {e}")
        return ""
    
    # You might want to perform some basic cleaning on the text
    # This is a simple example; more advanced cleaning can be added later
    cleaned_text = " ".join(full_text.split())
    return cleaned_text

if __name__ == "__main__":
    # This block is for testing the parser on its own
    # You'll need to create a dummy PDF file named 'resume.pdf' in the backend folder for this to work
    sample_file_path = "resume.pdf"
    print(f"Attempting to parse {sample_file_path}...")
    text = parse_pdf_resume(sample_file_path)
    if text:
        print("\n✅ PDF parsed successfully! Here is the extracted text:")
        print("---" * 20)
        print(text[:500] + "...") # Print the first 500 characters for a preview
        print("---" * 20)
    else:
        print("\n❌ Parsing failed or no text was found.")