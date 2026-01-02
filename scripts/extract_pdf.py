import sys
import os

try:
    import pypdf
    print("Using pypdf")
    def extract_text(path):
        reader = pypdf.PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
except ImportError:
    try:
        import PyPDF2
        print("Using PyPDF2")
        def extract_text(path):
            reader = PyPDF2.PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except ImportError:
        print("No PDF library found (pypdf or PyPDF2).")
        sys.exit(1)

def main():
    pdf_path = r"d:\code\MPI\docs\about\2512.24880v1.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        text = extract_text(pdf_path)
        # Save to a text file for reading
        with open("pdf_content.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Text extracted to pdf_content.txt")
        # Print first 2000 chars to stdout
        print(text[:2000])
    except Exception as e:
        print(f"Error extracting text: {e}")

if __name__ == "__main__":
    main()
