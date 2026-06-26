
import zipfile
import xml.etree.ElementTree as ET
import os

def get_docx_text(path):
    """
    Take the path of a docx file as argument, return the text in unicode.
    """
    document = zipfile.ZipFile(path)
    xml_content = document.read('word/document.xml')
    document.close()
    tree = ET.fromstring(xml_content)
    
    # Namespaces
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    paragraphs = []
    for paragraph in tree.findall('.//w:p', ns):
        texts = [node.text for node in paragraph.findall('.//w:t', ns) if node.text]
        if texts:
            paragraphs.append("".join(texts))
    
    return "\n".join(paragraphs)

from pathlib import Path

# Resolve base paths relative to the project root directory (parent of src/)
ROOT_DIR = Path(__file__).resolve().parent.parent
docs_dir = ROOT_DIR / 'docs'

files = [
    'job_description.docx',
    'redrob_signals_doc.docx',
    'submission_spec.docx',
    'README.docx'
]

for f in files:
    full_path = docs_dir / f
    print(f"--- {f} ---")
    try:
        text = get_docx_text(full_path)
        print(text)
    except Exception as e:
        print(f"Error reading {f}: {e}")
    print("\n" + "="*50 + "\n")
