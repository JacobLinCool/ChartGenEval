"""Check the assembled submission. Requires PyMuPDF and pdffonts."""
from pathlib import Path
import re
import subprocess

import fitz

HERE = Path(__file__).resolve().parent

def main():
    pdf = HERE / "paper.pdf"
    doc = fitz.open(pdf)
    assert len(doc) == 5, f"Expected 4 technical pages + 1 reference page, got {len(doc)}"
    assert not doc.is_encrypted
    assert pdf.stat().st_size < 5_000_000
    captions = []
    for number, page in enumerate(doc, 1):
        assert tuple(round(x) for x in page.rect) == (0, 0, 612, 792)
        text = page.get_text()
        assert "??" not in text
        found = re.findall(r"^(?:Fig\. \d+\.|Table \d+\.)", text, re.M)
        assert len(found) <= 1, (number, found)
        captions.append(found)
        # Allow glyph ascenders/descenders around the official TeX text box.
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    assert x0 >= 53 and x1 <= 561 and y0 >= 70 and y1 <= 725, span
                    # TeX points convert to PDF points by 72 / 72.27.
                    # Mathematical subscripts are naturally smaller.
                    if not span["font"].startswith("CM"):
                        assert span["size"] >= 8.96, span
    assert captions == [["Fig. 1."], ["Fig. 2."], ["Table 1."], ["Table 2."], []], captions
    last = doc[-1].get_text()
    assert "COMPLIANCE WITH ETHICAL STANDARDS" in last and "REFERENCES" in last
    assert all(heading not in last for heading in ("METHODS", "RESULTS", "DISCUSSION", "Conclusions"))
    assert "Conclusions" in doc[3].get_text() and "ACKNOWLEDGMENTS" in doc[3].get_text()
    assert doc[0].search_for("ABSTRACT")[0].x0 < 306
    assert doc[1].search_for("Fig. 2.")[0].y1 < doc[1].search_for("METHODS")[0].y0
    assert doc[2].search_for("Table 1.")[0].y1 < doc[2].search_for("3. RESULTS")[0].y0
    # Post hoc Table 2 floats to the top of page 4, above its Results subsection.
    assert doc[3].search_for("Table 2.")[0].y0 < 100
    assert doc[3].search_for("Table 2.")[0].y1 < doc[3].search_for("3.2.")[0].y0
    for page in doc:
        for font in page.get_fonts(full=True):
            assert doc.extract_font(font[0])[3], f"Unembedded font: {font}"
            assert font[2] != "Type3", font
    source = (HERE / "paper.tex").read_text()
    assert re.findall(r"\\section\{([^}]+)\}", source)[:4] == ["Introduction", "Methods", "Results", "Discussion"]
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S)[1]
    assert 100 <= len(abstract.split()) <= 150
    log = (HERE / "paper.log").read_text()
    assert "Overfull" not in log and "undefined" not in log and "LaTeX Warning" not in log
    subprocess.run(["pdffonts", str(pdf)], check=True, stdout=subprocess.DEVNULL)
    print(f"PASS: 5 Letter pages; figures on pages 1/2, tables on pages 3/4; {len(abstract.split())}-word abstract.")
    print("PASS: embedded Type 1 fonts, readable text sizes, page bounds, references, and LaTeX overflow checks.")

if __name__ == "__main__":
    main()
