import subprocess

import sys

import os

from pathlib import Path

from datetime import datetime

from fpdf import FPDF

def main():

    if len(sys.argv) < 2:

        print("Usage: python run_to_pdf.py <script.py>")

        sys.exit(1)

    script = Path(sys.argv[1])

    out_dir = Path("results")

    out_dir.mkdir(exist_ok = True)

    #Rum the model script and capture everything it prints

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    proc = subprocess.run(
        [sys.executable, str(script)],
        
        capture_output = True, text = True,

        encoding = "utf-8", errors = "replace", env = env
    )

    output = proc.stdout

    if proc.returncode != 0:

        output += "\n[stderr]\n" + proc.stderr

    print(output) #still show it in terminal

    #Build the PDF (Courier keeps tables and classifications reports aligned)

    header = f"{script.name}  |  {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "=" * 70 + "\n\n"

    text = (header + output).expandtabs(4)

    text = text.encode("latin-1", "replace").decode("latin-1") #core fonts are latin-1 only

    pdf = FPDF()
    
    pdf.set_auto_page_break(auto = True, margin = 15)

    pdf.add_page()

    pdf.set_font("Courier", size = 9)

    pdf.multi_cell(0, 4.5, text, new_x = "LMARGIN", new_y = "NEXT")

    pdf_path = out_dir / f"{script.stem}_output.pdf"

    pdf.output(str(pdf_path))

    print(f"\nSaved PDF: {pdf_path}")

if __name__ == "__main__":

    main()
