#!/usr/bin/env python3
"""
Generate final PDF from LaTeX source.
Uses pdflatex if available on the system.
"""

import subprocess
import os
import shutil
from pathlib import Path


def generate_pdf():
    """Compile LaTeX to PDF."""
    
    print("="*70)
    print("PDF GENERATION")
    print("="*70)
    
    tex_file = "PROJECT_REPORT_DRAFT.tex"
    
    if not os.path.exists(tex_file):
        print(f"ERROR: {tex_file} not found")
        return False
    
    # Check for pdflatex
    pdflatex_path = shutil.which("pdflatex")
    if not pdflatex_path:
        print("✗ pdflatex not found on system")
        print("   LaTeX installation required. Alternatives:")
        print("   1. Install MiKTeX (Windows): https://miktex.org/")
        print("   2. Install TeX Live (Linux): sudo apt-get install texlive-full")
        print("   3. Install MacTeX (macOS): https://www.tug.org/mactex/")
        print("\n   Workaround:")
        print("   - Use online compiler: https://www.overleaf.com/")
        print("   - Upload PROJECT_REPORT_DRAFT.tex and compile there")
        return False
    
    print(f"✓ pdflatex found: {pdflatex_path}")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run pdflatex (multiple passes to resolve references)
    print("\nCompiling LaTeX (Pass 1/3)...")
    for pass_num in range(1, 4):
        cmd = [
            pdflatex_path,
            "-interaction=nonstopmode",
            "-output-directory=output",
            tex_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✓ Pass {pass_num}/3 successful")
            else:
                print(f"⚠ Pass {pass_num}/3 had warnings/errors (see log)")
                if "Error" in result.stdout:
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "Error" in line or "error" in line:
                            print(f"  {line}")
        except subprocess.TimeoutExpired:
            print(f"✗ Pass {pass_num}/3 timed out (>60s)")
            return False
        except Exception as e:
            print(f"✗ Pass {pass_num}/3 failed: {e}")
            return False
    
    # Check for PDF output
    pdf_path = Path("output") / tex_file.replace(".tex", ".pdf")
    
    if pdf_path.exists():
        print(f"\n{'='*70}")
        print("✓ PDF GENERATION SUCCESSFUL")
        print(f"{'='*70}")
        print(f"\nOutput: {pdf_path}")
        print(f"Size: {pdf_path.stat().st_size / 1024:.1f} KB")
        
        # Copy to root for easy access
        shutil.copy(pdf_path, "PROJECT_REPORT_DRAFT.pdf")
        print(f"Copied to: PROJECT_REPORT_DRAFT.pdf")
        
        return True
    else:
        print(f"\n✗ PDF not generated")
        print(f"  Expected: {pdf_path}")
        print(f"  Check output/ directory for logs")
        return False


if __name__ == "__main__":
    success = generate_pdf()
    exit(0 if success else 1)
