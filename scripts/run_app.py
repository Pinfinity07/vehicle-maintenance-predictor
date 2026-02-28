"""Script to run the Gradio web app"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.app import build_app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Launching Vehicle Maintenance Predictor Web App")
    print("=" * 60 + "\n")
    
    app = build_app()
    app.launch(share=True)
