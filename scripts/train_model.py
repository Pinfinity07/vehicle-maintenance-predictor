"""Script to train the model"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.training import train_model

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Training Vehicle Maintenance Predictor Model")
    print("=" * 60 + "\n")
    
    best_dt, train_acc, test_acc, preprocessor = train_model()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60 + "\n")
