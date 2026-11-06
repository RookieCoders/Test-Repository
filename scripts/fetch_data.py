from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.data import acquire
print(acquire(Path(__file__).resolve().parents[1]/'data/raw',manual='--manual' in sys.argv))
