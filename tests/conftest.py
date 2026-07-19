import sys
from pathlib import Path

# Make the project root importable (so `import src...` works under pytest).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
