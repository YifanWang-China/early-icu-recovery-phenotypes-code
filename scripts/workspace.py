"""Location of private inputs and generated outputs."""
import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("ICU_WORKSPACE", ".")).expanduser().resolve()
