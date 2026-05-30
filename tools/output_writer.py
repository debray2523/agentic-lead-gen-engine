"""Write evaluated leads to JSON output files."""
import json, os, logging
from datetime import datetime
logger = logging.getLogger(__name__)

def write_output(leads: list, metadata: dict) -> str:
    out_dir = os.getenv("OUTPUT_DIR", "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"leads_{ts}.json")
    with open(path, "w") as f:
        json.dump({"metadata": metadata, "leads": leads}, f, indent=2, default=str)
    logger.info("Output written to %s", path)
    return path
