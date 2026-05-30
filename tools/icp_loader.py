"""Load and validate ICP YAML configuration."""
import yaml, logging
logger = logging.getLogger(__name__)

def load_icp(path: str) -> dict:
    with open(path) as f:
        icp = yaml.safe_load(f)
    logger.info("Loaded ICP: %s", icp.get("name", "unnamed"))
    return icp
