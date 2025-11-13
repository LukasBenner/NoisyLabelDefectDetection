

from typing import Any, Dict, Optional

from utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    
    if not metric_name:
        log.info("No optimized metric specified; skipping metric retrieval.")
        return None

    if metric_name not in metric_dict:
        raise KeyError(f"Metric '{metric_name}' not found in the metric dictionary.")
    
    return metric_dict[metric_name]