"""Model quota manager - track usage and detect quota exhaustion"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

MODEL_STATS_PATH = "model_stats.json"


class ModelQuotaManager:
    """Manages model quotas and tracks usage."""
    
    def __init__(self):
        self.stats = self._load_stats()
        self._reset_daily_if_needed()
    
    def _load_stats(self) -> Dict:
        """Load model statistics from file."""
        if os.path.exists(MODEL_STATS_PATH):
            try:
                with open(MODEL_STATS_PATH, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    for model in stats:
                        if "count" not in stats[model]:
                            stats[model]["count"] = stats[model].get("daily_count", 0)
                            stats[model]["errors"] = stats[model].get("errors", 0)
                    return stats
            except:
                pass
        return {}
    
    def _reset_daily_if_needed(self):
        """Reset counters if date changed."""
        today = datetime.now().strftime("%Y-%m-%d")
        for model in list(self.stats.keys()):
            if self.stats[model].get("date") != today:
                self.stats[model] = {"date": today, "count": 0, "errors": 0}
    
    def _save_stats(self):
        """Save statistics to file."""
        try:
            with open(MODEL_STATS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except:
            pass
    
    def initialize_model(self, model_name: str):
        """Initialize tracking for a model."""
        today = datetime.now().strftime("%Y-%m-%d")
        if model_name not in self.stats:
            self.stats[model_name] = {"date": today, "count": 0, "errors": 0}
            self._save_stats()
        elif self.stats[model_name].get("date") != today:
            self.stats[model_name] = {"date": today, "count": 0, "errors": 0}
            self._save_stats()
    
    def record_request(self, model_name: str, success: bool = True):
        """Record a request for a model."""
        self.initialize_model(model_name)
        self.stats[model_name]["count"] += 1
        if not success:
            self.stats[model_name]["errors"] = self.stats[model_name].get("errors", 0) + 1
        self._save_stats()
    
    def mark_quota_exhausted(self, model_name: str):
        """Mark a model as quota exhausted (when API returns 429/RESOURCE_EXHAUSTED)."""
        self.initialize_model(model_name)
        self.stats[model_name]["quota_exhausted"] = True
        self.stats[model_name]["last_error_time"] = datetime.now().isoformat()
        self._save_stats()
    
    def check_quota_status(self, model_name: str) -> Dict:
        """Check if model had 429 errors."""
        self.initialize_model(model_name)
        quota_exhausted = self.stats[model_name].get("quota_exhausted", False)
        
        return {
            "model": model_name,
            "quota_exhausted_detected": quota_exhausted,
            "status": "error" if quota_exhausted else "ok"
        }
    
    def get_next_available_model(self, models: List[str]) -> Optional[str]:
        """Get model with lowest usage that's not exhausted."""
        candidates = []
        for model in models:
            status = self.check_quota_status(model)
            if status["status"] != "exhausted":
                candidates.append((model, status["percentage"]))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def is_all_exhausted(self, models: List[str]) -> bool:
        """Check if all models are exhausted."""
        for model in models:
            if self.check_quota_status(model)["status"] != "exhausted":
                return False
        return True
    
    def get_status_summary(self, models: List[str]) -> str:
        """Get status summary."""
        all_status = [self.check_quota_status(m) for m in models]
        exhausted = [s for s in all_status if s["status"] == "error"]
        
        if exhausted and len(exhausted) == len(models):
            return "All models returned errors"
        elif exhausted:
            return f"Some models had errors"
        return "Models available"
    
    def get_detailed_status(self, models: List[str]) -> List[Dict]:
        """Get detailed status for all models with usage info."""
        return [self.check_quota_status(m) for m in models]


quota_manager = ModelQuotaManager()

if __name__ == "__main__":
    models = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
    for model in models:
        quota_manager.initialize_model(model)
        for _ in range(5):
            quota_manager.record_request(model)
    print(quota_manager.get_status_summary(models))
