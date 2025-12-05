import json
import time
import os

STATUS_FILE = "benchmark_status.json"

class BenchmarkManager:
    def __init__(self):
        self.status = {
            "active_model": None,
            "task": "Idle",
            "progress": 0,
            "logs": [],
            "results": []
        }
        self._load_status()

    def _load_status(self):
        if os.path.exists(STATUS_FILE):
            try:
                with open(STATUS_FILE, "r") as f:
                    saved_status = json.load(f)
                    # Keep results, reset active state
                    self.status["results"] = saved_status.get("results", [])
            except:
                pass
        self._save_status()

    def _save_status(self):
        with open(STATUS_FILE, "w") as f:
            json.dump(self.status, f, indent=2)

    def update_status(self, model=None, task=None, progress=None):
        if model is not None:
            self.status["active_model"] = model
        if task is not None:
            self.status["task"] = task
        if progress is not None:
            self.status["progress"] = progress
        self._save_status()

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.status["logs"].insert(0, log_entry)
        # Keep last 50 logs
        self.status["logs"] = self.status["logs"][:50]
        self._save_status()
        print(message)

    def add_result(self, result_entry):
        # Check if result for this model already exists and update it
        existing_idx = -1
        for i, res in enumerate(self.status["results"]):
            if res["name"] == result_entry["name"]:
                existing_idx = i
                break
        
        if existing_idx >= 0:
            self.status["results"][existing_idx] = result_entry
        else:
            self.status["results"].append(result_entry)
        
        self._save_status()
