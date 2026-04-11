import json
from datetime import datetime
from config import DATA_DIR

def log_result(params: dict, val_history: list, val_acc_history: list, test_accuracy: float, test_loss: float):
    results = {
        "date": datetime.now().isoformat(),
        "epochs_run": len(val_history),
        "best_val_loss": round(min(val_history), 4),
        "best_val_accuracy": round(max(val_acc_history), 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_loss": round(test_loss, 4),
        "params": params,
    }

    log_path = DATA_DIR / "log_result.json"
    try:
        with open(log_path, "r") as f:
            logs = json.load(f)
    except FileNotFoundError:
        logs = []

    logs.append(results)

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"Results logged to {log_path}")