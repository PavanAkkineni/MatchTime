import pandas as pd

# Load your original predictions CSV
df = pd.read_csv("/work/users/a/k/akkineni/Matchtime/MatchTime/models/output_predictions_from_top39.csv")

# Function to filter out single-letter names
def filter_name_parts(name):
    return [part.strip() for part in name.split() if len(part.strip()) > 2]

# Function to compute accuracy
def compute_accuracy(players, predicted_players):
    players = [filter_name_parts(p) for p in str(players).split(',')]
    print("players", players)
    predicted = [filter_name_parts(p) for p in str(predicted_players).split(',')]
    print("predicted", predicted)

    flat_players = [n for group in players for n in group]
    flat_predicted = [n for group in predicted for n in group]
    print("flat_players", flat_players)
    print("flat_predicted", flat_predicted)

    if any("nodata" in p.lower() for p in flat_players):
        actual = [p.lower() for p in flat_players if p.lower() != "nodata"]
        out = 100 if actual and any(pred.lower() in actual for pred in flat_predicted) else 0
        print("out", out)
        return out
    else:
        matches = sum(1 for pred in flat_predicted if pred.lower() in [p.lower() for p in flat_players])
        out = 100 if matches >= 2 else 75 if matches == 1 else 0
        print("out", out)
        return out

# Extract event type
df["event"] = df["video_folder"].apply(lambda x: x.split('_')[-3] if isinstance(x, str) else "unknown")

# Compute refined accuracy
df["accuracy"] = [compute_accuracy(p, pred) for p, pred in zip(df["players"], df["predicted_players"])]

# Group by event and compute mean accuracy, count, and std
event_accuracy = df.groupby("event").agg(
    mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    data_points=("accuracy", "count")
).reset_index()

# Save final result
event_accuracy.to_csv("/work/users/a/k/akkineni/Matchtime/MatchTime/models/refined_event_accuracy_top391.csv", index=False)
print("✅ Saved event-wise refined accuracy with counts and std deviation to 'refined_event_accuracy.csv'")
