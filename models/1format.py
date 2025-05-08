import pandas as pd

# Load CSV
df = pd.read_csv("/work/users/a/k/akkineni/Matchtime/MatchTime/models/top39.csv")

# Helper function to extract event, player1, and player2 from video name
def extract_metadata(video_name):
    parts = video_name.split('_')
    if len(parts) < 3:
        return "unknown", "unknown", "unknown"
    event = parts[-3]
    player1 = parts[-2]
    player2 = parts[-1].replace('.mkv', '')
    return event, player1, player2

# Group by video name
grouped = df.groupby("Video Name")
output_rows = []

for video, group in grouped:
    event, player1, player2 = extract_metadata(video)
    detected = group["Player Name"].tolist()
    detected += [None] * (2 - len(detected))  # pad if only one prediction

    output_rows.append({
        "video_folder": video,
        "players": f"{player1}, {player2}",
        "predicted_players": f"{detected[0]}, {detected[1]}"
    })

# Save output
output_df = pd.DataFrame(output_rows)
output_df.to_csv("/work/users/a/k/akkineni/Matchtime/MatchTime/models/output_predictions_from_top39.csv", index=False)
print("✅ File saved as: output_predictions_from_excel.csv")
