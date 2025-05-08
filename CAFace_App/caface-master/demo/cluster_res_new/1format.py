import os
import re
import pandas as pd

# Define the root directory
root_dir = "/work/users/a/k/akkineni/Matchtime/MatchTime/CAFace_App/caface-master/demo/cluster_res_new04player1"

# Helper function to extract event, player1, and player2 from folder name
def parse_folder_name(folder_name):
    parts = folder_name.split('_')
    if len(parts) < 3:
        return None, None, None
    event = parts[-3]
    player1 = parts[-2]
    player2 = parts[-1].replace('.mkv', '')
    return event, player1, player2

# Helper function to extract predicted player from PDF filename
def parse_pdf_filename(pdf_name):
    match = re.match(r'cluster_and_aggregate_(.+?)_top\d\.pdf', pdf_name)
    if match:
        return match.group(1)
    return None

# Collect data
data = []

# Traverse through each subfolder
for item in os.listdir(root_dir):
    item_path = os.path.join(root_dir, item)
    if os.path.isdir(item_path):
        event, player1, player2 = parse_folder_name(item)
        predicted_players = []
        
        for subfolder in os.listdir(item_path):
            subfolder_path = os.path.join(item_path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(".pdf"):
                        predicted_player = parse_pdf_filename(file)
                        if predicted_player:
                            predicted_players.append(predicted_player)
        
        # Pad predicted_players to always have two values
        predicted_players += [None] * (2 - len(predicted_players))
        data.append({
            "video_folder": item,
            "players": f"{player1}, {player2}",
            "predicted_players": f"{predicted_players[0]}, {predicted_players[1]}"
        })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
output_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/CAFace_App/caface-master/demo/cluster_res_new04player1/output_predictions.csv"
df.to_csv(output_path, index=False)

