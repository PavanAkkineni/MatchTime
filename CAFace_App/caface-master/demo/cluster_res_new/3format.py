# Re-run after kernel reset
import pandas as pd
import ast

# Reload the CSVs
commentary_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec_new/video_commentaries.csv"
predictions_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/CAFace_App/caface-master/demo/cluster_res_new04/ouput_predictions_csv1-50.csv"

commentary_df = pd.read_csv(commentary_path)
predictions_df = pd.read_csv(predictions_path)

# Ensure string consistency
commentary_df['video_name'] = commentary_df['video_name'].astype(str)
predictions_df['video_folder'] = predictions_df['video_folder'].astype(str)

results = []

# Match and process
for _, pred_row in predictions_df.iterrows():
    video_name = pred_row['video_folder']
    matched = commentary_df[commentary_df['video_name'] == video_name]
    if matched.empty:
        continue
    comm_row = matched.iloc[0]

    # Parse comments
    commentaries = []
    for col in ["commentary_1", "commentary_2", "commentary_3", "commentary_4"]:
        val = comm_row.get(col, "")
        try:
            parsed = ast.literal_eval(val)
            commentaries.append(parsed[0] if isinstance(parsed, list) else str(parsed))
        except:
            commentaries.append(str(val))

    anonymized = str(comm_row['anonymized'])
    best_comment = max(commentaries, key=lambda c: len(set(c.lower().split()) & set(anonymized.lower().split())))

    predicted_players = [p.strip() for p in str(pred_row["predicted_players"]).split(",")]
    description_predict = best_comment
    for player in predicted_players:
        if "[PLAYER]" in description_predict:
            description_predict = description_predict.replace("[PLAYER]", player, 1)

    row_data = {
        "video_name": video_name,
        "anonymized": anonymized,
        "anonymized_predict": best_comment,
        "description": comm_row["description"],
        "description_predict": description_predict,
    }

    for col in comm_row.index:
        row_data[f"commentary_{col}"] = comm_row[col]
    for col in pred_row.index:
        row_data[f"prediction_{col}"] = pred_row[col]

    results.append(row_data)

# Save to CSV
final_df = pd.DataFrame(results)
csv_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec_new/final_description_predictions_output.csv"
final_df.to_csv(csv_path, index=False)
