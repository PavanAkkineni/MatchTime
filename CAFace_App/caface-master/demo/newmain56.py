import sys
import pyrootutils
import os
import torch

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(os.path.dirname(root))
sys.path.append(os.path.join(os.path.dirname(root), 'caface'))

import argparse
import cv2
from face_detection.detector import FaceDetector
from face_alignment.aligner import FaceAligner
from model_loader import load_caface
from dataset import get_all_files, natural_sort, prepare_imagelist_dataloader, to_tensor
from tqdm import tqdm
import numpy as np
from inference import infer_features, fuse_feature
import visualization
from newpipe56 import process_all_videos


import sys
import pyrootutils
import os
import torch
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from face_detection.detector import FaceDetector
from face_alignment.aligner import FaceAligner
from model_loader import load_caface
from dataset import get_all_files, natural_sort, prepare_imagelist_dataloader, to_tensor
from inference import infer_features, fuse_feature
import visualization
import shutil

# Set up root
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
sys.path.append(os.path.dirname(root))
sys.path.append(os.path.join(os.path.dirname(root), 'caface'))

# Safe folder cleaner
def safe_clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# Argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument("--ckpt_path", type=str, default='../pretrained_models/CAFace_AdaFaceWebFace4M.ckpt')
parser.add_argument("--cluster_root", type=str, default='/work/users/a/k/akkineni/Matchtime/MatchTime/models/uniquefaces_new')
parser.add_argument("--gallery_dir", type=str, default='/work/users/a/k/akkineni/Matchtime/MatchTime/models/player_images/england_epl_2015-2016')
parser.add_argument("--save_root", type=str, default='/work/users/a/k/akkineni/Matchtime/MatchTime/CAFace_App/caface-master/demo/cluster_res_new04')
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--fusion_method", type=str, default='cluster_and_aggregate',
                    choices=['cluster_and_aggregate', 'average'])
args = parser.parse_args()
video_folder = "/work/users/a/k/akkineni/Matchtime/MatchTime/models/1min30sec_new56"
# Load components
detector = FaceDetector()
aligner = FaceAligner()
aggregator, model, hyper_param = load_caface(args.ckpt_path, device=args.device)
summary_rows = []

# Walk through all clusters in nested folders
for file in os.listdir(video_folder):
    filename = os.path.join(video_folder,file)
    print(video_folder)
    print(filename)
    process_all_videos(filename)
    
    game_clusters_folder = os.path.join(args.cluster_root,file)
    
    cluster_sizes = []
    for cluster in os.listdir(game_clusters_folder):
        cluster_path = os.path.join(game_clusters_folder, cluster)
        if os.path.isdir(cluster_path):
            num_files = len([f for f in os.listdir(cluster_path) if os.path.isfile(os.path.join(cluster_path, f))])
            cluster_sizes.append((cluster, num_files))
    
    # Step 2: Sort by number of files descending
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Step 3: Get max count
    if not cluster_sizes:
        print("⚠️ No cluster folders found.")
        continue
    
    max_count = cluster_sizes[0][1]
    
    
    # Step 4: Filter clusters with exactly the max count
    top_clusters = [name for name, count in cluster_sizes if count == max_count]
    if len(top_clusters) >= 2:
        pass
    else:
        if len(cluster_sizes) > 1:
            max_cnt2 = cluster_sizes[1][1]
            top_clusters2 = [name for name, count in cluster_sizes if count == max_cnt2]
            top_clusters.append(top_clusters2[0])
        else:
            pass
    # Step 5: Process only these clusters
    for cluster in sorted(top_clusters):
        cluster_path = os.path.join(game_clusters_folder, cluster)

        print(f"\n📁 Processing Cluster: {cluster_path}")
        relative_cluster_path = os.path.relpath(cluster_path, args.cluster_root)
        print("relative_cluster_path", relative_cluster_path)
        save_dir = os.path.join(args.save_root, relative_cluster_path)
        
        aligned_dir = os.path.join(save_dir, 'aligned')

        if os.path.exists(aligned_dir):
            shutil.rmtree(aligned_dir)
        os.makedirs(aligned_dir, exist_ok=True)

        aligned_paths = []
        for i, img_path in enumerate(natural_sort(get_all_files(cluster_path))):
            img = cv2.imread(img_path)
            detected = detector.detect(img)
            aligned = aligner.align(img)
            if aligned is None:
                continue
            aligned_path = os.path.join(aligned_dir, f"{i}.jpg")
            aligned.save(aligned_path)
            aligned_paths.append(aligned_path)

        if len(aligned_paths) < 1:
            print(f"⚠️ Skipping {cluster_path}, no valid aligned faces.")
            continue

        dataloader = prepare_imagelist_dataloader(aligned_paths, batch_size=16, num_workers=0)
        probe_features, probe_intermediates = infer_features(dataloader, model, aggregator, hyper_param, device=args.device)
        probe_fused_feature, probe_weights = fuse_feature(probe_features, aggregator, probe_intermediates,
                                                          method=args.fusion_method, device=args.device)

        top_folder = (os.path.basename(file)).split("_")[0]
        match_folder = os.path.join(args.gallery_dir, top_folder)
        if not os.path.exists(match_folder):
            print(f"⚠️ Skipping: Gallery folder not found → {match_folder}")
            continue

        gallery_path_list = [os.path.join(match_folder, f)
                             for f in os.listdir(match_folder)
                             if os.path.isfile(os.path.join(match_folder, f))]

        max_similarity = -1.0
        best_pdf_path = ""
        best_player = ""
        all_pdf_paths = []

        for gallery_path in gallery_path_list:
            player = os.path.splitext(os.path.basename(gallery_path))[0]
            pdf_path = os.path.join(save_dir, f"{args.fusion_method}_{player}_top2.pdf")
            gallery_image = aligner.align(detector.detect(cv2.imread(gallery_path)))
            if gallery_image is None:
                continue
            gallery_image_tensor = to_tensor(gallery_image, device=args.device)
            with torch.no_grad():
                gallery_feature, _ = model(gallery_image_tensor)
            gallery_feature = gallery_feature.detach().cpu().numpy()
            print("pdf_path",pdf_path)
            similarity = visualization.make_similarity_plot(
                pdf_path,
                probe_features, probe_weights, probe_fused_feature,
                aligned_paths,
                gallery_feature,
                gallery_image
            )

            all_pdf_paths.append((pdf_path, similarity))
            if similarity > max_similarity:
                max_similarity = similarity
                best_pdf_path = pdf_path
                best_player = player

        for pdf_path, _ in all_pdf_paths:
            if pdf_path != best_pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)

        summary_rows.append({
            "Folder": game_clusters_folder,
            "Cluster": cluster,
            "Similarity": max_similarity,
            "Top_Player": best_player,
            "PDF_Path": best_pdf_path
        })

        print(f"✅ Best match for {relative_cluster_path}: {best_player} → {os.path.basename(best_pdf_path)}")

        # 🔁 UPDATE EXCEL AFTER EACH CLUSTER
        folder_clusters = defaultdict(list)
        for row in summary_rows:
            folder = row["Folder"]
            cluster = row["Cluster"]
            top_player = row["Top_Player"]

            aligned_dir = os.path.join(args.save_root, os.path.relpath(folder, args.cluster_root), cluster, 'aligned')
            image_count = len([
                f for f in os.listdir(aligned_dir)
                if os.path.isfile(os.path.join(aligned_dir, f))
            ])

            folder_clusters[folder].append({
                "Cluster": cluster,
                "Image_Count": image_count,
                "Top_Player": top_player
            })

        final_rows = []
        for folder, clusters in folder_clusters.items():
            top_clusters = sorted(clusters, key=lambda x: x["Image_Count"], reverse=True)[:2]
            for entry in top_clusters:
                final_rows.append({
                    "Folder Name": folder,
                    "Cluster ID": entry["Cluster"],
                    "Image Count": entry["Image_Count"],
                    "Top Matched Player": entry["Top_Player"]
                })

        df_final = pd.DataFrame(final_rows)
        excel_path = os.path.join(args.save_root, "top2_clusters_by_folder.xlsx")
        df_final.to_excel(excel_path, index=False)
        print(f"📝 Excel updated at: {excel_path}")
        summary_rows.clear()
        torch.cuda.empty_cache()
        del probe_features, probe_intermediates, probe_fused_feature, probe_weights
        del gallery_feature, gallery_image_tensor, gallery_image
        del dataloader, aligned_paths

