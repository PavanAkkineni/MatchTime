import os
import shutil

soccer_net_root = "/work/users/c/h/chidaksh/MatchTime/baidu_embgs/BaiduData"
features_root = "/work/users/c/h/chidaksh/MatchTime/baidu_embgs/features/baidu_soccer_embeddings"

feature_files = ["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"]

def create_directory_structure(feature_files):
    for league in os.listdir(soccer_net_root):
        league_dir = os.path.join(soccer_net_root, league)
        if os.path.exists(league_dir):
            for season in os.listdir(league_dir):
                season_dir = os.path.join(league_dir, season)
                if os.path.isdir(season_dir): 
                    new_dir_name = f"{league}_{season}"
                    new_dir_path = os.path.join(features_root, new_dir_name)
                    copy_new_dir_path = new_dir_path
                    if not os.path.exists(new_dir_path):
                        os.makedirs(new_dir_path)
                    for game_folder in os.listdir(season_dir):
                        game_path = os.path.join(season_dir, game_folder)
                        new_dir_path = os.path.join(copy_new_dir_path, game_folder)
                        if not os.path.exists(new_dir_path):
                            os.makedirs(new_dir_path)
                            print(f"Created directory: {new_dir_path}")
                        if os.path.isdir(game_path):
                            for npy_file in feature_files:
                                source_file = os.path.join(game_path, npy_file)
                                if os.path.exists(source_file):
                                    shutil.move(source_file, new_dir_path)
                                    print(f"Moved {npy_file} to {new_dir_path}")

if __name__ == "__main__":
    create_directory_structure(feature_files)
