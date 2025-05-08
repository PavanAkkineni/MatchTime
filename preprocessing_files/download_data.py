#from SoccerNet.Downloader import SoccerNetDownloader
#
#mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/work/users/a/k/akkineni/Matchtime/MatchTime/features/")
#
#print("Hi")
#
#mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"]) # download Features
#mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train", "valid", "test"]) # download Features reduced with PCA
#mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["train", "valid", "test"]) # download Player Bounding Boxes inferred with MaskRCNN
#mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["train", "valid", "test"]) # download Field Calibration inferred with CCBV
#mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train", "valid", "test"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports
#
#print("Going to download games")
#mySoccerNetDownloader.password = "s0cc3rn3t"
#mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"]) # download 224p Videos
#mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train", "valid", "test"]) # download 720p Videos 
#mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet") # download 720p Videos 
#mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet-Tracking") # download single camera RAW Videos 

from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/work/users/a/k/akkineni/Matchtime/MatchTime/features/")

# Download SoccerNet labels
mySoccerNetDownloader.downloadGames(files=["Labels.json"], split=["train", "valid", "test"]) # download labels
print('1')
for i in range(10):
    print('.')

mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"]) # download labels SN v2
print('2')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train", "valid", "test"]) # download labels for camera shot
print('3')
for i in range(10):
    print('.')
# Download SoccerNet features
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train", "valid", "test"]) # download Features
print('4')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train", "valid", "test"]) # download Features reduced with PCA
print('5')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["train", "valid", "test"]) # download Player Bounding Boxes inferred with MaskRCNN
print('6')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["train", "valid", "test"]) # download Field Calibration inferred with CCBV
print('7')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train", "valid", "test"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports
print('8')
for i in range(10):
    print('.')
# Download SoccerNet Challenge set (require password from NDA to download videos)
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["challenge"]) # download ResNET Features
print('9')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["challenge"]) # download ResNET Features reduced with PCA
print('10')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["challenge"]) # download 224p Videos (require password from NDA)
print('11')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["challenge"]) # download 720p Videos (require password from NDA)
print('12')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["challenge"]) # download Player Bounding Boxes inferred with MaskRCNN
print('13')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["challenge"]) # download Field Calibration inferred with CCBV
print('14')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["challenge"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports
print('15')
for i in range(10):
    print('.')
# Download development kit per task
mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test", "challenge"])
print('16')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="caption-2023", split=["train", "valid", "test", "challenge"])
print('17')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])
print('18')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="reid-2023", split=["train", "valid", "test", "challenge"])
print('19')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="spotting-2023", split=["train", "valid", "test", "challenge"])
print('20')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2023", split=["train", "valid", "test", "challenge"], password="s0cc3rn3t")
print('21')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
print('22')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"]) # 4K Images
print('23')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"], version="fullhd") # FullHD Images
print('24')
for i in range(10):
    print('.')
# Download SoccerNet videos (require password from NDA to download videos)
mySoccerNetDownloader.password = "s0cc3rn3t"
print('25')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"]) # download 224p Videos
print('26')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train", "valid", "test"]) # download 720p Videos
print('27')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet") # download 720p Videos
print('28')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadRAWVideo(dataset="SoccerNet-Tracking") # download single camera RAW Videos
print('29')
for i in range(10):
    print('.')
# Download SoccerNet in OSL ActionSpotting format
mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="ResNET_PCA512")
print('30')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="baidu_soccer_embeddings")
print('31')
for i in range(10):
    print('.')
mySoccerNetDownloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="224p", password="s0cc3rn3t")
print('32')
for i in range(10):
    print('.')