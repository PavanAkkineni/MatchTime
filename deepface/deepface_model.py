from deepface import DeepFace

result = DeepFace.verify(
  img1_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/deepface/test1.jpg",
  img2_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/deepface/test1.jpg",
)

objs = DeepFace.analyze(
  img_path = "/work/users/a/k/akkineni/Matchtime/MatchTime/deepface/img3.jpg", 
  actions = ['age', 'gender', 'race', 'emotion'],
)

print(result)
print(objs)