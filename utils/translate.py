label = {"cane": "dog",
         "cavallo": "horse",
         "elefante": "elephant",
         "farfalla": "butterfly",
         "gallina": "chicken",
         "gatto": "cat",
         "mucca": "cow",
         "pecora": "sheep",
         "ragno": "spider",
         "scoiattolo": "squirrel"}

label_to_index = {key: idx for idx, key in enumerate(label.keys())}
index_to_label = {idx: value for key, value in label.items() for idx, k in enumerate(label.keys()) if k == key}
