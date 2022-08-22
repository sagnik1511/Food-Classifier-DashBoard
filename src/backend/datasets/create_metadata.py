import pandas as pd
from glob import glob

classes = {
    "Bread": 0,
    "Dairy product": 1,
    "Dessert": 2,
    "Egg": 3,
    "Fried food": 4,
    "Meat": 5,
    "Noodles-Pasta": 6,
    "Rice": 7,
    "Seafood": 8,
    "Soup": 9,
    "Vegetable-Fruit": 10
}


def generate_metadata(files, name):
    dataframe = pd.DataFrame({"path": files})
    dataframe["path"] = dataframe["path"].apply(lambda x: "/".join(x.split("\\")[-2:]))
    dataframe["class"] = dataframe["path"].apply(lambda x: x.split("/")[-2])
    dataframe["label"] = dataframe["class"].apply(lambda x: classes[x])
    dataframe = dataframe.sample(frac=1)
    dataframe.to_csv(f"dataset/food_dataset/{name}.csv", index=False)

train_set = sorted(glob("dataset\\food_dataset\\training\\*\\*g"))
val_set = sorted(glob("dataset\\food_dataset\\validation\\*\\*g"))
test_set = sorted(glob("dataset\\food_dataset\evaluation\\*\\*g"))

generate_metadata(train_set, "train")
generate_metadata(val_set, "validation")
generate_metadata(test_set, "test")

print("Metadata generated successfully...")




