import torch

class toy_dataset:
    def __init__(self):
        self.ATR2IDX = {
            'red': 0,
            'green': 1,
            'blue': 2,
            'yellow' : 3,
            'black': 4,
            'purple' : 5,
        }

        self.OBJ2IDX = {
            'Circle': 0,
            'Rectangle': 1,
            'Triangle': 2,
            'Pentagon': 3,
            'Oval': 4,
            'Hexagon': 5,
        }

        self.IDX2ATR = {v : k for k, v in self.ATR2IDX.items()}

        self.IDX2OBJ = {v : k for k, v in self.OBJ2IDX.items()}

        self.classes = []
        for va in self.IDX2ATR.values():
            for vo in self.IDX2OBJ.values():
                self.classes.append(f"{va} {vo}")

        self.CLS2IDX = {self.classes[i] : i for i in range(len(self.classes))}

class Zappo50K:
    def __init__(self):
        self.ATR2IDX = {
            'Heel': 0,
            'Flat': 1,
        }

        self.OBJ2IDX = {
            'Boot': 0,
            'Shoe': 1,
            'Slipper': 2,
            'Sandal': 3,
        }

        self.IDX2ATR = {v : k for k, v in self.ATR2IDX.items()}

        self.IDX2OBJ = {v : k for k, v in self.OBJ2IDX.items()}

        self.classes = []
        for va in self.IDX2ATR.values():
            for vo in self.IDX2OBJ.values():
                self.classes.append(f"{va} {vo}")

        self.CLS2IDX = {self.classes[i] : i for i in range(len(self.classes))}

class CelebA:
    def __init__(self):
        self.ATR2IDX = {
            'Brown_Hair': 0,
            'Blond_Hair': 1,
            'Gray_Hair': 2,
            'Black_Hair': 3
        }

        self.OBJ2IDX = {
            'Male': 0,
            'Female': 1,
        }

        self.IDX2ATR = {v : k for k, v in self.ATR2IDX.items()}

        self.IDX2OBJ = {v : k for k, v in self.OBJ2IDX.items()}

        self.classes = []
        for va in self.IDX2ATR.values():
            for vo in self.IDX2OBJ.values():
                self.classes.append(f"{va} {vo}")

        self.CLS2IDX = {self.classes[i] : i for i in range(len(self.classes))}

class Mnist:
    def __init__(self):
        self.ATR2IDX = {
            'Zero': 0,
            'One': 1,
            'Two': 2,
            'Three': 3,
            'Four': 4,
            'Five' : 5,
            'Six' : 6,
            'Seven' : 7,
            'Eight' : 8,
            'Nine' : 9,
        }

        self.OBJ2IDX = {
            'White': 0,
            'Red': 1,
            'Green': 2,
            'Blue': 3,
            'Yellow': 4
        }

        self.IDX2ATR = {v : k for k, v in self.ATR2IDX.items()}

        self.IDX2OBJ = {v : k for k, v in self.OBJ2IDX.items()}

        self.classes = []
        for va in self.IDX2ATR.values():
            for vo in self.IDX2OBJ.values():
                self.classes.append(f"{va} {vo}")

        self.CLS2IDX = {self.classes[i] : i for i in range(len(self.classes))}

class Phison:
    def __init__(self):
        self.ATR2IDX = {
            'Good': 0,
            'Shift': 1
        }

        self.OBJ2IDX = {
            'Group1': 0,
            'Group2': 1,
            'Group3': 2,
            'Group4': 3,
            'Group5': 4,
            'Group6': 5,
            'Group7': 6
        }

        self.IDX2ATR = {v : k for k, v in self.ATR2IDX.items()}

        self.IDX2OBJ = {v : k for k, v in self.OBJ2IDX.items()}

        self.classes = []
        for va in self.IDX2ATR.values():
            for vo in self.IDX2OBJ.values():
                self.classes.append(f"{va} {vo}")

        self.CLS2IDX = {self.classes[i] : i for i in range(len(self.classes))}
"""
class CelebA:
    def __init__(self):
        self.ATR2IDX = {
            'Brown_Hair': 0,
            'Blond_Hair': 1,
            'Gray_Hair': 2,
            'Black_Hair': 3
        }

        self.OBJ2IDX = {
            'Male': 0,
            'Female': 1,
        }

        self.IDX2ATR = {v: k for k, v in self.ATR2IDX.items()}
        self.IDX2OBJ = {v: k for k, v in self.OBJ2IDX.items()}

        self.classes = []
        for va in self.IDX2ATR.values():
            for vo in self.IDX2OBJ.values():
                class_name = f"{va} {vo}"
                # 過濾掉 'Gray_Hair Female'
                if class_name != "Gray_Hair Female":
                    self.classes.append(class_name)

        self.CLS2IDX = {self.classes[i]: i for i in range(len(self.classes))}
"""