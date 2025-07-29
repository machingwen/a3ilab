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
            'BrownHair': 0,
            'BlondHair': 1,
            'GrayHair': 2,
            'BlackHair': 3
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
        
class Colored_MNIST:
    def __init__(self):
        self.ATR2IDX = {
            'bw': 0,
            'red': 1,
            'grn': 2,
            'blu': 3,
            'ylw': 4
        }

        self.OBJ2IDX = {
            '0': 0,
            '1': 1,
            '2':2,
            '3':3,
            '4':4,
            '5':5,
            '6':6,
            '7':7,
            '8':8,
            '9':9
        }

        self.IDX2ATR = {v : k for k, v in self.ATR2IDX.items()}

        self.IDX2OBJ = {v : k for k, v in self.OBJ2IDX.items()}

        self.classes = []
        for va in self.IDX2ATR.values():
            for vo in self.IDX2OBJ.values():
                self.classes.append(f"{va}_{vo}")

        self.CLS2IDX = {self.classes[i] : i for i in range(len(self.classes))}