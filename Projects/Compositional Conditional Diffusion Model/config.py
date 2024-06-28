import torch

ATR2IDX = {
    'good': 0,
    'broke': 1,
    'shift': 2,
}

OBJ2IDX = {
    'group1': 0,
    'group3': 1,
    'group7': 2,
}
# OBJ2IDX = {
#     'Group1': 0,
#     'Group2': 1,
#     'Group3': 2,
#     'Group4': 3,
#     'Group5': 4,
# }
# OBJ2IDX = {
#     'C0201': 0,
#     'F1210': 1,
#     'L2016': 2,
#     'R0402': 3,
#     'SOT23': 4,
# }
IDX2ATR = {v : k for k, v in ATR2IDX.items()}

IDX2OBJ = {v : k for k, v in OBJ2IDX.items()}

classes = []
for va in IDX2ATR.values():
    for vo in IDX2OBJ.values():
        classes.append(f"{va} {vo}")
        
CLS2IDX = {classes[i] : i for i in range(len(classes))}