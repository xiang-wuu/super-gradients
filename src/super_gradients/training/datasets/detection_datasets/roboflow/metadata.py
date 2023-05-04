__all__ = ["DATASETS_METADATA", "DATASETS_CATEGORIES"]


DATASETS_METADATA = {
    "aerial-pool": {"category": "aerial", "train": 673, "test": 96, "valid": 177, "size": 946, "num_classes": 7, "num_classes_found": 5},
    "secondary-chains": {"category": "aerial", "train": 103, "test": 16, "valid": 43, "size": 162, "num_classes": 1, "num_classes_found": 1},
    "aerial-spheres": {"category": "aerial", "train": 318, "test": 51, "valid": 104, "size": 473, "num_classes": 6, "num_classes_found": 6},
    "soccer-players-5fuqs": {"category": "aerial", "train": 114, "test": 16, "valid": 33, "size": 163, "num_classes": 3, "num_classes_found": 3},
    "weed-crop-aerial": {"category": "aerial", "train": 823, "test": 118, "valid": 235, "size": 1176, "num_classes": 2, "num_classes_found": 2},
    "aerial-cows": {"category": "aerial", "train": 1084, "test": 299, "valid": 340, "size": 1723, "num_classes": 1, "num_classes_found": 1},
    "cloud-types": {"category": "aerial", "train": 3528, "test": 504, "valid": 1008, "size": 5040, "num_classes": 4, "num_classes_found": 4},
    "apex-videogame": {"category": "videogames", "train": 2583, "test": 415, "valid": 691, "size": 3689, "num_classes": 2, "num_classes_found": 2},
    "farcry6-videogame": {"category": "videogames", "train": 82, "test": 14, "valid": 24, "size": 120, "num_classes": 11, "num_classes_found": 11},
    "csgo-videogame": {"category": "videogames", "train": 1774, "test": 207, "valid": 446, "size": 2427, "num_classes": 2, "num_classes_found": 2},
    "avatar-recognition-nuexe": {"category": "videogames", "train": 225, "test": 30, "valid": 59, "size": 314, "num_classes": 1, "num_classes_found": 1},
    "halo-infinite-angel-videogame": {"category": "videogames", "train": 462, "test": 71, "valid": 136, "size": 669, "num_classes": 4, "num_classes_found": 4},
    "team-fight-tactics": {"category": "videogames", "train": 1162, "test": 112, "valid": 307, "size": 1581, "num_classes": 59, "num_classes_found": 59},
    "robomasters-285km": {"category": "videogames", "train": 1945, "test": 278, "valid": 556, "size": 2779, "num_classes": 9, "num_classes_found": 9},
    "stomata-cells": {"category": "microscopic", "train": 1482, "test": 209, "valid": 414, "size": 2105, "num_classes": 2, "num_classes_found": 2},
    "bccd-ouzjz": {"category": "microscopic", "train": 255, "test": 36, "valid": 73, "size": 364, "num_classes": 3, "num_classes_found": 3},
    "parasites-1s07h": {"category": "microscopic", "train": 1484, "test": 215, "valid": 411, "size": 2110, "num_classes": 8, "num_classes_found": 8},
    "cells-uyemf": {"category": "microscopic", "train": 16, "test": 2, "valid": 4, "size": 22, "num_classes": 1, "num_classes_found": 1},
    "4-fold-defect": {"category": "microscopic", "train": 503, "test": 33, "valid": 134, "size": 670, "num_classes": 1, "num_classes_found": 1},
    "bacteria-ptywi": {"category": "microscopic", "train": 30, "test": 10, "valid": 10, "size": 50, "num_classes": 1, "num_classes_found": 1},
    "cotton-plant-disease": {"category": "microscopic", "train": 724, "test": 102, "valid": 198, "size": 1024, "num_classes": 1, "num_classes_found": 1},
    "mitosis-gjs3g": {"category": "microscopic", "train": 213, "test": 30, "valid": 61, "size": 304, "num_classes": 1, "num_classes_found": 1},
    "phages": {"category": "microscopic", "train": 1155, "test": 103, "valid": 164, "size": 1422, "num_classes": 2, "num_classes_found": 2},
    "liver-disease": {"category": "microscopic", "train": 2782, "test": 400, "valid": 794, "size": 3976, "num_classes": 4, "num_classes_found": 4},
    "asbestos": {"category": "microscopic", "train": 932, "test": 133, "valid": 266, "size": 1331, "num_classes": 4, "num_classes_found": 4},
    "underwater-pipes-4ng4t": {"category": "underwater", "train": 5617, "test": 779, "valid": 1575, "size": 7971, "num_classes": 1, "num_classes_found": 1},
    "aquarium-qlnqy": {"category": "underwater", "train": 448, "test": 63, "valid": 127, "size": 638, "num_classes": 7, "num_classes_found": 7},
    "peixos-fish": {"category": "underwater", "train": 821, "test": 118, "valid": 261, "size": 1200, "num_classes": 12, "num_classes_found": 2},
    "underwater-objects-5v7p8": {"category": "underwater", "train": 5320, "test": 760, "valid": 1520, "size": 7600, "num_classes": 5, "num_classes_found": 5},
    "coral-lwptl": {"category": "underwater", "train": 427, "test": 74, "valid": 93, "size": 594, "num_classes": 14, "num_classes_found": 14},
    "tweeter-posts": {"category": "documents", "train": 87, "test": 9, "valid": 21, "size": 117, "num_classes": 2, "num_classes_found": 2},
    "tweeter-profile": {"category": "documents", "train": 425, "test": 61, "valid": 121, "size": 607, "num_classes": 1, "num_classes_found": 1},
    "document-parts": {"category": "documents", "train": 906, "test": 150, "valid": 318, "size": 1374, "num_classes": 2, "num_classes_found": 2},
    "activity-diagrams-qdobr": {"category": "documents", "train": 259, "test": 45, "valid": 74, "size": 378, "num_classes": 19, "num_classes_found": 19},
    "signatures-xc8up": {"category": "documents", "train": 257, "test": 37, "valid": 74, "size": 368, "num_classes": 1, "num_classes_found": 1},
    "paper-parts": {"category": "documents", "train": 8472, "test": 1209, "valid": 2359, "size": 12040, "num_classes": 46, "num_classes_found": 19},
    "tabular-data-wf9uh": {"category": "documents", "train": 3251, "test": 206, "valid": 409, "size": 3866, "num_classes": 12, "num_classes_found": 12},
    "paragraphs-co84b": {"category": "documents", "train": 4209, "test": 633, "valid": 1221, "size": 6063, "num_classes": 7, "num_classes_found": 7},
    "thermal-dogs-and-people-x6ejw": {
        "category": "electromagnetic",
        "train": 142,
        "test": 20,
        "valid": 41,
        "size": 203,
        "num_classes": 2,
        "num_classes_found": 2,
    },
    "solar-panels-taxvb": {"category": "electromagnetic", "train": 112, "test": 19, "valid": 30, "size": 161, "num_classes": 5, "num_classes_found": 5},
    "radio-signal": {"category": "electromagnetic", "train": 1954, "test": 278, "valid": 566, "size": 2798, "num_classes": 2, "num_classes_found": 2},
    "thermal-cheetah-my4dp": {"category": "electromagnetic", "train": 90, "test": 14, "valid": 25, "size": 129, "num_classes": 2, "num_classes_found": 2},
    "x-ray-rheumatology": {"category": "electromagnetic", "train": 135, "test": 16, "valid": 34, "size": 185, "num_classes": 12, "num_classes_found": 12},
    "acl-x-ray": {"category": "electromagnetic", "train": 2141, "test": 306, "valid": 612, "size": 3059, "num_classes": 1, "num_classes_found": 1},
    "abdomen-mri": {"category": "electromagnetic", "train": 1887, "test": 238, "valid": 479, "size": 2604, "num_classes": 1, "num_classes_found": 1},
    "axial-mri": {"category": "electromagnetic", "train": 253, "test": 39, "valid": 79, "size": 371, "num_classes": 2, "num_classes_found": 2},
    "gynecology-mri": {"category": "electromagnetic", "train": 2122, "test": 253, "valid": 526, "size": 2901, "num_classes": 3, "num_classes_found": 3},
    "brain-tumor-m2pbp": {"category": "electromagnetic", "train": 6930, "test": 990, "valid": 1980, "size": 9900, "num_classes": 3, "num_classes_found": 3},
    "bone-fracture-7fylg": {"category": "electromagnetic", "train": 326, "test": 44, "valid": 88, "size": 458, "num_classes": 4, "num_classes_found": 4},
    "flir-camera-objects": {"category": "electromagnetic", "train": 9306, "test": 1452, "valid": 2854, "size": 13612, "num_classes": 4, "num_classes_found": 4},
    "hand-gestures-jps7z": {"category": "real world", "train": 642, "test": 94, "valid": 178, "size": 914, "num_classes": 14, "num_classes_found": 14},
    "smoke-uvylj": {"category": "real world", "train": 522, "test": 76, "valid": 148, "size": 746, "num_classes": 1, "num_classes_found": 1},
    "wall-damage": {"category": "real world", "train": 325, "test": 40, "valid": 96, "size": 461, "num_classes": 3, "num_classes_found": 3},
    "corrosion-bi3q3": {"category": "real world", "train": 840, "test": 105, "valid": 304, "size": 1249, "num_classes": 3, "num_classes_found": 3},
    "excavators-czvg9": {"category": "real world", "train": 2244, "test": 144, "valid": 267, "size": 2655, "num_classes": 3, "num_classes_found": 3},
    "chess-pieces-mjzgj": {"category": "real world", "train": 202, "test": 29, "valid": 58, "size": 289, "num_classes": 13, "num_classes_found": 13},
    "road-signs-6ih4y": {"category": "real world", "train": 1376, "test": 229, "valid": 488, "size": 2093, "num_classes": 21, "num_classes_found": 21},
    "street-work": {"category": "real world", "train": 611, "test": 87, "valid": 175, "size": 873, "num_classes": 11, "num_classes_found": 8},
    "construction-safety-gsnvb": {"category": "real world", "train": 997, "test": 90, "valid": 119, "size": 1206, "num_classes": 5, "num_classes_found": 5},
    "road-traffic": {"category": "real world", "train": 494, "test": 133, "valid": 187, "size": 814, "num_classes": 12, "num_classes_found": 7},
    "washroom-rf1fa": {"category": "real world", "train": 1885, "test": 318, "valid": 775, "size": 2978, "num_classes": 10, "num_classes_found": 10},
    "circuit-elements": {"category": "real world", "train": 672, "test": 36, "valid": 64, "size": 772, "num_classes": 46, "num_classes_found": 31},
    "mask-wearing-608pr": {"category": "real world", "train": 105, "test": 15, "valid": 29, "size": 149, "num_classes": 2, "num_classes_found": 2},
    "cables-nl42k": {"category": "real world", "train": 4816, "test": 794, "valid": 1220, "size": 6830, "num_classes": 11, "num_classes_found": 11},
    "soda-bottles": {"category": "real world", "train": 1547, "test": 216, "valid": 486, "size": 2249, "num_classes": 6, "num_classes_found": 3},
    "truck-movement": {"category": "real world", "train": 740, "test": 107, "valid": 215, "size": 1062, "num_classes": 7, "num_classes_found": 5},
    "wine-labels": {"category": "real world", "train": 3172, "test": 630, "valid": 841, "size": 4643, "num_classes": 12, "num_classes_found": 12},
    "digits-t2eg6": {"category": "real world", "train": 2912, "test": 367, "valid": 824, "size": 4103, "num_classes": 10, "num_classes_found": 10},
    "vehicles-q0x2v": {"category": "real world", "train": 2634, "test": 458, "valid": 966, "size": 4058, "num_classes": 12, "num_classes_found": 12},
    "peanuts-sd4kf": {"category": "real world", "train": 268, "test": 42, "valid": 77, "size": 387, "num_classes": 2, "num_classes_found": 2},
    "printed-circuit-board": {"category": "real world", "train": 548, "test": 44, "valid": 80, "size": 672, "num_classes": 34, "num_classes_found": 23},
    "pests-2xlvx": {"category": "real world", "train": 509, "test": 55, "valid": 153, "size": 717, "num_classes": 28, "num_classes_found": 28},
    "cavity-rs0uf": {"category": "real world", "train": 287, "test": 38, "valid": 93, "size": 418, "num_classes": 2, "num_classes_found": 2},
    "leaf-disease-nsdsr": {"category": "real world", "train": 1589, "test": 296, "valid": 616, "size": 2501, "num_classes": 3, "num_classes_found": 3},
    "marbles": {"category": "real world", "train": 54, "test": 32, "valid": 19, "size": 105, "num_classes": 2, "num_classes_found": 2},
    "pills-sxdht": {"category": "real world", "train": 316, "test": 45, "valid": 90, "size": 451, "num_classes": 8, "num_classes_found": 8},
    "poker-cards-cxcvz": {"category": "real world", "train": 964, "test": 128, "valid": 193, "size": 1285, "num_classes": 53, "num_classes_found": 53},
    "number-ops": {"category": "real world", "train": 4869, "test": 623, "valid": 1636, "size": 7128, "num_classes": 15, "num_classes_found": 15},
    "insects-mytwu": {"category": "real world", "train": 696, "test": 100, "valid": 199, "size": 995, "num_classes": 10, "num_classes_found": 10},
    "cotton-20xz5": {"category": "real world", "train": 367, "test": 20, "valid": 19, "size": 406, "num_classes": 4, "num_classes_found": 4},
    "furniture-ngpea": {"category": "real world", "train": 454, "test": 74, "valid": 161, "size": 689, "num_classes": 3, "num_classes_found": 3},
    "cable-damage": {"category": "real world", "train": 919, "test": 134, "valid": 265, "size": 1318, "num_classes": 2, "num_classes_found": 2},
    "animals-ij5d2": {"category": "real world", "train": 700, "test": 100, "valid": 200, "size": 1000, "num_classes": 10, "num_classes_found": 10},
    "coins-1apki": {"category": "real world", "train": 6121, "test": 699, "valid": 1599, "size": 8419, "num_classes": 4, "num_classes_found": 4},
    "apples-fvpl5": {"category": "real world", "train": 489, "test": 30, "valid": 178, "size": 697, "num_classes": 2, "num_classes_found": 2},
    "people-in-paintings": {"category": "real world", "train": 634, "test": 81, "valid": 194, "size": 909, "num_classes": 1, "num_classes_found": 1},
    "circuit-voltages": {"category": "real world", "train": 92, "test": 15, "valid": 25, "size": 132, "num_classes": 6, "num_classes_found": 6},
    "uno-deck": {"category": "real world", "train": 6295, "test": 899, "valid": 1798, "size": 8992, "num_classes": 15, "num_classes_found": 15},
    "grass-weeds": {"category": "real world", "train": 1661, "test": 245, "valid": 580, "size": 2486, "num_classes": 1, "num_classes_found": 1},
    "gauge-u2lwv": {"category": "real world", "train": 158, "test": 25, "valid": 52, "size": 235, "num_classes": 2, "num_classes_found": 2},
    "sign-language-sokdr": {"category": "real world", "train": 504, "test": 72, "valid": 144, "size": 720, "num_classes": 26, "num_classes_found": 26},
    "valentines-chocolate": {"category": "real world", "train": 68, "test": 6, "valid": 13, "size": 87, "num_classes": 22, "num_classes_found": 22},
    "fish-market-ggjso": {"category": "real world", "train": 14180, "test": 1202, "valid": 3116, "size": 18498, "num_classes": 21, "num_classes_found": 19},
    "lettuce-pallets": {"category": "real world", "train": 1060, "test": 151, "valid": 299, "size": 1510, "num_classes": 5, "num_classes_found": 5},
    "shark-teeth-5atku": {"category": "real world", "train": 191, "test": 36, "valid": 53, "size": 280, "num_classes": 4, "num_classes_found": 4},
    "bees-jt5in": {"category": "real world", "train": 5640, "test": 836, "valid": 1604, "size": 8080, "num_classes": 1, "num_classes_found": 1},
    "sedimentary-features-9eosf": {"category": "real world", "train": 156, "test": 21, "valid": 45, "size": 222, "num_classes": 5, "num_classes_found": 5},
    "currency-v4f8j": {"category": "real world", "train": 576, "test": 82, "valid": 155, "size": 813, "num_classes": 10, "num_classes_found": 10},
    "trail-camera": {"category": "real world", "train": 941, "test": 131, "valid": 239, "size": 1311, "num_classes": 2, "num_classes_found": 2},
    "cell-towers": {"category": "real world", "train": 705, "test": 101, "valid": 202, "size": 1008, "num_classes": 2, "num_classes_found": 2},
}

DATASETS_CATEGORIES = tuple(set(metadata["category"] for metadata in DATASETS_METADATA.values()))


# The number of classes from their doc is different from what we find in the dataset.
_NUM_CLASSES_FOUND = {
    "aerial-pool": 5,
    "secondary-chains": 1,
    "aerial-spheres": 6,
    "soccer-players-5fuqs": 3,
    "weed-crop-aerial": 2,
    "aerial-cows": 1,
    "cloud-types": 4,
    "apex-videogame": 2,
    "farcry6-videogame": 11,
    "csgo-videogame": 2,
    "avatar-recognition-nuexe": 1,
    "halo-infinite-angel-videogame": 4,
    "team-fight-tactics": 59,
    "robomasters-285km": 9,
    "stomata-cells": 2,
    "bccd-ouzjz": 3,
    "parasites-1s07h": 8,
    "cells-uyemf": 1,
    "4-fold-defect": 1,
    "bacteria-ptywi": 1,
    "cotton-plant-disease": 1,
    "mitosis-gjs3g": 1,
    "phages": 2,
    "liver-disease": 4,
    "asbestos": 4,
    "underwater-pipes-4ng4t": 1,
    "aquarium-qlnqy": 7,
    "peixos-fish": 2,
    "underwater-objects-5v7p8": 5,
    "coral-lwptl": 14,
    "tweeter-posts": 2,
    "tweeter-profile": 1,
    "document-parts": 2,
    "activity-diagrams-qdobr": 19,
    "signatures-xc8up": 1,
    "paper-parts": 19,
    "tabular-data-wf9uh": 12,
    "paragraphs-co84b": 7,
    "thermal-dogs-and-people-x6ejw": 2,
    "solar-panels-taxvb": 5,
    "radio-signal": 2,
    "thermal-cheetah-my4dp": 2,
    "x-ray-rheumatology": 12,
    "acl-x-ray": 1,
    "abdomen-mri": 1,
    "axial-mri": 2,
    "gynecology-mri": 3,
    "brain-tumor-m2pbp": 3,
    "bone-fracture-7fylg": 4,
    "flir-camera-objects": 4,
    "hand-gestures-jps7z": 14,
    "smoke-uvylj": 1,
    "wall-damage": 3,
    "corrosion-bi3q3": 3,
    "excavators-czvg9": 3,
    "chess-pieces-mjzgj": 13,
    "road-signs-6ih4y": 21,
    "street-work": 8,
    "construction-safety-gsnvb": 5,
    "road-traffic": 7,
    "washroom-rf1fa": 10,
    "circuit-elements": 31,
    "mask-wearing-608pr": 2,
    "cables-nl42k": 11,
    "soda-bottles": 3,
    "truck-movement": 5,
    "wine-labels": 12,
    "digits-t2eg6": 10,
    "vehicles-q0x2v": 12,
    "peanuts-sd4kf": 2,
    "printed-circuit-board": 23,
    "pests-2xlvx": 28,
    "cavity-rs0uf": 2,
    "leaf-disease-nsdsr": 3,
    "marbles": 2,
    "pills-sxdht": 8,
    "poker-cards-cxcvz": 53,
    "number-ops": 15,
    "insects-mytwu": 10,
    "cotton-20xz5": 4,
    "furniture-ngpea": 3,
    "cable-damage": 2,
    "animals-ij5d2": 10,
    "coins-1apki": 4,
    "apples-fvpl5": 2,
    "people-in-paintings": 1,
    "circuit-voltages": 6,
    "uno-deck": 15,
    "grass-weeds": 1,
    "gauge-u2lwv": 2,
    "sign-language-sokdr": 26,
    "valentines-chocolate": 22,
    "fish-market-ggjso": 19,
    "lettuce-pallets": 5,
    "shark-teeth-5atku": 4,
    "bees-jt5in": 1,
    "sedimentary-features-9eosf": 5,
    "currency-v4f8j": 10,
    "trail-camera": 2,
    "cell-towers": 2,
}


def _fetch_datasets_metadata():
    """Fetch the dataset statistics from the official roboflow repository. Combine it with the _NUM_CLASSES_FOUND to update it with the real number of classes.
    Note: This is a manual script to generate DATASETS_METADATA and is only for dev purpose. Call this only if the dataset statistics were changed."""

    import json
    import pandas as pd

    # Load the dataset_stats.csv from official repo: https://github.com/roboflow/roboflow-100-benchmark/blob/main/metadata/datasets_stats.csv
    # It includes some metadata about each of the dataset.
    df = pd.read_csv("https://raw.githubusercontent.com/roboflow/roboflow-100-benchmark/main/metadata/datasets_stats.csv")

    # Select only relevant columns
    df = df[["dataset", "category", "train", "test", "valid", "size", "num_classes"]]
    df["num_classes"] = df["num_classes"].astype(int)

    # Add information about the number of classes found in the dataset.
    df["num_classes_found"] = df.dataset.apply(lambda x: _NUM_CLASSES_FOUND.get(x, 0)).astype(int)

    return json.loads(df.set_index("dataset").to_json(orient="index"))
