import csv
from pathlib import Path
from itertools import combinations, product


# --------------------------------
# Paths
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
OUTPUT_FILE = BASE_DIR / "pairs.csv"


# --------------------------------
# Find image files
# --------------------------------

image_files = list(IMAGE_DIR.glob("*.png"))


# --------------------------------
# Group images by person
# --------------------------------

groups = {}

for image_path in image_files:

    # Example:
    # modi_1.png → modi

    person = image_path.stem.rsplit("_", 1)[0]

    if person not in groups:
        groups[person] = []

    groups[person].append(image_path)


print("Found groups:")

for person, images in groups.items():

    print(
        f"{person}: {len(images)} images"
    )


# --------------------------------
# Create pairs
# --------------------------------

pairs = []


# =================================
# Genuine pairs
# =================================

for person, images in groups.items():

    for image1, image2 in combinations(images, 2):

        pairs.append({
            "image1": str(image1.relative_to(BASE_DIR.parent)),
            "image2": str(image2.relative_to(BASE_DIR.parent)),
            "label": "genuine"
        })


# =================================
# Impostor pairs
# =================================

people = list(groups.keys())


for person1, person2 in combinations(people, 2):

    for image1, image2 in product(
        groups[person1],
        groups[person2]
    ):

        pairs.append({
            "image1": str(image1.relative_to(BASE_DIR.parent)),
            "image2": str(image2.relative_to(BASE_DIR.parent)),
            "label": "impostor"
        })


# --------------------------------
# Save CSV
# --------------------------------

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["image1", "image2", "label"]
    )

    writer.writeheader()

    writer.writerows(pairs)


# --------------------------------
# Summary
# --------------------------------

genuine_count = sum(
    1 for pair in pairs
    if pair["label"] == "genuine"
)

impostor_count = sum(
    1 for pair in pairs
    if pair["label"] == "impostor"
)


print("\n==============================")
print("Pair Generation Complete")
print("==============================")

print(f"Total pairs: {len(pairs)}")
print(f"Genuine pairs: {genuine_count}")
print(f"Impostor pairs: {impostor_count}")

print(f"\n✓ Saved to: {OUTPUT_FILE}")
