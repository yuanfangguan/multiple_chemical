import csv

# === File paths ===
INPUT_FILE = "../../data/raw/TASK2_Train_mixture_Dataset.csv"
STIMULUS_MAP_PATH = "../../data/raw/TASK2_Stimulus_definition.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


# Helper function to count valid component IDs in the stimulus definition
def count_components(components_str):
    comps = str(components_str).split(";")
    valid_comps = [c.strip() for c in comps if c.strip().isdigit()]
    return len(valid_comps)


# --- Step 1: Map Stimulus IDs to their component counts ---
stim2count = {}

print("🔍 Loading stimulus definitions...")
with open(STIMULUS_MAP_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        stim_id = row["id"]
        num_chems = count_components(row["components"])
        stim2count[stim_id] = num_chems

# --- Step 2: Split the dataset based on component counts ---
print("\n📝 Splitting dataset (Train = 2-drug mixtures, Test = all others)...")
with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(
    TRAIN_FILE, "w", newline="", encoding="utf-8"
) as train_out, open(TEST_FILE, "w", newline="", encoding="utf-8") as test_out:

    reader = csv.DictReader(infile)

    writer_train = csv.DictWriter(train_out, fieldnames=reader.fieldnames)
    writer_test = csv.DictWriter(test_out, fieldnames=reader.fieldnames)

    writer_train.writeheader()
    writer_test.writeheader()

    train_count = 0
    test_count = 0

    for row in reader:
        stim_id = row["stimulus"]

        # Drop the outlier to match your prediction script environment
        if stim_id == "AN873":
            continue

        is_train = False
        # Check how many chemicals are in this stimulus
        if stim_id in stim2count:
            if stim2count[stim_id] == 2:
                is_train = True

        if is_train:
            writer_train.writerow(row)
            train_count += 1
        else:
            writer_test.writerow(row)
            test_count += 1

print("\n=== Split Completed Successfully ===")
print(f"📝 Train set saved ({train_count} rows containing exactly 2 drugs)")
print(f"📝 Test set saved ({test_count} rows containing single components or 3+ drugs)")
