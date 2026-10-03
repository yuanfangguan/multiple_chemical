with open("../../data/raw/Morgan_fingerprints.csv", 'rb') as f:
    for i, line in enumerate(f):
        try:
            line.decode('utf-8')
        except UnicodeDecodeError as e:
            print(f"Decoding error on line {i + 1}: {e}")
            break
