import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to outputs folder
output_dir = os.path.join(os.getcwd(), "outputs")

def safe_read_excel(path):
    """Try reading excel if file exists and is valid"""
    try:
        df = pd.read_excel(path)
        print(df.head(10))  # show first 10 rows
    except Exception as e:
        print(f"Could not read {path}: {e}")

print("\n=== Sentiment Report ===\n")
report_file = os.path.join(output_dir, "sentiment_report.txt")
if os.path.exists(report_file):
    with open(report_file, "r") as f:
        print(f.read())
else:
    print("sentiment_report.txt not found")

print("\n=== Sentiment Counts ===\n")
safe_read_excel(os.path.join(output_dir, "sentiment_counts.xlsx"))

print("\n=== Test Predictions ===\n")
safe_read_excel(os.path.join(output_dir, "test_predictions.xlsx"))

print("\n=== Keyword Sentiment ===\n")
safe_read_excel(os.path.join(output_dir, "keyword_sentiment.xlsx"))

# Show images one by one
img_files = [
    "confusion_matrix.png",
    "sentiment_distribution.png",
    "top_words_neg.png",
    "top_words_neu.png",
    "top_words_pos.png"
]

for img in img_files:
    path = os.path.join(output_dir, img)
    if os.path.exists(path):
        print(f"\n[Opening image: {img}]")
        img_data = plt.imread(path)
        plt.imshow(img_data)
        plt.axis("off")
        plt.show()
    else:
        print(f"{img} not found")