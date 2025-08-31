from flask import Flask, render_template_string, send_from_directory
import pandas as pd
import os

app = Flask(__name__)
output_dir = os.path.join(os.getcwd(), "outputs")

@app.route("/")
def home():
    # Load text
    txt = "No report found."
    report_file = os.path.join(output_dir, "sentiment_report.txt")
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            txt = f.read()

    # Load Excel tables
    counts = pd.read_excel(os.path.join(output_dir, "sentiment_counts.xlsx")).to_html(index=False)
    preds = pd.read_excel(os.path.join(output_dir, "test_predictions.xlsx")).head(10).to_html(index=False)
    keywords = pd.read_excel(os.path.join(output_dir, "keyword_sentiment.xlsx")).to_html(index=False)

    # Simple HTML template
    html = f"""
    <h1>📊 Sentiment Project Showcase</h1>

    <h2>Sentiment Report</h2>
    <pre>{txt}</pre>

    <h2>Sentiment Counts</h2>
    {counts}

    <h2>Test Predictions (Top 10)</h2>
    {preds}

    <h2>Keyword Sentiment</h2>
    {keywords}

    <h2>Images</h2>
    <img src="/outputs/confusion_matrix.png" width="400"><br>
    <img src="/outputs/sentiment_distribution.png" width="400"><br>
    <img src="/outputs/top_words_neg.png" width="400"><br>
    <img src="/outputs/top_words_neu.png" width="400"><br>
    <img src="/outputs/top_words_pos.png" width="400"><br>
    """
    return render_template_string(html)

@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    return send_from_directory(output_dir, filename)

if __name__ == "__main__":
    app.run(debug=True)