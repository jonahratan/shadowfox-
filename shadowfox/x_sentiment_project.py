import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

LABEL_MAP = {-1: "Negative", 0: "Neutral", 1: "Positive"}

STOPWORDS = {
    'the','is','are','a','an','of','to','and','in','on','for','with','at','by','from','it','this',
    'that','be','as','or','if','not','was','were','but','about','so','we','they','you','i','he','she',
    'them','their','our','your','have','has','had','will','would','can','could','should','do','did','done',
    'just','into','over','under','than','then','there','here','out','up','down','off','too','very','also','rt'
}

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', ' ', s)        # URLs
    s = re.sub(r'[@#]\w+', ' ', s)                 # mentions/hashtags
    s = re.sub(r'[^a-z\'\s]', ' ', s)              # punctuation/numbers
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def top_words(df_subset, n=20):
    counter = Counter()
    for t in df_subset['text_norm']:
        tokens = [w for w in t.split() if w not in STOPWORDS and len(w) > 2]
        counter.update(tokens)
    return pd.DataFrame(counter.most_common(n), columns=['word','count'])

def plot_bar(series_or_df, title, xlab, ylab, outpath, xrot=0):
    plt.figure()
    if isinstance(series_or_df, pd.Series):
        series_or_df.plot(kind='bar', rot=xrot, title=title)
    else:
        plt.bar(series_or_df.iloc[:,0], series_or_df.iloc[:,1])
        plt.title(title); plt.xlabel(xlab); plt.ylabel(ylab); plt.xticks(rotation=xrot)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main(csv_path, outdir, keywords):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = pd.read_csv(csv_path)[['clean_text','category']]
    df = df.dropna(subset=['clean_text','category']).copy()
    df['category'] = df['category'].astype(int)
    df['sentiment'] = df['category'].map(LABEL_MAP)

    # 2) Clean
    df['text_norm'] = df['clean_text'].astype(str).apply(normalize_text)

    # 3) EDA
    sent_counts = df['sentiment'].value_counts().reindex(['Positive','Neutral','Negative']).fillna(0).astype(int)
    sent_counts.to_csv(outdir/'sentiment_counts.csv')
    plot_bar(sent_counts, 'Sentiment Distribution', 'Sentiment', 'Count', outdir/'sentiment_distribution.png')

    # Top words per class
    top_pos = top_words(df[df['sentiment']=='Positive']); top_pos.to_csv(outdir/'top_words_positive.csv', index=False)
    top_neu = top_words(df[df['sentiment']=='Neutral']);  top_neu.to_csv(outdir/'top_words_neutral.csv',  index=False)
    top_neg = top_words(df[df['sentiment']=='Negative']); top_neg.to_csv(outdir/'top_words_negative.csv', index=False)
    plot_bar(top_pos, 'Top Words in Positive Tweets', 'Word','Count', outdir/'top_words_pos.png', xrot=90)
    plot_bar(top_neu, 'Top Words in Neutral Tweets',  'Word','Count', outdir/'top_words_neu.png', xrot=90)
    plot_bar(top_neg, 'Top Words in Negative Tweets', 'Word','Count', outdir/'top_words_neg.png', xrot=90)

    # Optional: topic/keyword sentiment slices
    if keywords:
        rows = []
        for kw in keywords:
            subset = df[df['text_norm'].str.contains(rf'\b{re.escape(kw.lower())}\b', regex=True)]
            counts = subset['sentiment'].value_counts()
            rows.append({
                'keyword': kw,
                'n_tweets': int(subset.shape[0]),
                'positive': int(counts.get('Positive',0)),
                'neutral':  int(counts.get('Neutral',0)),
                'negative': int(counts.get('Negative',0))
            })
        pd.DataFrame(rows).to_csv(outdir/'keyword_sentiment.csv', index=False)

    # 4) Train/Test split + Model
    X = df['text_norm'].values
    y = df['category'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,1), min_df=5, max_df=0.9, max_features=100_000)),
        ('clf', LinearSVC(dual='auto'))
    ])
    model.fit(X_train, y_train)

    # 5) Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=['Negative','Neutral','Positive'], digits=4)

    cm = confusion_matrix(y_test, y_pred, labels=[-1,0,1])
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (LinearSVC)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(ticks=[0,1,2], labels=['Negative','Neutral','Positive'])
    plt.yticks(ticks=[0,1,2], labels=['Negative','Neutral','Positive'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.tight_layout()
    plt.savefig(outdir/'confusion_matrix.png')
    plt.close()

    # Save example predictions
    pred_df = pd.DataFrame({
        'text': X_test,
        'true_label': [LABEL_MAP[i] for i in y_test],
        'pred_label': [LABEL_MAP[i] for i in y_pred]
    })
    pred_df.to_csv(outdir/'test_predictions.csv', index=False)

    # Save a short report
    report_txt = f"""# Sentiment Analysis Report

Dataset size: {len(df)} tweets

Class distribution:
Positive: {int(sent_counts.get('Positive',0))}
Neutral:  {int(sent_counts.get('Neutral',0))}
Negative: {int(sent_counts.get('Negative',0))}

Model: TF-IDF (uni-grams) + LinearSVC
Test Accuracy: {acc:.4f}
Macro F1: {f1m:.4f}

Classification Report:
{report}
"""
    (outdir/'sentiment_report.txt').write_text(report_txt, encoding='utf-8')
    print(report_txt)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="X (Twitter) Sentiment Analysis")
    p.add_argument('--csv', default='X_data.csv', help='Path to dataset CSV')
    p.add_argument('--out', default='outputs', help='Directory to save charts & reports')
    p.add_argument('--keywords', nargs='*', default=[], help='Optional keywords for topic slices, e.g. --keywords modi bjp congress')
    args = p.parse_args()
    main(args.csv, args.out, args.keywords)
