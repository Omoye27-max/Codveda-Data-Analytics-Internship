import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
df=pd.read_csv("Level_3/cleaned_data.csv")
print("First 5 Rows")
print(df.head())
print("\nDataset Information")
print(df.info())
print("\nMissing Values")
print(df.isnull().sum())
print("\nColumn names:")
print (df.columns.tolist())
y= df["Sentiment"]
print("\nNumber of unique sentiment classes:")
print(df["Sentiment"].nunique())
print("\nSentiment categories:")
print(sorted(df["Sentiment"].unique()))
print("\nSentiment frequency:")
print(df["Sentiment"].value_counts())
print("\nAll Sentiment Categories:")
print(sorted(df["Sentiment"].unique()))
df["Sentiment"] = (df["Sentiment"].astype(str).str.strip().str.lower())
print("\nCleaned Sentiment Categories:")
print(df["Sentiment"].value_counts())
print("\nNumber of unique sentiment categories after cleaning:")
print(df["Sentiment"].nunique())
print("\nAll 191 Sentiment Categories:")
print(sorted(df["Sentiment"].unique()))
positive_sentiments = [  "acceptance", "accomplishment", "admiration", "adoration",
"adrenaline", "adventure", "affection", "amazement", "amusement",
    "anticipation", "appreciation", "artisticburst", "awe", "blessed",
    "breakthrough", "calmness", "captivation", "celebration", "charm",
    "colorful", "compassion", "compassionate", "confidence", "confident",
    "connection", "contentment", "coziness", "creative inspiration",
    "creativity", "culinary adventure", "culinaryodyssey", "curiosity",
    "dazzle", "determination", "ecstasy", "elation", "elegance",
    "empowerment", "enchantment", "energy", "engagement", "enjoyment",
    "enthusiasm", "euphoria", "excitement", "exploration",
    "festivejoy", "free-spirited", "freedom", "friendship",
    "fulfillment", "grandeur", "grateful", "gratitude", "happiness",
    "happy", "harmony", "heartwarming", "hope", "hopeful", "hypnotic",
    "imagination", "immersion", "inspiration", "inspired", "intrigue",
    "joy", "joy in baking", "joyfulreunion", "kind", "kindness",
    "love", "marvel", "melodic", "mesmerizing", "mindfulness",
    "motivation", "nature's beauty", "optimism", "overjoyed",
    "playful", "playfuljoy", "positive", "positivity", "pride",
    "proud", "radiance", "rejuvenation", "relief", "renewed effort",
    "resilience", "reverence", "romance", "satisfaction", "serenity",
    "solace", "spark", "success", "surprise", "sympathy", "tenderness",
    "thrill", "thrilling journey", "touched", "tranquility", "triumph",
    "vibrancy", "whimsy", "wonder", "wonderment", "zest"]
negative_sentiments = ["anger", "anxiety", "apprehensive", "bad", "betrayal", "bitter",
    "bitterness", "bittersweet", "boredom", "confusion", "darkness",
    "desolation", "despair", "desperation", "devastated", "disappointed",
    "disappointment", "disgust", "dismissive", "embarrassed",
    "emotionalstorm", "envious", "envy", "exhaustion", "fear",
    "fearful", "frustrated", "frustration", "grief", "hate",
    "heartache", "heartbreak", "helplessness", "intimidation",
    "isolation", "jealous", "jealousy", "loneliness", "loss",
    "lostlove", "melancholy", "miscalculation", "numbness",
    "overwhelmed", "pressure", "regret", "resentment", "sad",
    "sadness", "shame", "solitude", "sorrow", "suffering", "yearning"]
neutral_sentiments = ["ambivalence", "challenge", "contemplation", "emotion",
    "envisioning history", "innerjourney", "indifference",
    "journey", "nostalgia", "obstacle", "ocean's freedom",
    "pensive", "reflection", "ruins", "runway creativity",
    "suspense", "whispers of the past", "winter magic", "neutral"]
def group_sentiment(sentiment):
    if sentiment in positive_sentiments:
        return "Positive"
    elif sentiment in negative_sentiments:
        return "Negative"
    else:
        return "Neutral"
df["Sentiment_Group"] = df["Sentiment"].apply(group_sentiment)
print("\nGrouped Sentiment Distribution:")
print(df["Sentiment_Group"].value_counts())
X = df["Text"].astype(str)
y = df["Sentiment_Group"]
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_tfidf = vectorizer.fit_transform(X)
print("\nText successfully converted to numerical features")
print("Feature matrix shape:", X_tfidf.shape)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf,y,test_size=0.2,random_state=42,stratify=y)
print("\nData split successfully")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nModel trained successfully!")
y_pred = model.predict(X_test)
print("\nPredictions:")
print(y_pred[:10])
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.title("Sentiment Classification Confusion Matrix")
plt.tight_layout()
plt.show()
df.to_csv("sentiment_grouped_cleaned.csv", index=False)
print("\nGrouped sentiment dataset saved successfully!")
plt.figure(figsize=(7, 5))
df["Sentiment_Group"].value_counts().plot(kind="bar")
plt.title("Grouped Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
df.to_csv("Level3_Task3_Final_Cleaned_Sentiment_Data.csv", index=False)
print("Final cleaned sentiment dataset saved successfully!")