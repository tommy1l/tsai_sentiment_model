import pandas as pd
from sklearn.metrics import accuracy_score

df_model = pd.read_csv("analyze_model_old.csv")
df_human = pd.read_csv("oldtweet50_human.csv")

df = df_model.merge(df_human, on='Tweets')

df = df.rename(columns={
    'Positivity': 'Pos_model',
    'Negativity': 'Neg_model',
    'Pos': 'Pos_human',
    'Neg': 'Neg_human'
})

df['Pos_model_rounded'] = df['Pos_model'].round()
df['Neg_model_rounded'] = df['Neg_model'].round()
df['Pos_human_rounded'] = df['Pos_human'].round()
df['Neg_human_rounded'] = df['Neg_human'].round()

total = len(df)
pos_correct = (df['Pos_model_rounded'] == df['Pos_human_rounded']).sum()
neg_correct = (df['Neg_model_rounded'] == df['Neg_human_rounded']).sum()
both_correct = ((df['Pos_model_rounded'] == df['Pos_human_rounded']) &
                (df['Neg_model_rounded'] == df['Neg_human_rounded'])).sum()

print("🔍 Accuracy Report")
print(f"Positivity accuracy: {pos_correct}/{total} ({pos_correct / total:.2%})")
print(f"Negativity accuracy: {neg_correct}/{total} ({neg_correct / total:.2%})")
print(f"Overall accuracy (both correct): {both_correct}/{total} ({both_correct / total:.2%})")

mismatches = df[
    (df['Pos_model_rounded'] != df['Pos_human_rounded']) |
    (df['Neg_model_rounded'] != df['Neg_human_rounded'])
]

mismatches.to_csv("mismatched_tweets.csv", index=False)

print("\n⚠️ Saved mismatched rows to 'mismatched_tweets.csv'")
