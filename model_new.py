import requests
import time
import pandas as pd

df_tweets = pd.read_csv("newtweet50.csv", usecols=[0])

api_key = "api_key_here"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

positivity_scores = []
negativity_scores = []

for i in range(len(df_tweets)):
    quote = df_tweets.iloc[i, 0]

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are a psychologist analyzing the emotional tone of tweets written in Mandarin Chinese."
            },
            {
                "role": "user",
                "content": (
                    "You are a psychologist analyzing the affect of tweets written in Mandarin Chinese.\n\n"
                    "Please rate the sentiment of the following tweet using two separate and independent dimensions:\n"
                    "- Positivity: rate from 1 (not at all positive) to 5 (extremely positive)\n"
                    "- Negativity: rate from 1 (not at all negative) to 5 (extremely negative)\n\n"
                    "Respond with two numbers only, separated by a comma. No additional text or explanation necessary.\n\n"
                    f"Tweet: {quote}"
                )
            }
        ]
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        response.raise_for_status()

        output = response.json()["choices"][0]["message"]["content"].strip()
        pos_str, neg_str = output.split(",")
        pos = int(pos_str.strip())
        neg = int(neg_str.strip())

    except Exception as e:
        print(f"Error processing tweet {i}: {e}")
        pos, neg = None, None

    positivity_scores.append(pos)
    negativity_scores.append(neg)

    time.sleep(2.5)

df_tweets["Positivity"] = positivity_scores
df_tweets["Negativity"] = negativity_scores
df_tweets.to_csv("analyze_new.csv", index=False)
