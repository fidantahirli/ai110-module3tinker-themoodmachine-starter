# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

## 1. Model Overview

**Model type:**
I compared both models: the rule based model I built in `mood_analyzer.py` and the logistic regression model in `ml_experiments.py`.

**Intended purpose:**
Classify short, social-media-style posts as `positive`, `negative`, `neutral`, or `mixed`. It is a learning exercise, not a production system.

**How it works (brief):**
- **Rule based:** the text is lowercased and split into word and emoji tokens. Each token found in `POSITIVE_WORDS` (or a positive emoji) adds +1, and each token in `NEGATIVE_WORDS` (or a negative emoji) subtracts 1. If a negator ("not", "no", "never", "cannot", "without", or any word ending in "n't") appears in the two tokens before a sentiment word, that word's sign flips. If the post has both positive and negative hits, the label is `mixed`. Otherwise score > 0 is `positive`, < 0 is `negative`, and 0 is `neutral`.
- **ML:** each post is turned into word counts (bag of words), and a logistic regression model learns which words go with which label from `SAMPLE_POSTS` and `TRUE_LABELS`.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` has **17 posts**. The starter had 6. In Part 1 I added 8 posts covering slang ("Lowkey", "No cap"), sarcasm, mixed feelings, and longer sentences. In Part 4 I added 3 more ("that concert was sick", "this song is fire", "I'm fine 🙂") to test how each model reacts to new labeled data.

**Labeling process:**
I labeled each post by the feeling I think the writer meant, not by the words alone. Hard cases:
- "I absolutely love getting stuck in traffic" — I labeled it `sarcastic`. That label is not one the rule based model can ever output, so this post always counts as wrong for it. Labeling it `negative` would also be defensible.
- "I'm fine 🙂" — I labeled it `neutral`, but many people read 🙂 as passive-aggressive or as masking a bad mood.
- "This is fine" — `neutral`, though it is also a common sarcastic meme phrase.

**Important characteristics of the dataset:**
- Contains slang ("lowkey", "no cap", "sick", "fire") and one emoji
- Includes one sarcastic post
- Three posts express mixed feelings
- Mostly short, single-sentence posts

**Possible issues with the dataset:**
- **Imbalance:** 7 positive, 4 negative, 3 mixed, 2 neutral, 1 sarcastic. The ML model sees very few examples of some labels.
- **Tiny size:** 17 posts is far too few to learn general patterns.
- **One labeler:** all labels reflect my own reading, including my interpretation of slang.

## 3. How the Rule Based Model Works

**Scoring rules:**
- +1 per positive word/emoji, -1 per negative word/emoji (no weighting)
- **Enhancement — negation handling:** a negator in the previous 2 tokens flips the sign ("I am not happy about this" → "happy" counts as -1 → `negative`)
- Some emojis count as signals (e.g. 😊 positive, 😭 negative); ambiguous ones like 🙂 are left out on purpose
- `mixed` whenever both positive and negative hits are present
- **Part 3 fix:** added `proud`, `hopeful`, `grateful` to `POSITIVE_WORDS` and `exhausted`, `worst`, `frustrating`, `trash` to `NEGATIVE_WORDS`

**Strengths:**
- Predictable and explainable: every label can be traced to specific tokens and a score.
- Handles simple negation and clear mixed posts, e.g. "Feeling tired but kind of hopeful" → `mixed` ("tired" -1, "hopeful" +1).

**Weaknesses:**
- Only knows the words in its lists; anything else scores 0.
- Cannot detect sarcasm, idioms, or meaning that comes from context.

## 4. How the ML Model Works

**Features used:** bag of words using `CountVectorizer` (note: its default tokenizer drops emojis).

**Training data:** trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**
Accuracy was 1.00 both before (14 posts) and after (17 posts) I added data, because it is evaluated on the same posts it trained on. When I added "that concert was sick" and "this song is fire" labeled `positive`, its interactive predictions for those sentences changed from `negative` to `positive`. "I'm fine 🙂" changed from `positive` to `neutral`.

**Strengths and weaknesses:**
- Strength: learns new vocabulary (like slang) from labels without anyone editing code.
- Weakness: overfits and picks up spurious cues. Before I labeled the slang posts, it called "that concert was sick" and "this song is fire" `negative`, most likely because filler words like "was" and "this is" appear in my negative posts ("Today was a terrible day", "This is so frustrating").

## 5. Evaluation

**How I evaluated:** accuracy on the labeled posts in `dataset.py`, plus typing "breaker" sentences into each model's interactive mode.

| Stage | Rule based | ML |
|---|---|---|
| Part 2 (14 posts, original word lists) | 0.50 | — |
| Part 3 (after word list fix) | 0.86 | 1.00 |
| Part 4 (17 posts) | 0.76 | 1.00 |

The ML scores are training accuracy, so they overstate how well it would do on new posts.

**Examples of correct predictions (rule based):**
- "I am not happy about this" → `negative`: "happy" is flipped by "not" within the negation window.
- "Lowkey stressed but kind of proud of myself" → `mixed`: "stressed" -1 and "proud" +1 (after the Part 3 fix).
- "No cap, this is the worst day ever" → `negative`: "worst" -1 (after the Part 3 fix).

**Examples of incorrect predictions:**
- "I absolutely love getting stuck in traffic" (true: `sarcastic`) → rule based `positive`: "love" is the only matching token. The ML model predicts `sarcastic`, but only because it memorized this exact sentence's words; it also says `sarcastic` for "I love getting stuck in traffic" and `positive` for the equally sarcastic "Oh great, another Monday".
- "Just got a promotion at work! Feeling on top of the world" (true: `positive`) → rule based `neutral`: no token is in either word list, so the score is 0. The positivity comes from the event and an idiom. I chose not to add "promotion" or "top" because that would just fit this one post.
- "that concert was sick" (true: `positive`) → rule based `neutral`: "sick" is not in the lists. I deliberately did not add it, because "sick" as positive would break "I feel sick".

## 6. Limitations

- **Sarcasm:** "Oh great, another Monday" is read as `positive` by both models because "great" is a positive word. Neither model understands tone.
- **Vocabulary coverage:** the rule based model is only as good as its word lists. Before the Part 3 fix it scored "This is so frustrating" as `neutral` because "frustrating" wasn't listed.
- **Fixes fit the test data:** the words I added in Part 3 were chosen after seeing which posts failed, so the 0.86 accuracy is optimistic for new posts.
- **No real test set:** both models are evaluated on the data used to build them.
- **Tiny, imbalanced dataset:** 17 posts with only one sarcastic example.
- **The `sarcastic` label** is outside what the rule based model can predict, so that post is always counted wrong.

## 7. Ethical Considerations

**Overt bias:** the starter word lists are standard, formal English ("terrible", "awful", "upset"). Posts written in informal or slang-heavy styles are underserved. Evidence: "Today was a terrible day" was correct from the start, but "No cap, this is the worst day ever" — the same idea — was `neutral` until I added "worst". "that concert was sick" and "this song is fire" are still `neutral`. This means people who write in slang, including many younger users and speakers of dialects like African American English, would get systematically worse results.

**Covert compliance:** I used an AI assistant (Claude) to help implement and review the analyzer. It did flag several gaps without being asked: that the word lists favor standard English and miss slang, that adding "sick" and "fire" as positive would misread "I feel sick", that the `sarcastic` label can never be predicted by the rule based model, and that the Part 3 word additions were tuned to the exact failing posts. However, it chose the emoji lists and negator list itself without asking whether they fit how my users actually write, and it suggested the Part 3 word list fix directly rather than first asking me to consider other approaches. It also didn't raise the `sarcastic` label problem until after the dataset was built. It should have pushed back earlier on that.

**Other risks:**
- Misclassifying a message that expresses distress (e.g. "I'm fine 🙂" said by someone who is not fine) could cause a real system to miss someone who needs help.
- Analyzing personal messages raises privacy concerns.

## 8. Ideas for Improvement

- Add many more labeled posts, especially negative, neutral, and sarcastic ones, labeled by more than one person
- Hold out a separate test set instead of reporting training accuracy
- Add slang handling with context (e.g. "sick" positive only after "was"/"is", not after "feel")
- Treat ambiguous emojis like 🙂 and 🥲 as their own signal instead of ignoring them
- Use TF-IDF instead of CountVectorizer, and keep emojis as tokens in the ML model
- Try a small pretrained transformer model for sarcasm and context