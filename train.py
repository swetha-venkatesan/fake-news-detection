import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# 1. Load and label the data

fake = pd.read_csv(r'D:\Appro_Project_1(image_classification)\fake_news_detection(1)\dataset\Fake.csv', on_bad_lines='skip')
true = pd.read_csv(r'D:\Appro_Project_1(image_classification)\fake_news_detection(1)\dataset\True.csv', on_bad_lines='skip')


fake['label'] = 'FAKE'
true['label'] = 'REAL'

# 2. Combine datasets
df = pd.concat([fake, true], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# 3. Use 'text' column if available
text_column = 'text' if 'text' in df.columns else 'content'

x = df[text_column]
y = df['label']

# 4. Split, vectorize, and train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x_train_vect = vectorizer.fit_transform(x_train)
x_test_vect = vectorizer.transform(x_test)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_vect, y_train)

# 5. Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved as model.pkl and vectorizer.pkl")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# import pickle

# data = {
#     "text":["the economy is doing great,claims government",
#             "breaking:aliens have landed in new york",
#             "vaccines proven effecctive,new study shows",
#             "celebrirty says the moon is hollow",
#             "NASA confirms marrs covers discovery",
#             "Fake news site claims earth is flat"],
#     "label":["REAL","FAKE","REAL","FAKE","REAL","FAKE"]
# }
# df = pd.DataFrame(data)

# x=df["text"]
# y=df["label"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# vectorized = TfidfVectorizer(stop_words="english", max_df=0.7)
# x_train_vect = vectorized.fit_transform(x_train)    
# x_test_vect = vectorized.transform(x_test)

# model = PassiveAggressiveClassifier(max_iter=50)
# model.fit(x_train_vect,y_train )   

# pickle.dump(model, open("model.pkl", "wb"))
# pickle.dump(vectorized, open("vectorizer.pkl","wb"))