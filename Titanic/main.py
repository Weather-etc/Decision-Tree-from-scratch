import pandas as pd
from ID3.id3 import ID3tree

from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")
# select training data
y = train_data['Survived']
features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
X = train_data[features]
# select test data
X_test = test_data[features]

model = ID3tree(X, y, 0.6)
model.fit()
# fit test data & output to csv file
res0 = model.predict(X_test)
res1 = [i[0] for i in res0]
res = map(int, res1)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': res})
output.to_csv('./submission.csv', index=False)

"""
y = train_data['Survived']
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('./submission.csv', index=False)
print("Finished")
"""
