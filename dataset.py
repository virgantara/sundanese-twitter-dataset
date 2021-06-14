import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
dataset = pd.read_csv('newdataset.csv')
mood_count=dataset['label'].value_counts()
sns.countplot(x='label',data=dataset,order=['anger','joy','fear','sadness'])
plt.show()
print(mood_count)