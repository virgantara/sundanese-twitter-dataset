import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

labels = ['TF-IDF', 'BoW', 'N-GRAM']
prec = [94, 90, 93]
rec = [98,94,95]
f1 = [96,92,94]

plotdata = pd.DataFrame({
    'TF-IDF' : [94,98,96],
    'BoW' : [90,94,95],
    'N-GRAM' : [93,95,94],
},
index=["Prec", "Rec", "F1"]
)
plotdata.plot(kind="bar")
plt.title("Performance in Anger Emotion")
plt.xlabel("Measurement")
plt.ylabel("%")

plt.show()