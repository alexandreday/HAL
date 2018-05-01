datafile = "001.csv"
data = pd.read_csv(datafile)
X = data.values
y = pd.read_csv("001_label.csv").values.flatten()
#print(y)
#exit()
# apply transformation to data. Note
# that CLUSTER does automatically zscore the data.

pos_0 = (y == 0)
fig, ax = plt.subplots(3,4,figsize=(8,6))
for i in range(3):
    for j in range(4):
        Xsub = X[pos_0]
        ax[i,j].hist(Xsub[:,4*i+j], bins=100, log=True, alpha=0.4)
        Xsub = X[(pos_0 == False)]
        ax[i,j].hist(Xsub[:,4*i+j], bins=100, log=True, alpha=0.4)

plt.tight_layout()
plt.show()
exit()