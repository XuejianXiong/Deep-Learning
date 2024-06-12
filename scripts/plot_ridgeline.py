"""
import numpy as np
from ridgeplot import ridgeplot

my_samples = [np.random.normal(n / 1.2, size=600) for n in range(100, 0, -1)]
#print(my_samples)
fig = ridgeplot(samples=my_samples)
fig.update_layout(height=45*100, width=800)
fig.show()
"""



import pandas as pd
from joypy import joyplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(21)

df = pd.read_csv("data/temp.csv", sep="\t")

print(df.info())


##Trying to create a colormap 
norm = plt.Normalize(df["PRECURSORABUNDANCE"].min(), df["PRECURSORABUNDANCE"].max())
ar = np.array(df["PRECURSORABUNDANCE"])

original_cmap = plt.cm.viridis
cmap = matplotlib.colors.ListedColormap(original_cmap(norm(ar)))
sm = matplotlib.cm.ScalarMappable(cmap=original_cmap, norm=norm)
sm.set_array([])

fig, axes = joyplot(
      df, 
      by="RUN", 
      column="PRECURSORABUNDANCE", 
      #figsize=(5,8),
      linewidth=0.05,
      #overlap=5,
      #colormap=plt.cm.hsv,
      colormap=cmap,
      #x_range=[0,20]
)
fig.colorbar(sm, ax=axes, label="RUN")
plt.xlabel("PRECURSORABUNDANCE")
plt.text(
      40, 
      0.8, 
      f"Precursor Abundance (log2) In Runs",
      fontsize=12
)

plt.show()
