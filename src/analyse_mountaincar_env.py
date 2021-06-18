import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def height(x):
    return 0.45 * np.sin(3*x) + 0.55


sns.set_style("darkgrid")

x_min = -1.2
x_max = 0.6
m = 1
g = 1
X = np.linspace(x_min, x_max, 100)

Y = height(X)
F_gs = - np.sin(np.arctan(1.35*np.cos(3*X))) * m * g
F_gs_gym = - np.cos(3*X) * m * g

fig = plt.figure(figsize=(6, 8))
axes = fig.subplots(nrows=2, ncols=1, sharex=True)

ax = axes[0]
ax = sns.lineplot(x=X, y=Y, ax=ax)
ax.set_ylabel("y")
ax.set_title("Height")

ax = axes[1]
data = pd.DataFrame([X, F_gs, F_gs_gym]).T
data.columns = ["x", "analytical", "gym approximation"]
data = data.melt(id_vars=["x"])
data = data.rename(columns={"variable": "source"})
ax = sns.lineplot(data=data, x="x", y="value", hue="source", ax=ax)
ax.set_xlabel("x")
ax.set_ylabel("F_gs")
ax.set_title("Gravity force in direction of trajectory s")

fig.set_tight_layout(True)

plt.show()
