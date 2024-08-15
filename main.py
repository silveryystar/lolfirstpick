import requests
import json
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def f(x, m, b):
    return m*x+b


def show(xlabel, ylabel, title):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def sub_show(xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


response = requests.get("https://leaguephd.com/en/stats/")
content = response.content
content = str(content)

start = content.find('[{"')
stop = content.find('}]', start)
data = ""

for i in range(start, stop+2):
    data += content[i]

while "\\" in data:
    data = data.replace("\\", "")

data = json.loads(data)

win_pct = [i["win_pct"] for i in data]
games = [np.log10(i["games"]) for i in data]
fp_value = [i["fp_value"] for i in data]
lane = [i["champion"]["lane"] for i in data]

m, b = np.polyfit(win_pct, fp_value, 1)

predict = [f(i, m, b) for i in win_pct]
r2 = r2_score(fp_value, predict)

n = np.linspace(min(win_pct), max(win_pct), 1000)

plt.plot(n, f(n, m, b), color="black", label=f"y={round(m, 1)}x{round(b, 1)}, R²={round(r2, 3)}")
plt.scatter(win_pct, fp_value, c=games, cmap="rainbow")
plt.colorbar(label="log10(Games)")

show("WR", "FPS", "FPS-WR Game Analysis")


lanes = ["Top", "Jungle", "Mid", "ADC", "Support"]
wps, fps = [], []
for i in lanes:
    wp = [win_pct[j] for j in range(len(lane)) if lane[j] == i]
    fp = [fp_value[j] for j in range(len(lane)) if lane[j] == i]

    wps.append(wp)
    fps.append(fp)

    m, b = np.polyfit(wp, fp, 1)

    plt.scatter(wp, fp, label=i)
    plt.plot(n, f(n, m, b))

show("WR", "FPS", "FPS-WR Role Analysis")


plt.subplot(1, 2, 1)

plt.violinplot(wps, showmeans=True)
plt.boxplot(wps)

sub_show("Role", "WR")

plt.subplot(1, 2, 2)

plt.violinplot(fps, showmeans=True)
plt.boxplot(fps)

sub_show("Role", "FPS")

plt.suptitle("WR- & FPS-Role Violins & Boxes")
plt.tight_layout()
plt.show()
