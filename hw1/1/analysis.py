import matplotlib.pyplot as plt
import sys

DIR = 'log/'
PNG = DIR + 'log_graph.png'
my_color = ['r', 'g', 'b']

def add_graph(i):
	steps = []
	rmse = []
	log_name = DIR + 'log' + str(i) + '.txt'
	with open(log_name, 'r') as f:
		content = f.readlines()
		for row in content[6:]:
			line = row.strip().split('\t')
			steps.append(int(line[0]))
			rmse.append(float(line[1]))
	plt.plot(steps, rmse, color=my_color[i])

plt.figure(figsize=(18,8))
plt.xlabel("steps")
plt.ylabel("RSME")
for i in range(3):
	add_graph(i)
plt.tight_layout()
plt.savefig(PNG)
