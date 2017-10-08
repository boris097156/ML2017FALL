import sys
import matplotlib.pyplot as plt
from my_function import LOG_NUM

DIR = sys.argv[1] + '/log/'
PNG = DIR + 'log_graph.png'
my_color = ['k', 'g', 'b', 'r']

def read_log(steps, rmse, log_name):
	with open(log_name, 'r') as f:
		content = f.readlines()
		for row in content:
			line = row.strip().split('\t')
			steps.append(int(line[0]))
			rmse.append(float(line[1]))

def add_graph():
	steps = [[] for _ in range(LOG_NUM+1)]
	rmse = [[] for _ in range(LOG_NUM+1)]
	#record each log rmse
	for i in range(LOG_NUM):
		log_name = DIR + 'log' + str(i) + '.txt'
		read_log(steps[i], rmse[i], log_name)
		plt.plot(steps[i], rmse[i], color=my_color[i])
	#record avg rmse
	for i in range(len(steps[0])):
		rmse_sum = 0
		for j in range(LOG_NUM):
			rmse_sum += rmse[j][i]
		rmse[LOG_NUM].append(rmse_sum/LOG_NUM)
	plt.plot(steps[0], rmse[LOG_NUM], color=my_color[LOG_NUM])

plt.figure(figsize=(18,8))
plt.xlabel("steps")
plt.ylabel("RSME")
add_graph()
plt.tight_layout()
plt.savefig(PNG)
