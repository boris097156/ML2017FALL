import matplotlib.pyplot as plt
import sys

f_log = 'log.txt'
start_step = 10
detailed_step = 5000

steps = []
lost = []

with open(f_log, 'r') as f:
	content = f.readlines()
	for row in content[2:]:
		line = row.strip().split('\t')
		steps.append(int(line[0]))
		lost.append(float(line[1]))

start_lost = lost[start_step:]
start_steps = steps[start_step:]

plt.xlabel("steps")
plt.ylabel("L")
plt.plot(start_steps, start_lost, color='r')
plt.tight_layout()
plt.savefig('log.png')

plt.close('all')

detailed_lost = lost[detailed_step:]
detailed_steps = steps[detailed_step:]

plt.xlabel("steps")
plt.ylabel("L")
plt.plot(detailed_steps, detailed_lost, color='r')
plt.tight_layout()
plt.savefig('log_detailed.png')