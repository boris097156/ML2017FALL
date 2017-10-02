import matplotlib.pyplot as plt
import sys

DIR = 'log/'
LOG = DIR + 'log.txt'
PNG = DIR + 'log.png'
DPNG = DIR + 'log_d.png'
start_step = 10
detailed_step = int(sys.argv[1])

steps = []
lost = []

with open(LOG, 'r') as f:
	content = f.readlines()
	for row in content[3:]:
		line = row.strip().split('\t')
		steps.append(int(line[0]))
		lost.append(float(line[1]))

start_lost = lost[start_step:]
start_steps = steps[start_step:]

plt.xlabel("steps")
plt.ylabel("L")
plt.plot(start_steps, start_lost, color='r')
plt.tight_layout()
plt.savefig(PNG)

plt.close('all')

detailed_lost = lost[detailed_step:]
detailed_steps = steps[detailed_step:]

plt.xlabel("steps")
plt.ylabel("L")
plt.plot(detailed_steps, detailed_lost, color='r')
plt.tight_layout()
plt.savefig(DPNG)
