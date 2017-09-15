import sys

f = open(sys.argv[1], 'r')
dic = {}
output_name = 'Q1.txt'

line = f.read().strip('\n').split()
for word in line:
	if word in dic:
		dic[word] += 1
	else:
		dic[word] = 1
with open(output_name, 'w') as out:
	for no, word in enumerate(dic):
		out.write(word + ' ' + str(no) + '  ' + str(dic[word]) + '\n')
