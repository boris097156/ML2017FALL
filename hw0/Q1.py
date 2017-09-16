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
	index = 0
	for word in line:
		if word in dic:
			out.write(word + ' ' + str(index) + ' ' + str(dic[word]) + '\n')
			index += 1
			del dic[word]
f.close()
