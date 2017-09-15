import sys

f = open(sys.argv[1], 'r')
dic = {}
line = f.read().strip('\n').split()
for word in line:
	if word in dic:
		dic[word] += 1
	else:
		dic[word] = 1
for no, word in enumerate(dic):
	print(word + ' ' + str(no) + '  ' + str(dic[word]))
