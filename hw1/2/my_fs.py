FEATURE_AMOUNT = 18

# calculate feature weight for all 18 items from all training data
def cal_weight(train_data):
	fs_weight = [[] for _ in range(FEATURE_AMOUNT)]
	train_size = len(train_data[1])
	for i in range(FEATURE_AMOUNT):
		my_sum = 0.0
		for j in range(train_size):
			my_sum += train_data[i][j]
		my_avg = my_sum/(train_size)
		my_squ = 0.0;
		for j in range(train_size):
			my_squ += (train_data[i][j]-my_avg)**2
		my_sd = (my_squ/train_size)**0.5
		fs_weight[i].append(my_avg)
		fs_weight[i].append(my_sd)
	return fs_weight

def scale_down(my_data, fs_weight):
	data_size = len(my_data[1])
	for i in range(FEATURE_AMOUNT):
		for j in range(data_size):
			my_data[i][j] -= fs_weight[i][0]
			my_data[i][j] /= fs_weight[i][1]
	return

def scale_up(my_data, fs_weight):
	data_size = my_data.shape[0]
	for i in range(data_size):
		my_data[i] *= fs_weight[9][1]
		my_data[i] += fs_weight[9][0]
	return

