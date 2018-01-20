
import sys
SEPERATION_STR = ' $$$ '
# def prepSNLI():
# 	file_names = ['dev', 'test', 'train']
# 	labels_dict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}
#
# 	for filename in file_names:
# 		print ('preprossing ' + filename + '...')
# 		fpr = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
# 		count = 0
# 		fpr.readline()
# 		fpw = open('data/snli/processed_data/'+filename+'.txt', 'w')
# 		for line in fpr:
# 			sentences = line.strip().split('\t')
# 			if sentences[3] == '-':
# 				continue
#
# 			tokens = sentences[1].split(' ')
# 			tokens = [token for token in tokens if token != '(' and token != ')']
# 			fpw.write(' '.join(tokens)+'\t')
# 			fpw.write(SEPERATION_STR)
# 			tokens = sentences[2].split(' ')
# 			tokens = [token for token in tokens if token != '(' and token != ')']
# 			fpw.write(' '.join(tokens)+'\t')
# 			fpw.write(SEPERATION_STR)
# 			fpw.write(str(labels_dict[sentences[0]])+'\n')
# 			count += 1
# 		fpw.close()
# 		fpr.close()
#
# if __name__ == "__main__":
# 	prepSNLI()

def Multinli():
	file_names = ['train']
	labels_dict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}

	for filename in file_names:
		data_file = open('data/multinli/multinli_1.0/multinli_1.0_'+filename+'.txt', 'r')
		# i = 0
		data_file.readline()
		processed_file = open('data/multinli/sequence/'+filename+'.txt', 'w')
		for line in data_file:
			sentences = line.strip().split('\t')
			if sentences[0] == '-':
				continue
			values = sentences[1].split(' ')
			values = [t for t in values if t != '(' and t != ')']
			processed_file.write(' '.join(values)+'\t')
			processed_file.write(SEPERATION_STR)
			values = sentences[2].split(' ')
			values = [t for t in values if t != '(' and t != ')' ]
			processed_file.write(' '.join(values)+'\t')
			processed_file.write(SEPERATION_STR)
			processed_file.write(str(labels_dict[sentences[0]])+'\n')
			# i += 1
		processed_file.close()
		data_file.close()

if __name__ == "__main__":
	Multinli()