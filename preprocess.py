import sys

SEPERATION_STR = ' $$$ '
def create_snli():
	file_names = ['dev', 'test', 'train']
	labels_dict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}

	for filename in file_names:
		data_file = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
		data_file.readline()
		processed_file = open('data/snli/processed_data/'+filename+'.txt', 'w')
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
		processed_file.close()
		data_file.close()



def create_multinli():
	file_names = ['train']
	labels_dict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}

	for filename in file_names:
		data_file = open('data/multinli/multinli_1.0/multinli_1.0_'+filename+'.txt', 'r')
		data_file.readline()
		processed_file = open('data/multinli/processed_data/'+filename+'.txt', 'w')
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
		processed_file.close()
		data_file.close()

if __name__ == "__main__":
	create_multinli()
	create_snli()