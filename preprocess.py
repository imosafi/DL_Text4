
import sys
SEPERATION_STR = ' $$$ '
def prepSNLI():
	filenames = ['dev', 'test', 'train']
	labelDict = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}

	for filename in filenames:
		print ('preprossing ' + filename + '...')
		fpr = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
		count = 0
		fpr.readline()
		fpw = open('data/snli/processed_data/'+filename+'.txt', 'w')
		for line in fpr:
			sentences = line.strip().split('\t')
			if sentences[3] == '-':
				continue

			tokens = sentences[1].split(' ')
			tokens = [token for token in tokens if token != '(' and token != ')']
			fpw.write(' '.join(tokens)+'\t')
			fpw.write(SEPERATION_STR)
			tokens = sentences[2].split(' ')
			tokens = [token for token in tokens if token != '(' and token != ')']
			fpw.write(' '.join(tokens)+'\t')
			fpw.write(SEPERATION_STR)
			fpw.write(str(labelDict[sentences[0]])+'\n')
			count += 1
		fpw.close()
		fpr.close()
	print('SNLI preprossing finished!')

if __name__ == "__main__":
	prepSNLI()

# def Multinli():
# 	filenames = ['train']
# 	labelDict = {'neutral':1, 'entailment':2, 'contradiction':3, '-':0}
#
# 	for filename in filenames:
# 		print ('preprossing ' + filename + '...')
# 		fpr = open('data/multinli/multinli_1.0/multinli_1.0_'+filename+'.txt', 'r')
# 		count = 0
# 		fpr.readline()
# 		fpw = open('data/multinli/sequence/'+filename+'.txt', 'w')
# 		for line in fpr:
# 			sentences = line.strip().split('\t')
# 			if sentences[0] == '-':
# 				continue
#
# 			tokens = sentences[1].split(' ')
# 			tokens = [token for token in tokens if token != '(' and token != ')']
# 			fpw.write(' '.join(tokens)+'\t')
# 			tokens = sentences[2].split(' ')
# 			tokens = [token for token in tokens if token != '(' and token != ')' ]
# 			fpw.write(' '.join(tokens)+'\t')
# 			fpw.write(str(labelDict[sentences[0]])+'\n')
# 			count += 1
# 		fpw.close()
# 		fpr.close()
# 	print('multinli preprossing finished!')
#
# if __name__ == "__main__":
# 	Multinli()