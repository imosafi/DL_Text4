
'''
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.
Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.
This software is provided by the copyright holder and creator "as is" and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
'''
import sys
def prepSNLI():
	filenames = ['dev', 'test', 'train']
	labelDict = {'neutral':1, 'entailment':2, 'contradiction':3, '-':0}

	for filename in filenames:
		print ('preprossing ' + filename + '...')
		fpr = open('data/snli/snli_1.0/snli_1.0_'+filename+'.txt', 'r')
		count = 0
		fpr.readline()
		fpw = open('data/snli/sequence/'+filename+'.txt', 'w')
		for line in fpr:
			sentences = line.strip().split('\t')
			if sentences[0] == '-':
				continue

			tokens = sentences[1].split(' ')
			tokens = [token for token in tokens if token != '(' and token != ')']
			fpw.write(' '.join(tokens)+'\t')
			tokens = sentences[2].split(' ')
			tokens = [token for token in tokens if token != '(' and token != ')' ]
			fpw.write(' '.join(tokens)+'\t')
			fpw.write(str(labelDict[sentences[0]])+'\n')
			count += 1
		fpw.close()
		fpr.close()
	print('SNLI preprossing finished!')

if __name__ == "__main__":
	prepSNLI()
