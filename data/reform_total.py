FN_TOTAL            = '/workspace/paperassistant/backend/block_classifier/data/total_acl_2021.txt'
#FN_TOTAL_REFORMED   = '/workspace/paperassistant/backend/block_classifier/data/total_reformed.txt'
SEP = '----------\n'
END_TAG = '<END>'

def get_reformed_filename(fn):
	return fn.split('.txt')[0] + '_reformed.txt'

def reform_abstract(fn, fn_reformed=None):
	'''
		Original Abstract :
			Sentence-A'\t'Tag-A
			Sentence-B'\t'Tag-B
			...
			Sentence-E'\t'Tag-E
			Sentence-F'\t'Tag-F
		
		Reformed Abstract :
			Sentence-A'\t'Tag-B
			Sentence-B'\t'Tag-C
			...
			Sentence-E'\t'Tag-F
			Sentence-F'\t'<END>
	'''

	if fn_reformed == None:
		fn_reformed = get_reformed_filename(fn)

	with open(fn, 'r', encoding='utf-8') as rf:
		lines = rf.readlines()
		fn_print = fn.split('/')[-1]
		print(f'size of {fn_print} :', len(lines))

	tag_prev = END_TAG
	for idx, line in enumerate(lines):
		if line == SEP:
			tag = END_TAG
			tag_prev = tag
			lines[idx-1] = sentence_prev + '\t' + tag + '\n'
			continue

		sentence, tag = line.rstrip().split('\t')
		if tag_prev != END_TAG:
			lines[idx-1] = sentence_prev + '\t' + tag + '\n'
		sentence_prev = sentence
		tag_prev = tag
		
	with open(fn_reformed, 'w', encoding='utf-8') as wf:
		wf.writelines(lines)
		fn_reformed_print = fn_reformed.split('/')[-1]
		print(f'size of {fn_reformed_print} :', len(lines))

	return fn_reformed

if __name__ == '__main__':
	reform_abstract(FN_TOTAL)
	pass