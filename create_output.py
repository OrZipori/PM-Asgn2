# Or Zipori 302933833
# Shauli Ravfogel 308046861

import utils

def find_best_perp(dev_data, test_data):

        epsilon = 0.01
        min_val, max_val = 0., 2.
        current_lambda = min_val
	
        perps = []
        lidstone = utils.Lidstone(dev_data, test_data) 
        lidstone.train()
		
        while current_lambda < max_val:

              lidstone.calc_probs(lambdaP = current_lambda)
              perps.append((current_lambda, lidstone.validate()))
              current_lambda += epsilon
        
        best_lambda, best_perp = min(perps, key = lambda pair: pair[1])
        return best_lambda, best_perp

def fill_table(ho_model, dev_data, test_data, best_lambda):

	tbl = ""
	train_size, held_size = len(ho_model.trainset), len(ho_model.headoutset)
	lidstone = utils.Lidstone(dev_data, test_data) 
	lidstone.train()
	lidstone.calc_probs(lambdaP = best_lambda)
        
	for r in range(10):

		prob = ho_model.get_r_prob(r)
		lidstone_prob = lidstone.count_probs[r] if r != 0 else lidstone.probs["unseen0"]
		
		print (lidstone_prob)
			
		f_lambda = lidstone_prob * len(lidstone.trainset)
		f_h = prob * held_size
		N_t = ho_model.get_N_r(r)
		t_r = ho_model.get_t_r(r)
		
		tbl += str(r) + "\t" + "{:.5f}".format(f_lambda) + "\t" + "{:.5f}".format(f_h) + "\t" + str(N_t) + "\t" + str(t_r) + "\n"
	
	return "\n" + tbl.strip()


def write_output(outputs):

	with open("output.txt", "w") as f:
		
		f.write("#Students\tOr\tZipori\t302933833\tShauli\tRavfogel\t308046861\n")
	
		for i,o in enumerate(outputs):
	
			f.write("#Output" + str(i+1) + "\t" + str(o) + "\n")
	
def create(dev_fname, test_fname, output_fname, input_word):

	outputs = []
	
	# 1. init
	
	outputs.append(dev_fname) # 1
	outputs.append(test_fname) # 2
	outputs.append(input_word) # 3
	outputs.append(output_fname) # 4
	outputs.append(utils.vocabSize) # 5
	outputs.append(1./utils.vocabSize) # 6
	
	# 2. Development set preprocessing
	
	dev_data = utils.fetchFile(dev_fname)
	test_data = utils.fetchFile(test_fname)
	dev_words = utils.createWordsDataSet(dev_data)
	
	outputs.append(len(dev_words)) # 7
	
	# 3. Lidstone model training
	
	lidstone = utils.Lidstone(dev_data, test_data) 
	lidstone.train()
	lidstone.calc_probs(lambdaP=0.0)
	
	outputs.append(len(lidstone.validset)) # 8
	outputs.append(len(lidstone.trainset)) # 9
	outputs.append(len(lidstone.word_counter.keys())) # c, 10
	outputs.append(lidstone.word_counter[input_word]) # d, 11
	outputs.append(lidstone.probs[input_word]) # d, 12
	outputs.append(0) # 13
	
	lidstone.calc_probs(lambdaP = 0.1)
	
	outputs.append(lidstone.probs[input_word]) # 14
	outputs.append(lidstone.probs["unseen0"]) # 15
	
	lidstone.calc_probs(lambdaP = 0.01)
	outputs.append(lidstone.validate()) # 16
	lidstone.calc_probs(lambdaP = 0.1)
	outputs.append(lidstone.validate()) # 17
	lidstone.calc_probs(lambdaP = 1)
	outputs.append(lidstone.validate()) # 18
	
	best_lambda, best_perp = find_best_perp(dev_data, test_data)

	outputs.append(best_lambda) # 19
	outputs.append(best_perp) # 20
	
	# 4. Held out model training
	
	ho = utils.Headout(dev_data, test_data)
	ho.train()
	
	outputs.append(len(ho.trainset)) # 21
	outputs.append(len(ho.headoutset)) # 22
	outputs.append(ho.probs[input_word]) # 23
	outputs.append(ho.probs["unseen0"]) # 24
	
	lidstone.calc_probs(lambdaP = 5)
	print(utils.debugModel(lidstone))
	print(utils.debugModel(ho))
	
	# 5. Models evaluation on test set
	
	outputs.append(len(lidstone.testset)) # 25
	lidstone.calc_probs(lambdaP = best_lambda)
	outputs.append(lidstone.validate(dev = False)) # 26
	outputs.append(ho.validate(dev = False)) # 27
	outputs.append("L" if outputs[-1] > outputs[-2] else "H") # 28
	
	
	buckets = ho.wordsToBuckets()
	tbl = fill_table(ho, dev_data, test_data, best_lambda)
	outputs.append(tbl)
	print (len(outputs))
	write_output(outputs)
