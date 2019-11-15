import numpy as np
import itertools

# y_true & y_pred is list of labels for each point. Recall, Precision, F1 is for predicted edges on underlying points
def pairwise_f1(y_true, y_pred):  # TODO Optimize this, we do not need to calculate trueNeg and that is a large fraction of all edges
	assert (len(y_true) == len(y_pred))
	truePos = 0
	falseNeg = 0
	
	trueNeg = 0
	falsePos = 0
	numPoints = len(y_true)

	for pid1, pid2 in itertools.combinations(range(numPoints), 2):
		if y_pred[pid1] == y_pred[pid2]:
			if y_true[pid1] == y_true[pid2]:
				truePos += 1  # TP
			else:
				falsePos += 1  # FP
		else:
			if y_true[pid1] == y_true[pid2]:
				falseNeg += 1  # FN
			else:
				trueNeg += 1  # TN
	
	precision 	= truePos / (truePos + falsePos) if (truePos + falsePos) > 0 else 1.
	recall 		= truePos / (truePos + falseNeg) if (truePos + falseNeg) > 0 else 1.
	f1 			= 2 * precision * recall / (precision + recall) if precision + recall > 0. else 0.
	
	return f1
