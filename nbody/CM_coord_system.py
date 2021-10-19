import numpy as np

def CM_system(p1, p2, q1, q2):

	q_rel = np.array([q1[i]-q2[i] for i in range(3)])
	p_rel = np.array([p1[i]-p2[i] for i in range(3)])
	
	return(q_rel, p_rel)
