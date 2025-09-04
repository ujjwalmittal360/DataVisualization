import numpy as np
from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances,cosine_similarity
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import chebyshev, minkowski

from scipy.spatial.distance import hamming


customer_A = np.array([4, 5, 2, 3, 4]).reshape(1, -1)
customer_B = np.array([5, 3, 2, 4, 5]).reshape(1, -1)

customer_A_binary = np.array([1, 0, 1, 1, 0, 1])
customer_B_binary = np.array([1, 1, 1, 0, 0, 1])

euclidean_dist = euclidean_distances(customer_A, customer_B)[0][0]

print("euclidean_distances",euclidean_dist)


manhattan_dist = manhattan_distances(customer_A, customer_B)[0][0]
print("manhattan_distances",manhattan_dist)


cosine_sim = cosine_similarity(customer_A, customer_B)[0][0]
print("cosine_similarity",cosine_sim)

hamming_dist = hamming(customer_A_binary, customer_B_binary) * len(customer_A_binary)
print("hamming",hamming_dist)

jaccard_sim = jaccard_score(customer_A_binary, customer_B_binary)
print("accard_score",jaccard_sim)




user1 = np.array([5, 3, 4, 4, 2])
user2 = np.array([4, 2, 5, 4, 3])


chebyshev_dist = chebyshev(user1, user2)

minkowski_dist = minkowski(user1, user2, p=3)


print("Chebyshev Distance:", chebyshev_dist)
print("Minkowski Distance (p=3):", minkowski_dist)
