import unittest
import mykmeanssp
import random
import math

class TestMyKMeansSP(unittest.TestCase):

    def test_basic_functionality(self):
        centroids = [[-4.8619, 0.9616, -5.192, -8.7806], [-7.6138, -1.7449, 7.3105, -9.6202], [-7.2635, -4.7385, 8.665, -7.5041]]
        datapoints = [[5.1987, 4.6511, 3.9588, 1.5633], [-9.1814, -1.9099, 7.5351, -9.1271], [-8.6599, -1.0249, 7.3827, -8.7116], [5.9535, 1.431, 4.0985, -0.452], [7.4093, 4.8439, 2.4968, 0.349], [-3.5389, 1.077, -7.335, -10.678], [-3.8622, 0.0303, -5.74, -7.8436], [-1.3859, 1.2227, -7.0055, -8.6457], [-10.4959, -4.0744, 9.252, -8.2888], [8.6108, 1.9186, 3.754, 2.6428], [-10.4333, -4.5075, 10.202, -9.4244], [-5.089, 1.643, -5.6783, -9.0886], [-5.2621, 3.9569, -6.0429, -9.3362], [3.6416, 2.8689, 3.774, 1.1974], [-9.0718, -2.5512, 8.2826, -9.2082], [6.6739, 3.0066, 4.9574, -1.8879], [-4.9023, 0.8963, -5.647, -8.5584], [4.5773, 2.5047, 3.3597, 2.1273], [-4.7168, 1.7971, -6.8804, -9.4231], [-5.9255, 0.4737, -5.6297, -10.4173], [-4.0028, 0.9123, -5.2014, -7.7724], [-5.0228, -0.1743, -5.0504, -9.6254], [5.1144, 2.069, 3.8066, 1.5165], [-9.1683, -3.656, 8.7669, -6.7468], [6.2742, 1.7176, 1.6906, 1.0659], [-2.7918, 0.1383, -7.2618, -8.892], [6.2997, 2.5083, 2.5256, 3.728], [-7.6138, -1.7449, 7.3105, -9.6202], [6.8126, 3.1593, 5.2757, 1.3372], [-8.2642, -2.8502, 9.1285, -7.5567], [5.5081, 3.1711, 1.0381, 3.1242], [-4.3371, 0.7319, -6.9581, -11.6021], [6.244, 2.451, 4.0639, 2.4154], [-8.6813, -3.6676, 9.0598, -8.8897], [-8.8269, -2.0954, 9.0742, -9.4112], [-9.8806, -3.4229, 9.0881, -8.0002], [-8.9953, -3.0733, 8.5037, -9.9938], [-8.5302, -2.6616, 9.3817, -9.5148], [-4.9131, 0.743, -7.3821, -8.482], [-3.958, 0.174, -3.9983, -7.4897], [-7.2635, -4.7385, 8.665, -7.5041], [6.9732, 2.5388, 1.7735, 1.9801], [-3.549, 0.7109, -5.5317, -7.9389], [4.8978, 3.2033, 2.0792, 1.1949], [-3.7136, 1.2962, -5.3944, -8.086], [-8.2774, -1.6664, 8.9386, -8.4941], [-8.0734, -2.7672, 8.053, -9.1868], [-4.8619, 0.9616, -5.192, -8.7806], [5.3131, 3.9346, 2.8077, 0.4719], [5.5855, 3.4143, 3.3298, 2.4453]]
        K = 3
        d = 4
        max_iter = 333
        epsilon = 0

        result = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)
        print(result)

        self.assertEqual(len(result), K)
        self.assertEqual(len(result[0]), d)

    def test_single_cluster(self):
        centroids = [[1.0, 1.0]]
        datapoints = [[1.1, 0.9], [0.9, 1.1], [1.0, 1.0], [1.2, 0.8]]
        K = 1
        d = 2
        max_iter = 300
        epsilon = 0.001

        result = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)

        self.assertEqual(len(result), K)
        self.assertEqual(len(result[0]), d)
        self.assertAlmostEqual(result[0][0], 1.05, places=2)
        self.assertAlmostEqual(result[0][1], 0.95, places=2)

    def test_three_dimensions(self):
        centroids = [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]
        datapoints = [[1.1, 0.9, 1.0], [4.9, 5.1, 5.0], [1.0, 1.1, 0.9], [5.1, 4.9, 5.1]]
        K = 2
        d = 3
        max_iter = 300
        epsilon = 0.001

        result = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)

        self.assertEqual(len(result), K)
        self.assertEqual(len(result[0]), d)

    def test_convergence(self):
        centroids = [[1.0, 2.0], [3.0, 4.0]]
        datapoints = [[1.1, 2.1], [2.9, 3.9], [1.2, 2.2], [3.1, 4.1]]
        K = 2
        d = 2
        max_iter = 1000
        epsilon = 0.00001

        result1 = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)
        result2 = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)

        for c1, c2 in zip(result1, result2):
            for v1, v2 in zip(c1, c2):
                self.assertAlmostEqual(v1, v2, places=4)

    def test_large_dataset(self):
        K = 5
        d = 10
        n_points = 1000
        centroids = [[random.uniform(0, 100) for _ in range(d)] for _ in range(K)]
        datapoints = [[random.uniform(0, 100) for _ in range(d)] for _ in range(n_points)]
        max_iter = 300
        epsilon = 0.001

        result = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)

        self.assertEqual(len(result), K)
        self.assertEqual(len(result[0]), d)

    def test_convergence(self):
        centroids = [[0.0, 0.0], [4.0, 4.0]]
        datapoints = [[1.0, 1.0], [1.5, 1.5], [3.5, 3.5], [4.0, 4.0]]
        K = 2
        d = 2
        max_iter = 1000
        epsilon = 0.01

        result = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)

        # Check if centroids have moved to the correct positions
        self.assertAlmostEqual(result[0][0], 1.25, delta=0.1)
        self.assertAlmostEqual(result[0][1], 1.25, delta=0.1)
        self.assertAlmostEqual(result[1][0], 3.75, delta=0.1)
        self.assertAlmostEqual(result[1][1], 3.75, delta=0.1)


    def test_high_dimensions(self):
        K = 2
        d = 100
        n_points = 50
        centroids = [[random.uniform(0, 10) for _ in range(d)] for _ in range(K)]
        datapoints = [[random.uniform(0, 10) for _ in range(d)] for _ in range(n_points)]
        max_iter = 300
        epsilon = 0.001

        result = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)

        self.assertEqual(len(result), K)
        self.assertEqual(len(result[0]), d)

    def test_stability(self):
        K = 3
        d = 2
        n_points = 100
        centroids = [[random.uniform(0, 10) for _ in range(d)] for _ in range(K)]
        datapoints = [[random.uniform(0, 10) for _ in range(d)] for _ in range(n_points)]
        max_iter = 300
        epsilon = 0.001

        results = []
        for _ in range(5):
            result = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)
            results.append(result)

        # Check if all results are similar (allowing for some small differences due to random initialization)
        for i in range(1, len(results)):
            for c1, c2 in zip(results[0], results[i]):
                for v1, v2 in zip(c1, c2):
                    self.assertAlmostEqual(v1, v2, delta=0.5)
    
    def test_all_points_identical(self):
        centroids = [[1.0, 1.0], [2.0, 2.0]]
        datapoints = [[1.0, 1.0]] * 10
        K = 2
        d = 2
        max_iter = 300
        epsilon = 0.001

        result = mykmeanssp.fit(centroids, datapoints, K, d, max_iter, epsilon)

        self.assertEqual(len(result), K)
        self.assertTrue(all([p == result[0] for p in datapoints]))


if __name__ == '__main__':
    unittest.main()