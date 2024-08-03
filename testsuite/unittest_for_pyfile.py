import subprocess
import sys
import numpy as np

def run_kmeans(k, itr, eps, file1, file2):
    cmd = [sys.executable, "kmeans_pp.py", str(k), str(itr), str(eps), file1, file2]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return result.stdout, result.stderr

def compare_outputs(expected, actual, tolerance=1e-4):
    expected_lines = expected.strip().split('\n')
    actual_lines = actual.strip().split('\n')
    
    if len(expected_lines) != len(actual_lines):
        return False
    
    for exp, act in zip(expected_lines, actual_lines):
        exp_values = list(map(float, exp.split(',')))
        act_values = list(map(float, act.split(',')))
        
        if len(exp_values) != len(act_values):
            return False
        
        if not np.allclose(exp_values, act_values, atol=tolerance):
            return False
    
    return True

# Test cases with expected outputs for class cases
test_cases = [
    # Class cases
    (3, 133, 0, "input_1_db_1.txt", "input_1_db_2.txt", 
     "47,26,39\n-4.2255,0.9759,-5.9958,-8.9800\n5.9463,2.9054,3.2229,1.4600\n-8.8386,-2.9007,8.6640,-8.7299"),
    (7, 300, 0, "input_2_db_1.txt", "input_2_db_2.txt", 
     "47,73,117,93,116,127,20\n1.5380,-8.2989,0.9508,-6.9507,2.9385,-7.6864,-0.7774,7.6524\n-9.5390,-2.8968,9.0357,-8.8930,7.4280,7.6344,-9.1317,2.7000\n5.5216,-8.5753,9.4997,3.3288,4.4103,4.9404,8.0717,-7.6402\n5.7112,6.8322,6.5469,-9.5108,-1.8912,-1.0378,0.4399,-0.3839\n1.3141,2.1640,-0.1899,-4.1663,-4.3603,0.7886,-1.9220,6.2263\n-1.3690,-7.0854,-7.1044,-8.7077,4.5270,-4.4527,-5.9762,6.5507\n-4.1693,0.4939,-5.7353,-8.8055,6.0066,3.0729,2.9528,1.4878"),
    (15, 750, 0, "input_3_db_1.txt", "input_3_db_2.txt", 
     "47,46,73,57,69,78,14,20,70,11,8,1,41,28,67\n-6.7490,6.3745,1.4481,-9.2954,0.8996,-6.6966\n2.0895,6.0712,-7.3971,-6.1608,9.1438,-9.9515\n5.7635,5.7618,7.3249,8.6543,-4.5315,-5.1717\n9.6405,2.8798,4.0746,4.4335,7.7634,-7.0890\n-0.8562,-5.0703,-4.0813,0.8880,-2.6612,4.5360\n-1.8604,-4.2527,-2.7075,2.3606,-3.5208,5.0872\n-1.2493,-6.8279,-6.9381,-9.1221,4.8769,-4.2370\n-2.7863,0.5422,-6.0424,-8.4235,6.1894,3.3598\n5.3672,-2.4253,-5.9268,-4.2660,5.0812,-8.4198\n0.5861,-3.4983,-3.5463,0.3504,-2.2625,6.6827\n5.4315,6.6036,6.6923,-9.4553,-1.8912,-1.0879\n3.2848,-7.7876,-1.5519,8.0079,5.5253,-8.8616\n2.0952,6.3185,-8.9860,3.2405,4.0726,2.7746\n-0.5125,-0.1814,2.9240,-7.6523,1.3078,9.1922\n0.8489,-0.9603,-7.4210,9.9542,-8.7760,2.0510"),
    # Normal cases
    (3, 100, 0.001, "small_2d_1.csv", "small_2d_2.csv", 
     "3,6,4\n0.3429,0.1124,0.3712\n0.0771,0.6590,0.4720\n0.8495,0.5212,0.7519"),
    (5, 200, 0.0001, "medium_3d_1.csv", "medium_3d_2.csv", 
     "47,49,80,61,75\n0.7034,0.5297,0.4336,0.2324,0.6802\n0.3708,0.7194,0.7161,0.7016,0.5562\n0.2107,0.2028,0.5002,0.3499,0.4362\n0.7631,0.7420,0.3408,0.5430,0.1470\n0.7746,0.2415,0.5560,0.7112,0.7707"),
    (10, 300, 0.00001, "large_5d_1.csv", "large_5d_2.csv", 
     "815,499,816,613,768,857,151,198,810,159\n0.2343,0.4278,0.4315,0.6484,0.5211,0.7419,0.7177,0.6249,0.2950\n0.7610,0.6313,0.3166,0.6539,0.3093,0.6737,0.3984,0.4614,0.2686\n0.7239,0.3192,0.4059,0.3495,0.3957,0.3012,0.6585,0.6469,0.7080\n0.3009,0.6803,0.4625,0.3183,0.4498,0.2934,0.4787,0.7935,0.2809\n0.6180,0.4136,0.7212,0.6446,0.5954,0.7503,0.5835,0.6462,0.6703\n0.5800,0.7265,0.5666,0.8046,0.7255,0.3135,0.5214,0.3828,0.5024\n0.6774,0.6269,0.4816,0.2908,0.7130,0.7085,0.2049,0.3497,0.5362\n0.3711,0.2262,0.6950,0.4140,0.3055,0.4187,0.4836,0.3172,0.3070\n0.3503,0.7468,0.4853,0.2741,0.4707,0.4967,0.7043,0.2411,0.5955\n0.2357,0.3253,0.4163,0.7105,0.5299,0.4033,0.3357,0.5503,0.7937"),
    # Edge cases
    (2, 50, 0.1, "single_point_1.csv", "single_point_2.csv", "Invalid number of clusters!"),
    # Invalid cases
    (0, 100, 0.001, "small_2d_1.csv", "small_2d_2.csv", "Invalid number of clusters!"),  # Invalid k
    (3, -1, 0.001, "small_2d_1.csv", "small_2d_2.csv", "Invalid maximum iteration!"),   # Invalid itr
    (3, 100, -0.001, "small_2d_1.csv", "small_2d_2.csv", "Invalid epsilon!"), # Invalid eps
    ("tal", 100, -0.001, "small_2d_1.csv", "small_2d_2.csv", "Invalid number of clusters!"), # Invalid k - string
]

for i, case in enumerate(test_cases):
    k, itr, eps, file1, file2, expected_output = case
    print(f"Test case {i+1}:")
    stdout, stderr = run_kmeans(k, itr, eps, file1, file2)
    print("Output:", stdout)
    if stderr:
        print("Error:", stderr)
    
    if expected_output:
        if expected_output in stdout or expected_output in stderr:
            print("Test PASSED: Output matches expected result.")
        elif compare_outputs(expected_output, stdout):
            print("Test PASSED: Output matches expected result.")
        else:
            print("Test FAILED: Output does not match expected result.")
    else:
        print("No expected output provided for this test case.")
    
    print("-" * 50)