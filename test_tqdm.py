# test_tqdm.py
from tqdm.auto import tqdm
import time

print("测试tqdm输出...")

# 测试基本的tqdm进度条
print("\n1. 测试基本进度条:")
for i in tqdm(range(10)):
    time.sleep(0.1)

# 测试带描述的进度条
print("\n2. 测试带描述的进度条:")
for i in tqdm(range(5), desc="处理中"):
    time.sleep(0.2)

# 测试嵌套进度条
print("\n3. 测试嵌套进度条:")
for i in tqdm(range(3), desc="外层"):
    for j in tqdm(range(4), desc=f"内层{i}", leave=False):
        time.sleep(0.1)

print("\n测试完成!")