# 代码大模型评测数据集探路报告

## 数据集: `BAAI/TACO`

**特征标签**: 纯算法竞赛题，I/O 格式。

> ❌ **获取失败，跳过**。报错信息: `RuntimeError: Dataset scripts are no longer supported, but found TACO.py`

---

## 数据集: `deepmind/code_contests`

**特征标签**: Google 高难度竞赛题。

- **成功扫描样本数**: 300 条
- **数据字典字段**: `['name', 'description', 'public_tests', 'private_tests', 'generated_tests', 'source', 'difficulty', 'solutions', 'incorrect_solutions', 'cf_contest_id', 'cf_index', 'cf_points', 'cf_rating', 'cf_tags', 'is_description_translated', 'untranslated_description', 'time_limit', 'memory_limit_bytes', 'input_file', 'output_file']`

### 1. 难度分布
- **0**: 99 题
- **7**: 39 题
- **10**: 36 题
- **8**: 31 题
- **9**: 28 题
- **11**: 26 题
- **12**: 12 题
- **2**: 9 题
- **6**: 6 题
- **13**: 5 题
- **1**: 2 题
- **14**: 2 题
- **16**: 1 题
- **15**: 1 题
- **21**: 1 题
- **3**: 1 题
- **17**: 1 题

### 2. 参考代码长度 (评估是否为长代码)
- **概貌**: 最小=`2`行 | 中位数=`39`行 | 最大=`809`行
- **行数分桶**:
  - `000-039` 行区间: 145 题
  - `040-079` 行区间: 75 题
  - `080-119` 行区间: 36 题
  - `120-159` 行区间: 9 题
  - `160-199` 行区间: 15 题
  - `200-239` 行区间: 2 题
  - `240-279` 行区间: 2 题
  - `280-319` 行区间: 1 题
  - `400-439` 行区间: 1 题
  - `520-559` 行区间: 1 题
  - `640-679` 行区间: 1 题
  - `800-839` 行区间: 1 题

### 3. 测试用例数量 (评估能否按比例给分)
- **概貌**: 最少=`0`个 | 中位数=`2`个 | 最多=`5`个

### 4. 原始真实数据抽样 (极度重要：请观察 Prompt 骨架和测试例格式)

#### Sample 1 真实数据
```json
{
  "name": "brcktsrm",
  "description": "Problem description.\nVipul is a hardworking super-hero who maintains the bracket ratio of all the strings in the world. Recently he indulged himself in saving the string population so much that he lost his ability for checking brackets (luckily, not permanently ).Being his super-hero friend help him in his time of hardship. \n\nInput\n\nThe first line of the input contains an integer T denoting the number of test cases. The description of T test cases follows.\nThe first line of each test case  ...[内容过长截断, 原长 925 字符]",
  "public_tests": "{'input': ['3\n((()))\n(())()\n()(()'], 'output': ['YES\nYES\nNO']}",
  "private_tests": "{'input': [], 'output': []}",
  "generated_tests": "{'input': ['3\n((()))\n(())()\n()())', '3\n((()()\n(())()\n()(()', '3\n((()))\n(())))\n()())', '3\n)))(((\n(())))\n()())', '3\n((()))\n(())()\n))(((', '3\n((()()\n(())()\n()(((', '3\n((()))\n(())()\n()()(', \"3\n((()()\n'())()\n()(((\", '3\n)))(((\n(())))\n()(*)', \"3\n)(()()\n'())()\n()(((\", '3\n))*(((\n(())))\n()(*)', \"3\n)()(()\n'())()\n()(((\", '3\n)*)(((\n(())))\n()(*)', \"3\n)()())\n'())()\n()(((\", '3\n)*)(((\n(()())\n()(*)', \"3\n)()())\n'()())\n()(((\", '3\n)*)(((\n(()())\n))(*(', \"3 ...[内容过长截断, 原长 4559 字符]",
  "source": "1",
  "difficulty": "6",
  "solutions": "{'language': [1, 1, 1], 'solution': [\"for _ in range(input()):\n    try:\n        eval(raw_input())\n        print 'YES'\n    except TypeError:\n        print 'YES'\n    except:\n        print 'NO'\", 'for _ in range(input()):\n    ins = raw_input().strip()\n    stck = []\n    res = \"YES\"\n    for x in ins:\n        if x == \"(\":\n            stck.append(x)\n        else:\n            if len(stck)>0:\n                stck.pop()\n            else:\n                res = \"NO\"\n               ...[内容过长截断, 原长 692 字符]",
  "incorrect_solutions": "{'language': [], 'solution': []}",
  "cf_contest_id": "0",
  "cf_index": "",
  "cf_points": "0.0",
  "cf_rating": "0",
  "cf_tags": "[]",
  "is_description_translated": "False",
  "untranslated_description": "",
  "time_limit": "None",
  "memory_limit_bytes": "0",
  "input_file": "",
  "output_file": "",
}
```

#### Sample 2 真实数据
```json
{
  "name": "comm3",
  "description": "The Chef likes to stay in touch with his staff. So, the Chef, the head server, and the sous-chef all carry two-way transceivers so they can stay in constant contact. Of course, these transceivers have a limited range so if two are too far apart, they cannot communicate directly.\n\n\nThe Chef invested in top-of-the-line transceivers which have a few advanced features. One is that even if two people cannot talk directly because they are out of range, if there is another transceiver that is close  ...[内容过长截断, 原长 2057 字符]",
  "public_tests": "{'input': ['3\n1\n0 1\n0 0\n1 0\n2\n0 1\n0 0\n1 0\n2\n0 0\n0 2\n2 1'], 'output': ['yes\nyes\nno\n']}",
  "private_tests": "{'input': [], 'output': []}",
  "generated_tests": "{'input': ['3\n1\n0 1\n0 -1\n1 0\n2\n0 1\n0 0\n1 0\n2\n0 0\n0 2\n2 1', '3\n2\n0 1\n0 -1\n1 0\n2\n0 1\n0 0\n1 0\n2\n0 0\n0 2\n2 1', '3\n2\n0 -1\n0 -1\n1 0\n2\n0 0\n0 0\n1 0\n2\n1 0\n1 2\n2 1', '3\n2\n0 -1\n-1 -1\n2 0\n2\n1 0\n0 -1\n1 -1\n2\n1 0\n1 2\n1 1', '3\n2\n0 -1\n0 -1\n2 0\n2\n2 0\n0 -1\n1 -2\n2\n1 0\n1 2\n1 1', '3\n2\n0 -1\n0 -1\n2 -1\n1\n2 0\n0 -1\n1 -2\n2\n1 0\n1 2\n1 1', '3\n2\n0 -2\n-1 -1\n1 0\n2\n-1 0\n-1 0\n1 -1\n1\n1 -1\n1 2\n1 1', '3\n2\n0 1\n0 -1\n2 -1\n1\n2 0\n0 -1\n1 -2\n2\n0 0\ ...[内容过长截断, 原长 8076 字符]",
  "source": "1",
  "difficulty": "1",
  "solutions": "{'language': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'solution': ['import math\nno_of_testcases = int(input())\nfor each in range(no_of_testcases):\n    dist = int(input())\n    point_1 = map(int,raw_input().split())\n    point_2 = map(int,raw_input().split())\n    point_3 = map(int,raw_input().split())    \n    point_12 =math.sqrt( math.pow((point_1[0] -point_2[0]),2) + math.pow((point_1[1] -point_2[1] ...[内容过长截断, 原长 25877 字符]",
  "incorrect_solutions": "{'language': [], 'solution': []}",
  "cf_contest_id": "0",
  "cf_index": "",
  "cf_points": "0.0",
  "cf_rating": "0",
  "cf_tags": "[]",
  "is_description_translated": "False",
  "untranslated_description": "",
  "time_limit": "None",
  "memory_limit_bytes": "0",
  "input_file": "",
  "output_file": "",
}
```

---

## 数据集: `codeparrot/apps`

**特征标签**: 经典算法题集，按难度分级。

> ❌ **获取失败，跳过**。报错信息: `RuntimeError: Dataset scripts are no longer supported, but found apps.py`

---

## 数据集: `FudanSELab/ClassEval`

**特征标签**: 面向对象（Class级别）工程题。自带骨架。

- **成功扫描样本数**: 100 条
- **数据字典字段**: `['task_id', 'skeleton', 'test', 'solution_code', 'import_statement', 'class_description', 'methods_info', 'class_name', 'test_classes', 'class_constructor', 'fields']`

### 2. 参考代码长度 (评估是否为长代码)
- **概貌**: 最小=`18`行 | 中位数=`40`行 | 最大=`111`行
- **行数分桶**:
  - `000-039` 行区间: 50 题
  - `040-079` 行区间: 48 题
  - `080-119` 行区间: 2 题

### 3. 测试用例数量 (评估能否按比例给分)
- *(未成功解析出测试用例数)*

### 4. 原始真实数据抽样 (极度重要：请观察 Prompt 骨架和测试例格式)

#### Sample 1 真实数据
```json
{
  "task_id": "ClassEval_0",
  "skeleton": "import logging\nimport datetime\n\nclass AccessGatewayFilter:\n    \"\"\"\n    This class is a filter used for accessing gateway filtering, primarily for authentication and access log recording.\n    \"\"\"\n\n    def __init__(self):\n        pass\n\n    def filter(self, request):\n        \"\"\"\n        Filter the incoming request based on certain rules and conditions.\n        :param request: dict, the incoming request details\n        :return: bool, True if the request is allowed, False othe ...[内容过长截断, 原长 2012 字符]",
  "test": "import unittest\n\nclass AccessGatewayFilterTestFilter(unittest.TestCase):\n    def test_filter_1(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/api/data', 'method': 'GET'}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n    def test_filter_2(self):\n        agf = AccessGatewayFilter()\n        request = {'path': '/api/data', 'method': 'POST'}\n        res = agf.filter(request)\n        self.assertTrue(res)\n\n    def test_filter_3(self):\n        a ...[内容过长截断, 原长 5696 字符]",
  "solution_code": "import logging\nimport datetime\n\n\nclass AccessGatewayFilter:\n\n    def __init__(self):\n        pass\n\n    def filter(self, request):\n        request_uri = request['path']\n        method = request['method']\n\n        if self.is_start_with(request_uri):\n            return True\n\n        try:\n            token = self.get_jwt_user(request)\n            user = token['user']\n            if user['level'] > 2:\n                self.set_current_user_info_and_log(user)\n                return ...[内容过长截断, 原长 1384 字符]",
  "import_statement": "['import logging', 'import datetime']",
  "class_description": "    \"\"\"\n    This class is a filter used for accessing gateway filtering, primarily for authentication and access log recording.\n    \"\"\"\n",
  "methods_info": "[{'method_name': 'filter', 'method_description': 'def filter(self, request):\n        \"\"\"\n        Filter the incoming request based on certain rules and conditions.\n        :param request: dict, the incoming request details\n        :return: bool, True if the request is allowed, False otherwise\n        >>> filter = AccessGatewayFilter()\n        >>> filter.filter({\'path\': \'/login\', \'method\': \'POST\'})\n        True\n\n        \"\"\"', 'test_class': 'AccessGatewayFilterTestFilter', ' ...[内容过长截断, 原长 9815 字符]",
  "class_name": "AccessGatewayFilter",
  "test_classes": "['AccessGatewayFilterTestFilter', 'AccessGatewayFilterTestIsStartWith', 'AccessGatewayFilterTestGetJwtUser', 'AccessGatewayFilterTest']",
  "class_constructor": "class AccessGatewayFilter: \n    def __init__(self):\n        pass\n\n",
  "fields": "[]",
}
```

#### Sample 2 真实数据
```json
{
  "task_id": "ClassEval_1",
  "skeleton": "import math\nclass AreaCalculator:\n    \"\"\"\n    This is a class for calculating the area of different shapes, including circle, sphere, cylinder, sector and annulus.\n    \"\"\"\n\n\n    def __init__(self, radius):\n        \"\"\"\n        Initialize the radius for shapes.\n        :param radius: float\n        \"\"\"\n        self.radius = radius\n\n    def calculate_circle_area(self):\n        \"\"\"\n        calculate the area of circle based on self.radius\n        :return: area of circl ...[内容过长截断, 原长 2117 字符]",
  "test": "import unittest\n\nclass AreaCalculatorTestCalculateCircleArea(unittest.TestCase):\n    def test_calculate_circle_area(self):\n        areaCalculator = AreaCalculator(2)\n        self.assertAlmostEqual(12.56, areaCalculator.calculate_circle_area(), delta=0.01)\n    def test_calculate_circle_area_2(self):\n        areaCalculator = AreaCalculator(2.5)\n        self.assertAlmostEqual(19.63, areaCalculator.calculate_circle_area(), delta=0.01)\n\n    def test_calculate_circle_area_3(self):\n        a ...[内容过长截断, 原长 5593 字符]",
  "solution_code": "import math\n\n\nclass AreaCalculator:\n\n    def __init__(self, radius):\n        self.radius = radius\n\n    def calculate_circle_area(self):\n        return math.pi * self.radius ** 2\n\n    def calculate_sphere_area(self):\n        return 4 * math.pi * self.radius ** 2\n\n    def calculate_cylinder_area(self, height):\n        return 2 * math.pi * self.radius * (self.radius + height)\n\n    def calculate_sector_area(self, angle):\n        return self.radius ** 2 * angle / 2\n\n    def calcul ...[内容过长截断, 原长 617 字符]",
  "import_statement": "['import math']",
  "class_description": "    \"\"\"\n    This is a class for calculating the area of different shapes, including circle, sphere, cylinder, sector and annulus.\n    \"\"\"\n",
  "methods_info": "[{'method_name': 'calculate_circle_area', 'method_description': 'def calculate_circle_area(self):\n        \"\"\"\n        calculate the area of circle based on self.radius\n        :return: area of circle, float\n        >>> areaCalculator = AreaCalculator(2)\n        >>> areaCalculator.calculate_circle_area()\n        12.566370614359172\n        \"\"\"', 'test_class': 'AreaCalculatorTestCalculateCircleArea', 'test_code': 'class AreaCalculatorTestCalculateCircleArea(unittest.TestCase):\n    def ...[内容过长截断, 原长 8618 字符]",
  "class_name": "AreaCalculator",
  "test_classes": "['AreaCalculatorTestCalculateCircleArea', 'AreaCalculatorTestCalculateSphereArea', 'AreaCalculatorTestCalculateCylinderArea', 'AreaCalculatorTestCalculateSectorArea', 'AreaCalculatorTestCalculateAnnulusArea', 'AreaCalculatorTestCalculateMain']",
  "class_constructor": "class AreaCalculator: \n    def __init__(self, radius):\n        \"\"\"\n        Initialize the radius for shapes.\n        :param radius: float\n        \"\"\"\n        self.radius = radius\n\n",
  "fields": "['self.radius']",
}
```

---

## 数据集: `google-research-datasets/mbpp`

**特征标签**: 基础函数级生成。

- **成功扫描样本数**: 300 条
- **数据字典字段**: `['task_id', 'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list']`

### 2. 参考代码长度 (评估是否为长代码)
- **概貌**: 最小=`2`行 | 中位数=`5`行 | 最大=`29`行
- **行数分桶**:
  - `000-039` 行区间: 300 题

### 3. 测试用例数量 (评估能否按比例给分)
- **概貌**: 最少=`3`个 | 中位数=`3`个 | 最多=`3`个

### 4. 原始真实数据抽样 (极度重要：请观察 Prompt 骨架和测试例格式)

#### Sample 1 真实数据
```json
{
  "task_id": "601",
  "text": "Write a function to find the longest chain which can be formed from the given set of pairs.",
  "code": "class Pair(object): \n	def __init__(self, a, b): \n		self.a = a \n		self.b = b \ndef max_chain_length(arr, n): \n	max = 0\n	mcl = [1 for i in range(n)] \n	for i in range(1, n): \n		for j in range(0, i): \n			if (arr[i].a > arr[j].b and\n				mcl[i] < mcl[j] + 1): \n				mcl[i] = mcl[j] + 1\n	for i in range(n): \n		if (max < mcl[i]): \n			max = mcl[i] \n	return max",
  "test_list": "['assert max_chain_length([Pair(5, 24), Pair(15, 25),Pair(27, 40), Pair(50, 60)], 4) == 3', 'assert max_chain_length([Pair(1, 2), Pair(3, 4),Pair(5, 6), Pair(7, 8)], 4) == 4', 'assert max_chain_length([Pair(19, 10), Pair(11, 12),Pair(13, 14), Pair(15, 16), Pair(31, 54)], 5) == 5']",
  "test_setup_code": "",
  "challenge_test_list": "[]",
}
```

#### Sample 2 真实数据
```json
{
  "task_id": "602",
  "text": "Write a python function to find the first repeated character in a given string.",
  "code": "def first_repeated_char(str1):\n  for index,c in enumerate(str1):\n    if str1[:index+1].count(c) > 1:\n      return c \n  return \"None\"",
  "test_list": "['assert first_repeated_char(\"abcabc\") == \"a\"', 'assert first_repeated_char(\"abc\") == \"None\"', 'assert first_repeated_char(\"123123\") == \"1\"']",
  "test_setup_code": "",
  "challenge_test_list": "[]",
}
```

---

