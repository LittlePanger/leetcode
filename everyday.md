# 2020/06

## [215. 数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)  2020/06/29

> 在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
>
> 示例 1:
>
> 输入: [3,2,1,5,6,4] 和 k = 2
> 输出: 5
> 示例 2:
>
> 输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
> 输出: 4
> 说明:
>
> 你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度

### 暴力解法

思路: 先排序, 再取值

时间复杂度: O(1)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort(reverse=True)
        return nums[k-1]
```

思路: 每次移除最大值, 移除k次

时间复杂度: O(n)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        while k > 1:
            k -=1
            nums.remove(max(nums))
        return max(nums)
```

### 分区减治(未实现)



### 总结

暴力解法简单, 但是面试估计不会这么简单



------

## [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)  2020/06/30

> 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
>
>  
>
> 示例 1：
>
> 输入：
> ["CQueue","appendTail","deleteHead","deleteHead"]
> [[],[3],[],[]]
> 输出：[null,null,3,-1]
> 示例 2：
>
> 输入：
> ["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
> [[],[],[5],[2],[],[]]
> 输出：[null,-1,null,null,5,2]
> 提示：
>
> 1 <= values <= 10000
> 最多会对 appendTail、deleteHead 进行 10000 次调用



思路:

两个栈, 一个入栈一个出栈, 

入队列: 数据进入栈

出队列: 若出入皆空则返回-1; 若出栈为空, 将入栈中的内容全部推入出栈中,返回出栈中最后一个

```python
class CQueue:

    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def appendTail(self, value: int) -> None:
        self.in_stack.append(value)

    def deleteHead(self) -> int:
        if not self.in_stack and not self.out_stack:
            return -1
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop()

# Your CQueue object will be instantiated and called as such:
# obj = CQueue()
# obj.appendTail(value)
# param_2 = obj.deleteHead()
```



### 总结

理解队列与栈的概念即可

- 队列: 先进先出

- 栈: 先进后出





# 2020/07

## [718. 最长重复子数组](https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray/) 2020/07/01 

难度中等

> 给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。
>
> 示例 1:
>
> 输入:
> A: [1,2,3,2,1]
> B: [3,2,1,4,7]
> 输出: 3
> 解释: 
> 长度最长的公共子数组是 [3, 2, 1]。
> 说明:
>
> 1 <= len(A), len(B) <= 1000
> 0 <= A[i], B[i] < 100



### 滑动窗口

思路: 

以A为基准, B每次右移一位, 将重复部分zip合并, 循环zip, 如果相同计数器+1, 否则记录计数结果并清空计数器

然后B每次向左移一位, 重复操作, 取最大的计数结果

```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        len_a = len(A)
        len_b = len(B)
        m_list = []
        
        def for_zip(a,b):
            m = 0
            for j in zip(a,b):
                if j[0] != j[1]:
                    if m != 0:
                        m_list.append(m)
                    m = 0
                else:
                    m += 1
            if m != 0:
                m_list.append(m)
        
        for i in range(len_a):
            for_zip(A[i:len_a],B[0:len_b - i])
            for_zip(A[0:len_a - i],B[i:len_b])
        return max(m_list) if m_list else 0
```



思路: 

A[0]与B[-1]开始, A向左, B向右, 直到A[-1],B[0]

```python
class Solution:
    def findLength(self, A: List[int], B: List[int]) -> int:
        len_a = len(A)
        len_b = len(B)
        m_list = []
        for i in range(1,len_a+len_b):
            if i <= len_b:
                m = 0
                for j in zip(A[0:i], B[len_b-i:]):
                    if j[0] == j[1]:
                        m += 1
                    else:
                        m_list.append(m)
                        m = 0
                m_list.append(m)
            else:
                m = 0
                for j in zip(A[i-len_a:], B[:len_a+len_b-i]):
                    if j[0] == j[1]:
                        m += 1
                    else:
                        m_list.append(m)
                        m = 0
                m_list.append(m)
        return max(m_list)
```



### 动态规划(未实现)

### 总结

第一次做按照滑动窗口第一种解法实现, 滑动窗口基本理解, 样例较多, 遗漏情况较多



## [378. 有序矩阵中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-sorted-matrix/)  2020/07/02



> 给定一个 n x n 矩阵，其中每行和每列元素均按升序排序，找到矩阵中第 k 小的元素。
> 请注意，它是排序后的第 k 小元素，而不是第 k 个不同的元素。
>
>  
>
> 示例：
>
> matrix = [
>    [ 1,  5,  9],
>    [10, 11, 13],
>    [12, 13, 15]
> ],
> k = 8,
>
> 返回 13。
>
> 提示：
> 你可以假设 k 的值永远是有效的，1 ≤ k ≤ n2 。



### 暴力解法

思路: 最暴力解法肯定是直接将矩阵展开成一维数组, 然后进行排序, 按照索引取值

时间复杂度O(n)  对n2个数排序的时间, 应该不是O(n)

空间复杂度O(n2)

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        l = []
        for i in matrix:
            l.extend(i)
        l.sort()
        return l[k-1]
```



### 二分查找

思路: 通过最小值(left)与最大值(right)找到中间值(mid), 然后计算出整个矩阵中小于中间值的数量num, 与k作比较, 若相同则得出结果, 若不同则根据大小继续二分查找



时间复杂度O(n2 * log2n)

空间复杂度O(1)

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        left = matrix[0][0]
        right = matrix[-1][-1]
        while left < right:
            mid = (left + right) // 2
            num = 0
            for i in matrix:
                for j in i:
                    if j <= mid:
                        num += 1
            if num < k:
                left = mid + 1
            else:
                right = mid
        return left
```



在对比矩阵与mid耗时过多, 利用有序矩阵的性质进行优化

从有序矩阵左下角开始走, 如果当前位置小于mid, 向右移一位, 如果大于mid, 向上移一位, 重复操作直到移出矩阵

时间复杂度：O(nlog(r-l))，二分查找进行次数为 O(log(r−l))，每次操作时间复杂度为 O(n)

空间复杂度：O(1)

```python
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        left = matrix[0][0]
        right = matrix[-1][-1]
        while left < right:
            mid = (left + right) // 2
            num = 0
            i, j = len(matrix)-1, 0
            while i >-1 and j <len(matrix):
                if matrix[i][j] <= mid:
                    num += i + 1
                    j += 1
                else:
                    i -= 1
            if num < k:
                left = mid + 1
            else:
                right = mid
        return left
```



### 归并排序(未实现)



## [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)  2020/07/03

> 将一个按照升序排列的有序数组，转换为一棵高度平衡二叉搜索树。
>
> 本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
>
> 示例:
>
> 给定有序数组: [-10,-3,0,5,9],
>
> 一个可能的答案是：[0,-3,9,-10,null,5]，它可以表示下面这个高度平衡二叉搜索树：
>
> ```
>       0
>      / \
>    -3   9
>    /   /
>  -10  5
> ```



### 二分查找

思路: 

​		根据二叉搜索树的性质, 二叉搜索树的中序遍历是升序序列, 题目是根据升序序列转化为二叉搜索树, 即根据中序遍历逆推出二叉树. 只根据一个遍历能推出的二叉树不是唯一的, 示例中的数组能推出的二叉树(且高度平衡)如下

```
      0                        0                         0                       0
     / \            	      / \                       / \                     / \
   -3   9       	    -10  5                -10   9                -3   5
   /   /          	         \      \                 \     /                 /        \
 -10  5        	          -3    9               -3  5               -10      9
```

​		观察二叉树, 可以根据中位数作为二叉搜索树的根节点, 这样分给左右子树的数字个数相同或只相差1, 可以使树保持平衡.

​		确定平衡二叉搜索树的根节点之后，其余的数字分别位于平衡二叉搜索树的左子树和右子树中，左子树和右子树分别也是平衡二叉搜索树，因此可以通过递归的方式创建平衡二叉搜索树.

​		在给定中序遍历序列数组的情况下，每一个子树中的数字在数组中一定是连续的，因此可以通过数组下标范围确定子树包含的数字，下标范围记为 [left, rignt]。对于整个中序遍历序列，下标范围从 left=0到 right = nums.length−1。当 left>right 时，平衡二叉搜索树为空。

以下三种方法中，方法一总是选择中间位置左边的数字作为根节点，方法二总是选择中间位置右边的数字作为根节点，方法三是方法一和方法二的结合，选择任意一个中间位置数字作为根节点。



方法一：中序遍历，总是选择中间位置左边的数字作为根节点

时间复杂度：O(n)，其中 n 是数组的长度。每个数字只访问一次。

空间复杂度：O(logn)，其中 n 是数组的长度。空间复杂度不考虑返回值，因此空间复杂度主要取决于递归栈的深度，递归栈的深度是 O(logn)。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def tree(left, right):
            if left > right:
                return None
            mid = (left + right) // 2

            node = TreeNode()
            node.val = nums[mid]
            node.left = tree(left, mid - 1)
            node.right = tree(mid + 1, right)
            return node
        return tree(0, len(nums) - 1)
```



## [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)  2020/07/06

> 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。
>
> 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
>
> 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
>
> 
>
> 网格中的障碍物和空位置分别用 1 和 0 来表示。
>
> 说明：m 和 n 的值均不超过 100。
>
> 示例 1:
>
> 输入:
> [
>   [0,0,0],
>   [0,1,0],
>   [0,0,0]
> ]
> 输出: 2
> 解释:
> 3x3 网格的正中间有一个障碍物。
> 从左上角到右下角一共有 2 条不同的路径：
> 1. 向右 -> 向右 -> 向下 -> 向下
> 2. 向下 -> 向下 -> 向右 -> 向右
>



### 动态规划

思路 : 

移动题目, 类似爬楼梯, 但是爬楼梯是一维的, 而本题是二维的

由题意分析可得, 移动到当前方块的可能情况`dp[i][j] = dp[i][j-1] + dp[i-1][j]`, 即当前位置左侧和上面移动到当前位置的可能情况之和, 状态转移方程如下:
$$
dp[i][j]  = \begin{cases}{ dp[i][j-1] + dp[i-1][j]}  \quad \quad (i,j)无障碍物 \\0  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad  \quad \quad (i,j)有障碍物 \end{cases}
$$
状态初始状态, 第一个位置只能有一种情况, 即 `dp[0][0] = 1`

第一行的各个位置只等于其左边的可能数, 第一列同理, 只等于其上边的可能数



时间复杂度O(nm)

空间复杂度O(1)    直接将可能数存在了原数组中

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
            for i in range(len(obstacleGrid)):
                for j in range(len(obstacleGrid[0])):
                    if obstacleGrid[i][j] == 0:
                        if i == 0 and j == 0:
                            obstacleGrid[i][j] = 1
                        elif j > 0 and i > 0:
                            obstacleGrid[i][j] = obstacleGrid[i][j - 1] + obstacleGrid[i - 1][j]
                        elif j > 0:
                            obstacleGrid[i][j] = obstacleGrid[i][j - 1]
                        elif i > 0:
                            obstacleGrid[i][j] = obstacleGrid[i - 1][j]
                    else:
                        obstacleGrid[i][j] = 0
            return obstacleGrid[-1][-1]
```



## [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)  2020/07/07

> 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
>
> 说明: 叶子节点是指没有子节点的节点。
>
> 示例: 
> 给定如下二叉树，以及目标和 sum = 22，
>
>                5
>              /   \
>             4     8
>            /      /  \
>           11   13  4
>          /  \      \
>         7    2      1
> 返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。
>



### 广度优先搜索结合队列

思路: 

记录从根节点到当前节点的路径和，以防止重复计算。使用两个队列，分别存储将要遍历的节点，以及根节点到这些节点的路径和即可。

时间复杂度 O(N)  N是树的节点数,  每个节点访问一次,时间不会超过节点数

空间复杂度 O(N)  N是树的节点数,  取决于储值队列的开销, 不会超过节点数

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        # 初始化
        nodeQueue = [root]
        valQueue = [root.val]
        while nodeQueue:
            # 取值
            node = nodeQueue.pop(0)
            val = valQueue.pop(0)
            # 根节点跳出本次循环, 根节点路径和满足条件返回true
            if node.left == None and node.right == None:
                if sum== val:
                    return True
                continue
            # 将节点与路径和放进队列中
            if node.left:
                nodeQueue.append(node.left)
                valQueue.append(node.left.val + val)
            if node.right:
                nodeQueue.append(node.right)
                valQueue.append(node.right.val + val)
        return False
```



### 递归

思路:

观察要求我们完成的函数，我们可以归纳出它的功能：询问是否存在从当前节点 root 到叶子节点的路径，满足其路径和为 sum。

假定从根节点到当前节点的值之和为 val，我们可以将这个大问题转化为一个小问题：是否存在从当前节点的子节点到叶子的路径，满足其路径和为 sum - val。

不难发现这满足递归的性质，若当前节点就是叶子节点，那么我们直接判断 sum 是否等于 val 即可（因为路径和已经确定，就是当前节点的值，我们只需要判断该路径和是否满足条件）。若当前节点不是叶子节点，我们只需要递归地询问它的子节点是否能满足条件即可。



时间复杂度：O(N)，其中 N 是树的节点数。对每个节点访问一次。

空间复杂度：O(H)，其中 H 是树的高度。空间复杂度主要取决于递归时栈空间的开销，最坏情况下，树呈现链状，空间复杂度为 O(N)。平均情况下树的高度与节点数的对数正相关，空间复杂度为 O(logN)。

```python
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right:
            return sum == root.val
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)
```










