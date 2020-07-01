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