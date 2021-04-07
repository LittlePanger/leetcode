## 数组

### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

> 给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
>
> 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。
>
>  
>
> 示例:
>
> 给定 nums = [2, 7, 11, 15], target = 9
>
> 因为 nums[0] + nums[1] = 2 + 7 = 9
> 所以返回 [0, 1]



```python
# 1 暴力法, 循环两遍, 可能超时
# 2 字典, 查找迅速, 循环一遍即可
# 时间复杂度 O(n)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, n in enumerate(nums):
            if target - n in d:
                return [d.get(target - n), i]
            d[n] = i
# 字典的好处是查询速度快
```



**tips :**

实测三种判断方式各运行100万次用时（win10 python3.7.3）：

1. `key in dict` 用时 1.088 秒
2. `dict.get(key)` 用时 1.294 秒
3. `dict[key]` 用时 1.01 秒



### [26.删除排序数组中的重复项](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)

> 给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。
>
> 不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
>
>  
>
> 示例 1:
>
> 给定数组 nums = [1,1,2], 
>
> 函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。 
>
> 你不需要考虑数组中超出新长度后面的元素。
>
> 示例 2:
>
> 给定 nums = [0,0,1,1,1,2,2,3,3,4],
>
> 函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
>
> 你不需要考虑数组中超出新长度后面的元素。



```python
# 数组是有序数组
# 类似与双指针, 一个指针指向上一个修改的位置, 一个指针指向循环中的数组
# 时间复杂度 O(n)
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        j = 0
        for i in range(1,len(nums)):
            if nums[j] != nums[i]:
                # 指向的是上一个修改的位置,所以先将位置右移后再赋值
                j += 1
                nums[j] = nums[i]
        return j +1
```



### [27. 移除元素](https://leetcode-cn.com/problems/remove-element/)

> 给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
>
> 不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
>
> 元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
>
>  
>
> 示例 1:
>
> 给定 nums = [3,2,2,3], val = 3,
>
> 函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。
>
> 你不需要考虑数组中超出新长度后面的元素。
> 示例 2:
>
> 给定 nums = [0,1,2,2,3,0,4,2], val = 2,
>
> 函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。
>
> 注意这五个元素可为任意顺序。
>
> 你不需要考虑数组中超出新长度后面的元素。
>



```python
# 同上一题, 类似双指针
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        j = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[j] = nums[i]
                j += 1
        return j
```



### [35. 搜索插入位置](https://leetcode-cn.com/problems/search-insert-position/)

> 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
>
> 你可以假设数组中无重复元素。
>
> 示例 1:
>
> 输入: [1,3,5,6], 5
> 输出: 2
> 示例 2:
>
> 输入: [1,3,5,6], 2
> 输出: 1
> 示例 3:
>
> 输入: [1,3,5,6], 7
> 输出: 4
> 示例 4:
>
> 输入: [1,3,5,6], 0
> 输出: 0



```python
# 思路: 如果数组某个值大于等于目标值, 则返回索引, 如果目标值大于数组中的任何数, 则返回长度
# 时间复杂度 O(n)
# 空间复杂度 O(1)
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        for i in range(len(nums)):
            if nums[i] >= target:
                return i
        else:
            return i + 1

# 思路: 将小于目标值的数字放进新数组中, 返回新数组长度即可, 缺点需要开辟新列表
# 时间复杂度 O(n)
# 空间复杂度 O(n)
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return len([n for n in nums if n< target])
```



### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

> 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
>
> 示例:
>
> 输入: [-2,1,-3,4,-1,2,1,-5,4],
> 输出: 6
> 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
> 进阶:
>
> 如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。
>



```python
# 暴力法肯定不行, 时间复杂度O(n2)

# 貌似算是贪心, 保证前面的子序是最优解
# 时间复杂度O(n)
# 空间复杂度O(1)
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            if nums[i-1] > 0:
                nums[i] = nums[i-1] + nums[i]
        return max(nums)

# 动态规划
```











