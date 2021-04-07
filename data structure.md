# 队列与栈

## 队列

### 先入先出的数据结构

在 FIFO 数据结构中，将首先处理添加到队列中的第一个元素。队列是典型的 FIFO 数据结构。插入（insert）操作也称作入队（enqueue），新元素始终被添加在队列的末尾。 删除（delete）操作也被称为出队（dequeue)。 只能移除第一个元素。

#### 循环队列

循环队列是一种线性数据结构，其操作表现基于 FIFO（先进先出）原则并且队尾被连接在队首之后以形成一个循环。它也被称为“环形缓冲器”。

循环队列的一个好处是我们可以利用这个队列之前用过的空间。在一个普通队列里，一旦一个队列满了，我们就不能插入下一个元素，即使在队列前面仍有空间。但是使用循环队列，我们能使用这些空间去存储新的值。

#### 设计循环队列

> 你的实现应该支持如下操作：
>
> - `MyCircularQueue(k)`: 构造器，设置队列长度为 k 。
> - `Front`: 从队首获取元素。如果队列为空，返回 -1 。
> - `Rear`: 获取队尾元素。如果队列为空，返回 -1 。
> - `enQueue(value)`: 向循环队列插入一个元素。如果成功插入则返回真。
> - `deQueue()`: 从循环队列中删除一个元素。如果成功删除则返回真。
> - `isEmpty()`: 检查循环队列是否为空。
> - `isFull()`: 检查循环队列是否已满。
>
>  
>
> **示例：**
>
> ```
> MyCircularQueue circularQueue = new MyCircularQueue(3); // 设置长度为 3
> circularQueue.enQueue(1);  // 返回 true
> circularQueue.enQueue(2);  // 返回 true
> circularQueue.enQueue(3);  // 返回 true
> circularQueue.enQueue(4);  // 返回 false，队列已满
> circularQueue.Rear();  // 返回 3
> circularQueue.isFull();  // 返回 true
> circularQueue.deQueue();  // 返回 true
> circularQueue.enQueue(4);  // 返回 true
> circularQueue.Rear();  // 返回 4
> ```
>
>  
>
> **提示：**
>
> - 所有的值都在 0 至 1000 的范围内；
> - 操作数将在 1 至 1000 的范围内；
> - 请不要使用内置的队列库。

思路:

在循环队列中，使用一个`数组`和两个指针`head` 和 `tail`, `head` 表示队列的起始位置，`tail` 表示队列的结束位置

```python
class MyCircularQueue:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        """
        self.k = k
        self.size = k
        self.queue = [-1] * k
        self.head = -1
        self.tail = -1
        

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        """
        if self.size == 0:
            return False
        if self.head < 0:
            self.head += 1
        if self.tail == self.k-1:
            self.tail = 0
        else:
            self.tail += 1
        self.queue[self.tail] = value
        self.size -= 1
        return True

    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        """
        if self.queue[self.head] != -1:
            self.queue[self.head] = -1
            if self.head == self.k-1:
                self.head = 0
            else:
                self.head += 1
            self.size += 1
            return True
        

    def Front(self) -> int:
        """
        Get the front item from the queue.
        """
        return self.queue[self.head]
        

    def Rear(self) -> int:
        """
        Get the last item from the queue.
        """
        return  self.queue[self.tail]
        

    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return [-1] * self.k == self.queue
        

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return self.size == 0
        


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
```



### 队列和广度优先搜索(没懂)

广度优先搜索（BFS）的一个常见应用是找出从根结点到目标结点的最短路径。



## 栈

### 后入先出的数据结构

在 LIFO 数据结构中，将`首先处理添加到队列`中的`最新元素`。

与队列不同，栈是一个 LIFO 数据结构。通常，插入操作在栈中被称作入栈 `push` 。与队列类似，总是`在堆栈的末尾添加一个新元素`。但是，删除操作，退栈 `pop` ，将始终`删除`队列中相对于它的`最后一个元素`。

#### [155. 最小栈](https://leetcode-cn.com/problems/min-stack/)

> 设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。
>
> push(x) —— 将元素 x 推入栈中。
> pop() —— 删除栈顶的元素。
> top() —— 获取栈顶元素。
> getMin() —— 检索栈中的最小元素。
>
>
> 示例:
>
> 输入：
> ["MinStack","push","push","push","getMin","pop","top","getMin"]
> [[],[-2],[0],[-3],[],[],[],[]]
>
> 输出：
> [null,null,null,null,-3,null,0,-2]
>
> 解释：
> MinStack minStack = new MinStack();
> minStack.push(-2);
> minStack.push(0);
> minStack.push(-3);
> minStack.getMin();   --> 返回 -3.
> minStack.pop();
> minStack.top();      --> 返回 0.
> minStack.getMin();   --> 返回 -2.

思路：

数据栈 + 辅助栈 , 两栈同步更新数据, 辅助栈中只推入目前最小值

时间复杂度：对于题目中的所有操作，时间复杂度均为O(1)。因为栈的插入、删除与读取操作都是 O(1)，定义的每个操作最多调用栈操作两次。

空间复杂度：O(n)，其中 n 为总操作数。最坏情况下，会连续插入 n 个元素，此时两个栈占用的空间为 O(n)。

```python
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_stack = []

    def push(self, x: int) -> None:
        self.stack.append(x)
        if self.min_stack:
            self.min_stack.append(min(self.min_stack[-1],x))
        else:
            self.min_stack.append(x)

    def pop(self) -> None:
        self.stack.pop(-1)
        self.min_stack.pop(-1)

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
```



#### [20. 有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

> 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
>
> 有效字符串需满足：
>
> 左括号必须用相同类型的右括号闭合。
> 左括号必须以正确的顺序闭合。
> 注意空字符串可被认为是有效字符串。
>
> 示例 1:
>
> 输入: "()"
> 输出: true
> 示例 2:
>
> 输入: "()[]{}"
> 输出: true
> 示例 3:
>
> 输入: "(]"
> 输出: false
> 示例 4:
>
> 输入: "([)]"
> 输出: false
> 示例 5:
>
> 输入: "{[]}"
> 输出: true



思路:

循环字符串, 将字符串推入栈中, 如果是左符号, 推入栈中, 否则判断栈顶是不是对应的左符号, 如果不是则返回False, 否则继续, 直到循环完毕, 如果栈中还剩余符号, 则返回False

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        d = {')':'(',']':'[','}':'{'}
        for i in s:
            if i in d:
                top = stack.pop(-1) if stack else ''
                if top != d[i]:
                    return False
            else:
                stack.append(i)
        return stack == [] 
```













































