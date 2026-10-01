"""A binary search tree with no duplicate values."""


class _Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self._root = None
        self._size = 0

    def insert(self, value):
        if self._root is None:
            self._root = _Node(value)
            self._size = 1
            return
        node = self._root
        while True:
            if value < node.value:
                if node.left is None:
                    node.left = _Node(value)
                    self._size += 1
                    return
                node = node.left
            elif value > node.value:
                if node.right is None:
                    node.right = _Node(value)
                    self._size += 1
                    return
                node = node.right
            else:
                return  # duplicate value

    def search(self, value):
        node = self._root
        while node is not None:
            if value == node.value:
                return True
            node = node.left if value < node.value else node.right
        return False

    def delete(self, value):
        parent = None
        node = self._root
        while node is not None and node.value != value:
            parent = node
            node = node.left if value < node.value else node.right
        if node is None:
            return False

        if node.left is None or node.right is None:
            child = node.left if node.left is not None else node.right
            if parent is None:
                self._root = child
            elif parent.left is node:
                parent.left = child
            else:
                parent.right = child
        else:
            # two children: replace with inorder successor
            succ_parent = node
            succ = node.right
            while succ.left is not None:
                succ_parent = succ
                succ = succ.left
            node.value = succ.value
            if succ_parent.left is succ:
                succ_parent.left = succ.right
            else:
                succ_parent.right = succ.right
        self._size -= 1
        return True

    def min(self):
        if self._root is None:
            return None
        node = self._root
        while node.left is not None:
            node = node.left
        return node.value

    def max(self):
        if self._root is None:
            return None
        node = self._root
        while node.right is not None:
            node = node.right
        return node.value

    def successor(self, value):
        node = self._root
        succ = None
        while node is not None:
            if value < node.value:
                succ = node.value
                node = node.left
            else:
                node = node.right
        return succ

    def inorder(self):
        result = []
        def rec(node):
            if node is not None:
                rec(node.left)
                result.append(node.value)
                rec(node.right)
        rec(self._root)
        return result

    def preorder(self):
        result = []
        def rec(node):
            if node is not None:
                result.append(node.value)
                rec(node.left)
                rec(node.right)
        rec(self._root)
        return result

    def postorder(self):
        result = []
        def rec(node):
            if node is not None:
                rec(node.left)
                rec(node.right)
                result.append(node.value)
        rec(self._root)
        return result

    def height(self):
        def rec(node):
            if node is None:
                return -1
            return 1 + max(rec(node.left), rec(node.right))
        return rec(self._root)

    def size(self):
        return self._size
