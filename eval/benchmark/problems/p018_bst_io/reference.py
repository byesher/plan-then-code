import sys


class Node:
    def __init__(self, v):
        self.v = v
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, v):
        if self.root is None:
            self.root = Node(v)
            self.size = 1
            return
        cur = self.root
        while True:
            if v == cur.v:
                return
            if v < cur.v:
                if cur.left is None:
                    cur.left = Node(v)
                    self.size += 1
                    return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = Node(v)
                    self.size += 1
                    return
                cur = cur.right

    def search(self, v):
        cur = self.root
        while cur:
            if v == cur.v:
                return True
            cur = cur.left if v < cur.v else cur.right
        return False

    def delete(self, v):
        parent = None
        cur = self.root
        while cur and cur.v != v:
            parent = cur
            cur = cur.left if v < cur.v else cur.right
        if cur is None:
            return False
        if cur.left is None or cur.right is None:
            child = cur.left if cur.left else cur.right
            if parent is None:
                self.root = child
            elif parent.left is cur:
                parent.left = child
            else:
                parent.right = child
        else:
            succ_parent = cur
            succ = cur.right
            while succ.left:
                succ_parent = succ
                succ = succ.left
            cur.v = succ.v
            if succ_parent.left is succ:
                succ_parent.left = succ.right
            else:
                succ_parent.right = succ.right
        self.size -= 1
        return True

    def inorder(self):
        out = []

        def rec(n):
            if n:
                rec(n.left)
                out.append(n.v)
                rec(n.right)

        rec(self.root)
        return out


bst = BST()
for line in sys.stdin.read().splitlines():
    parts = line.split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "insert":
        bst.insert(int(parts[1]))
    elif cmd == "search":
        print("true" if bst.search(int(parts[1])) else "false")
    elif cmd == "delete":
        print("true" if bst.delete(int(parts[1])) else "false")
    elif cmd == "inorder":
        print(" ".join(str(x) for x in bst.inorder()))
    elif cmd == "size":
        print(bst.size)
