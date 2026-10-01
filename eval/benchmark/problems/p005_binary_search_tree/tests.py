from solution import BST


def test_insert_search():
    t = BST()
    assert t.search(5) is False
    t.insert(5)
    assert t.search(5) is True
    assert t.size() == 1


def test_insert_duplicate_ignored():
    t = BST()
    t.insert(5)
    t.insert(5)
    assert t.size() == 1


def test_inorder_sorted():
    t = BST()
    for v in [5, 3, 8, 1, 4, 7, 9]:
        t.insert(v)
    assert t.inorder() == [1, 3, 4, 5, 7, 8, 9]


def test_preorder():
    t = BST()
    for v in [5, 3, 8, 1, 4, 7, 9]:
        t.insert(v)
    assert t.preorder() == [5, 3, 1, 4, 8, 7, 9]


def test_postorder():
    t = BST()
    for v in [5, 3, 8, 1, 4, 7, 9]:
        t.insert(v)
    assert t.postorder() == [1, 4, 3, 7, 9, 8, 5]


def test_min_max():
    t = BST()
    assert t.min() is None
    assert t.max() is None
    for v in [5, 3, 8, 1, 9]:
        t.insert(v)
    assert t.min() == 1
    assert t.max() == 9


def test_successor():
    t = BST()
    for v in [5, 3, 8, 1, 4, 7, 9]:
        t.insert(v)
    assert t.successor(4) == 5
    assert t.successor(5) == 7
    assert t.successor(9) is None


def test_height():
    t = BST()
    assert t.height() == -1
    t.insert(5)
    assert t.height() == 0
    t.insert(3)
    t.insert(1)
    assert t.height() == 2


def test_delete_leaf():
    t = BST()
    for v in [5, 3, 8, 1]:
        t.insert(v)
    assert t.delete(1) is True
    assert t.search(1) is False
    assert t.size() == 3


def test_delete_one_child():
    t = BST()
    for v in [5, 3, 8, 1, 2]:
        t.insert(v)
    assert t.delete(1) is True
    assert t.search(1) is False
    assert t.search(2) is True
    assert t.inorder() == [2, 3, 5, 8]


def test_delete_two_children():
    t = BST()
    for v in [5, 3, 8, 1, 4, 7, 9]:
        t.insert(v)
    assert t.delete(5) is True
    assert t.search(5) is False
    assert t.inorder() == [1, 3, 4, 7, 8, 9]


def test_delete_root_single():
    t = BST()
    t.insert(5)
    assert t.delete(5) is True
    assert t.size() == 0
    assert t.search(5) is False


def test_delete_missing():
    t = BST()
    t.insert(5)
    assert t.delete(9) is False
    assert t.size() == 1
