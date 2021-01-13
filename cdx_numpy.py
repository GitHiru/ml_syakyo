import numpy as np

np.version.full_version    # check version
a = np.array([1, 2, 3])
type(a)    # numpy.ndarray
a.shape    # (3,) :(row, col)
a.dtyeo    # dtype('int64')
a[0]    # 1

b = np.array([[1, 2, 3], [4, 5, 6]])
b[0]    # array([1, 2, 3])
b[0][0]    # 1 :0row 0col =[0, 0]

c = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
c[:2]    # 0~2行目列指定なくアクセス
