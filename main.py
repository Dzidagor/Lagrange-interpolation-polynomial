import numpy as np
import matplotlib.pyplot as plt

x_lst = np.array([-1, 0, 1, 2], dtype=float)
y_lst = np.array([-5, -10, -1, 34], dtype=float)


def the_lagrange_polynomial(x, y, inp_x):
    l = 0
    for i in range(len(y)):
        numerator = 1
        denominator = 1
        for j in range(len(x)):
            if i != j:
                numerator *= (inp_x - x[j])
                denominator *= (x[i] - x[j])
        l += y[i]*(numerator/denominator)
    return l


inp_x_lst = np.linspace(np.min(x_lst), np.max(x_lst), 100)
y_res_lst = [the_lagrange_polynomial(x_lst, y_lst, i) for i in inp_x_lst]
plt.plot(x_lst, y_lst, 'o', inp_x_lst, y_res_lst)
plt.grid(True)
plt.show()


x_lst = np.array([0.45, 0.47, 0.52, 0.61, 0.66, 0.70, 0.74, 0.79], dtype=float)
y_lst = np.array([2.5742, 2.3251, 2.0934, 1.8620, 1.7493, 1.6210, 1.3418, 1.1212], dtype=float)
input_x = 0.528

#---------------------------------------------------------------------------------------------------------


def aitkens_scheme(x, y, inp_x, j):
    p_lst = []
    for i in range(len(y) - 1):
        p_lst.append(round((1/(x[i+j] - x[i]))*np.linalg.det([[y[i], x[i] - inp_x], [y[i+1], x[i+j] - inp_x]]), 3))
    for k in range(len(p_lst)):
        if p_lst[k] == y[k]:
            return p_lst[k]
    return p_lst


res_p = y_lst
for j in range(1, len(y_lst) - 1):
    res_p = aitkens_scheme(x_lst, res_p, input_x, j)
    if isinstance(res_p, float):
        print(f"result: {res_p}")
        break
