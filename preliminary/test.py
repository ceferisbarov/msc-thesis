import scipy

a = [1,2,3]
b = [2,3,4]
out = scipy.special.kl_div(a, b)
print(out)