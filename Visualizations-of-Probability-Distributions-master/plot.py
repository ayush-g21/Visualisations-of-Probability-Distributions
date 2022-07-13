import matplotlib.pyplot as plt
N=[128,256,512]
abc=[9.202,60.661,561.091]
acb=[12.181,74.937,623.062]
bac=[11.845,67.014,694.912]
bca=[11.142,68.530,693.338]
cab=[10.871,64.240,681.309]
cba=[10.575,74.323,702.602]
plt.plot(N,abc,color="red",label="abc")
plt.plot(N,acb,color="blue",label="acb")
plt.plot(N,bca,color="cyan",label="bca")
plt.plot(N,cab,color="magenta",label="cab")
plt.plot(N,cba,color="black",label="cba")
plt.legend()
plt.xlabel("Number of rows in matrix")
plt.ylabel("Time taken for multiplication (in ms)")
plt.show()