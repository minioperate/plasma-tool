import numpy as np
import matplotlib.pyplot as plt

def RK4(f,fp,funcfp,dt,dx,BC,diff_type):
	k1 = np.zeros(len(f))
	k2 = np.zeros(len(f))
	k3 = np.zeros(len(f))
	k4 = np.zeros(len(f))
	ft = np.zeros(len(f))
	funcfp(f,fp,dx,BC,diff_type)
	for i in range(0,len(f)):
		k1[i]=fp[i]*dt
	for i in range(0,len(f)):
		ft[i]=f[i]+0.5*k1[i]
	funcfp(ft,fp,dx,BC,diff_type)
	for i in range(0,len(f)):
		k2[i]=fp[i]*dt
	for i in range(0,len(f)):
		ft[i]=f[i]+0.5*k2[i]
	funcfp(ft,fp,dx,BC,diff_type)
	for i in range(0,len(f)):
		k3[i]=fp[i]*dt
	for i in range(0,len(f)):
		ft[i]=f[i]+k3[i]
	funcfp(ft,fp,dx,BC,diff_type)
	for i in range(0,len(f)):
		k4[i]=fp[i]*dt
	for i in range(0,len(f)):
		f[i]=f[i]+(k1[i]+2.*k2[i]+2.*k3[i]+k4[i])/6.
def PC_Adam(f,fp,funcfp,dt,f1,f2,f3,dx,BC,diff_type):
	ft = np.zeros(len(f))
	funcfp(f,fp,dx,BC,diff_type)
	for i in range(0,len(f)):
		ft[i]=f[i]+(dt/24.)*(55.*fp[i]-59.*f1[i]+37.*f2[i]-9.*f3[i])
	for i in range(0,len(fp)):
		f3[i]=f2[i]
		f2[i]=f1[i]
		f1[i]=fp[i]
	ftt = np.zeros(len(f))
	checker = 10
	ii=0
	while(ii<30 and checker>1e-15):
		ii = ii+1
		for i in range(0,len(f)):
			ftt[i] = ft[i]
		funcfp(ft,fp,dx,BC,diff_type)
		for i in range(0,len(f)):
			ft[i]=f[i]+(dt/24.)*(9.*fp[i]+19.*f1[i]-5.*f2[i]+f3[i])
		checker = 0
		for i in range(0,len(f)):
			checker = checker + np.abs(ftt[i]-ft[i])
		checker = checker/(len(f))
		#print(ii,checker)
	for i in range(0,len(f)):
		f[i]=ft[i]
	print(ii,checker)