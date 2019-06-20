import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.animation as animation

#parameter setting

c = 10
k = 1
lamb = 2*np.pi/k
w = c*k
Boxsize = 15
L = Boxsize*(lamb)
N = 300
T = 2000
dx = L/N
x = np.linspace(0,N*dx/lamb,N)
co_num = 0.1
dt = co_num*dx/c
def f(x):
	if(x<4*lamb):
		return np.sqrt(x/(4*lamb))
	elif(x>(L-4*lamb)):
		return np.sqrt((L-x)/(4*lamb))
	else:
		return 1
func = np.zeros(N)
for i in range(0,N):
	func[i]=f(i*dx)
plt.plot(x,func)
plt.show()
ims = []
Ey = np.zeros(N)
Bz = np.zeros(N)
Eyt = np.zeros(N)
Bzt = np.zeros(N)
Eytt = np.zeros(N)
Bztt = np.zeros(N)
fE1 = np.zeros(N)
fE2 = np.zeros(N)
fE3 = np.zeros(N)
fE4 = np.zeros(N)
fB1 = np.zeros(N)
fB2 = np.zeros(N)
fB3 = np.zeros(N)
fB4 = np.zeros(N)
fEy = np.zeros(N)
fBz = np.zeros(N)
kE1 = np.zeros(N)
kE2 = np.zeros(N)
kE3 = np.zeros(N)
kE4 = np.zeros(N)
kB1 = np.zeros(N)
kB2 = np.zeros(N)
kB3 = np.zeros(N)
kB4 = np.zeros(N)
types = "broad"
#main loop
print('delta t is : ',dt)
print('The walking length is : ',T*c*dt)
print('Total length is : ',L)
if (types == "steady"):
	for i in range(10,210):
		Ey[i] = np.sin(k*(i-10)*dx)
		Bz[i] = np.sin(k*(i-10)*dx)/c
	#plt.plot(x,Ey,x,Bz)
	#plt.plot(x,Bz)
	#plt.show()
	#plt.close()
	fig=plt.figure(figsize=(8,5))
	for t in range(0,T):
		#prepape difference
		#dy/dt = f(y,t)
		#prepare k1 = dt * f(t,y)
		if(t<3):
			for i in range(2,N-2):
				kE1[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
				kB1[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
			for i in range(0,N):
				Eyt[i] = Ey[i] - kB1[i]*c*c*0.5*dt
				Bzt[i] = Bz[i] - kE1[i]*0.5*dt
			for i in range(2,N-2):
				kE2[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB2[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(0,N):
				Eyt[i] = Ey[i] - kB2[i]*c*c*0.5*dt
				Bzt[i] = Bz[i] - kE2[i]*0.5*dt
			for i in range(2,N-2):
				kE3[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB3[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(0,N):
				Eyt[i] = Ey[i] - fBz[i]*c*c*dt
				Bzt[i] = Bz[i] - fEy[i]*dt
			for i in range(2,N-2):
				kE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(0,N):
				Ey[i] = Ey[i] - (1./6.)*(kB1[i]+2*kB2[i]+2*kB3[i]+kB4[i])*c*c*dt
				Bz[i] = Bz[i] - (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])*dt
			if (t==0):
				for i in range(0,N):
					fE1[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB1[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
			elif (t==1):
				for i in range(0,N):
					fE2[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB2[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
			elif (t==2):
				for i in range(0,N):
					fE3[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB3[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
		else:
			ii=0
			for i in range(2,N-2):
				fE4[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
				fB4[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
			for i in range(0,N):
				Eyt[i] = Ey[i] - ((-3./8.)*fB1[i]+(37./24.)*fB2[i]+(-59./24.)*fB3[i]+(55./24.)*fB4[i])*c*c*dt
				Bzt[i] = Bz[i] - ((-3./8.)*fE1[i]+(37./24.)*fE2[i]+(-59./24.)*fE3[i]+(55./24.)*fE4[i])*dt
			for i in range(0,N):
				fE1[i]=fE2[i]
				fE2[i]=fE3[i]
				fE3[i]=fE4[i]
				fB1[i]=fB2[i]
				fB2[i]=fB3[i]
				fB3[i]=fB4[i]
			checker = 1e-6
			while(ii<20 and checker>1e-15):
				ii = ii+1
				for i in range(0,N):
					Eytt[i] = Eyt[i]
					Bztt[i] = Bzt[i]
				for i in range(2,N-2):
					fE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					fB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(0,N):
					Eyt[i] = Ey[i] - ((1./24.)*fB1[i]+(-5./24.)*fB2[i]+(19./24.)*fB3[i]+(9./24.)*fB4[i])*c*c*dt
					Bzt[i] = Bz[i] - ((1./24.)*fE1[i]+(-5./24.)*fE2[i]+(19./24.)*fE3[i]+(9./24.)*fE4[i])*dt
				checker = 0
				for i in range(0,N):
					checker = checker + np.abs(Eytt[i]-Eyt[i]) + np.abs(Bztt[i]-Bzt[i])
				checker = checker/(2*N)
			print(t,ii,checker)
			for i in range(0,N):
				Ey[i] = Eyt[i]*func[i]
				Bz[i] = Bzt[i]*func[i]
		if(t%10 == 0): 
			im=plt.plot(x,Ey,'r',x,Bz,'g',x,func,'b')
			ims.append(im)
		#plt.show()
		#plt.plot(x,Bz)
		#plt.plot(x,func)
	#plt.show()
	#plt.close()
	ani = animation.ArtistAnimation(fig, ims, interval=10, repeat_delay=1000)
	ani.save("test7.gif",writer='pillow')
	#plt.show()
elif (types == "point"):
	for t in range(0,T):
		Ey[0] = np.sin(k*(0)*dx-w*t*dt)
		Bz[0] = np.sin(k*(0)*dx-w*t*dt)/c
		left_E1 = np.sin(k*(-1)*dx-w*t*dt)
		left_E2 = np.sin(k*(-2)*dx-w*t*dt)
		left_B1 = np.sin(k*(-1)*dx-w*t*dt)/c
		left_B2 = np.sin(k*(-2)*dx-w*t*dt)/c
		fEy[0] = ((-1/12.)*Ey[2]+(2./3.)*Ey[1]-(2./3.)*left_E1+(1./12.)*left_E2)/(dx)
		fEy[1] = ((-1/12.)*Ey[3]+(2./3.)*Ey[2]-(2./3.)*Ey[0]+(1./12.)*left_E1)/(dx)
		fBz[0] = ((-1/12.)*Bz[2]+(2./3.)*Bz[1]-(2./3.)*left_B1+(1./12.)*left_B2)/(dx)
		fBz[1] = ((-1/12.)*Bz[3]+(2./3.)*Bz[2]-(2./3.)*Bz[0]+(1./12.)*left_B1)/(dx)
		#prepape difference
		#dy/dt = f(y,t)
		#prepare k1 = dt * f(t,y)
		if(t<3):
			Ey[0] = np.sin(k*(0)*dx-w*t*dt)
			Bz[0] = np.sin(k*(0)*dx-w*t*dt)/c
			left_E1 = np.sin(k*(-1)*dx-w*t*dt)
			left_E2 = np.sin(k*(-2)*dx-w*t*dt)
			left_B1 = np.sin(k*(-1)*dx-w*t*dt)/c
			left_B2 = np.sin(k*(-2)*dx-w*t*dt)/c
			kE1[0] = ((-1/12.)*Ey[2]+(2./3.)*Ey[1]-(2./3.)*left_E1+(1./12.)*left_E2)/(dx)
			kE1[1] = ((-1/12.)*Ey[3]+(2./3.)*Ey[2]-(2./3.)*Ey[0]+(1./12.)*left_E1)/(dx)
			kB1[0] = ((-1/12.)*Bz[2]+(2./3.)*Bz[1]-(2./3.)*left_B1+(1./12.)*left_B2)/(dx)
			kB1[1] = ((-1/12.)*Bz[3]+(2./3.)*Bz[2]-(2./3.)*Bz[0]+(1./12.)*left_B1)/(dx)
			for i in range(2,N-2):
				kE1[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
				kB1[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
			for i in range(0,N):
				Eyt[i] = Ey[i] - kB1[i]*c*c*0.5*dt
				Bzt[i] = Bz[i] - kE1[i]*0.5*dt
			left_E1 = np.sin(k*(-1)*dx-w*(t+0.5)*dt)
			left_E2 = np.sin(k*(-2)*dx-w*(t+0.5)*dt)
			left_B1 = np.sin(k*(-1)*dx-w*(t+0.5)*dt)/c
			left_B2 = np.sin(k*(-2)*dx-w*(t+0.5)*dt)/c
			kE2[0] = ((-1/12.)*Eyt[2]+(2./3.)*Eyt[1]-(2./3.)*left_E1+(1./12.)*left_E2)/(dx)
			kE2[1] = ((-1/12.)*Eyt[3]+(2./3.)*Eyt[2]-(2./3.)*Eyt[0]+(1./12.)*left_E1)/(dx)
			kB2[0] = ((-1/12.)*Bzt[2]+(2./3.)*Bzt[1]-(2./3.)*left_B1+(1./12.)*left_B2)/(dx)
			kB2[1] = ((-1/12.)*Bzt[3]+(2./3.)*Bzt[2]-(2./3.)*Bzt[0]+(1./12.)*left_B1)/(dx)
			for i in range(2,N-2):
				kE2[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB2[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(0,N):
				Eyt[i] = Ey[i] - kB2[i]*c*c*0.5*dt
				Bzt[i] = Bz[i] - kE2[i]*0.5*dt
			left_E1 = np.sin(k*(-1)*dx-w*(t+0.5)*dt)
			left_E2 = np.sin(k*(-2)*dx-w*(t+0.5)*dt)
			left_B1 = np.sin(k*(-1)*dx-w*(t+0.5)*dt)/c
			left_B2 = np.sin(k*(-2)*dx-w*(t+0.5)*dt)/c
			kE3[0] = ((-1/12.)*Eyt[2]+(2./3.)*Eyt[1]-(2./3.)*left_E1+(1./12.)*left_E2)/(dx)
			kE3[1] = ((-1/12.)*Eyt[3]+(2./3.)*Eyt[2]-(2./3.)*Eyt[0]+(1./12.)*left_E1)/(dx)
			kB3[0] = ((-1/12.)*Bzt[2]+(2./3.)*Bzt[1]-(2./3.)*left_B1+(1./12.)*left_B2)/(dx)
			kB3[1] = ((-1/12.)*Bzt[3]+(2./3.)*Bzt[2]-(2./3.)*Bzt[0]+(1./12.)*left_B1)/(dx)
			for i in range(2,N-2):
				kE3[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB3[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(0,N):
				Eyt[i] = Ey[i] - fBz[i]*c*c*dt
				Bzt[i] = Bz[i] - fEy[i]*dt
			left_E1 = np.sin(k*(-1)*dx-w*(t+1)*dt)
			left_E2 = np.sin(k*(-2)*dx-w*(t+1)*dt)
			left_B1 = np.sin(k*(-1)*dx-w*(t+1)*dt)/c
			left_B2 = np.sin(k*(-2)*dx-w*(t+1)*dt)/c
			kE2[0] = ((-1/12.)*Eyt[2]+(2./3.)*Eyt[1]-(2./3.)*left_E1+(1./12.)*left_E2)/(dx)
			kE2[1] = ((-1/12.)*Eyt[3]+(2./3.)*Eyt[2]-(2./3.)*Eyt[0]+(1./12.)*left_E1)/(dx)
			kB2[0] = ((-1/12.)*Bzt[2]+(2./3.)*Bzt[1]-(2./3.)*left_B1+(1./12.)*left_B2)/(dx)
			kB2[1] = ((-1/12.)*Bzt[3]+(2./3.)*Bzt[2]-(2./3.)*Bzt[0]+(1./12.)*left_B1)/(dx)
			for i in range(2,N-2):
				kE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(0,N):
				Ey[i] = Ey[i] - (1./6.)*(kB1[i]+2*kB2[i]+2*kB3[i]+kB4[i])*c*c*dt
				Bz[i] = Bz[i] - (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])*dt
			if (t==0):
				for i in range(0,N):
					fE1[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB1[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
			elif (t==1):
				for i in range(0,N):
					fE2[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB2[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
			elif (t==2):
				for i in range(0,N):
					fE3[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB3[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
		else:
			ii=0
			Ey[0] = np.sin(k*(0)*dx-w*t*dt)
			Bz[0] = np.sin(k*(0)*dx-w*t*dt)/c
			left_E1 = np.sin(k*(-1)*dx-w*t*dt)
			left_E2 = np.sin(k*(-2)*dx-w*t*dt)
			left_B1 = np.sin(k*(-1)*dx-w*t*dt)/c
			left_B2 = np.sin(k*(-2)*dx-w*t*dt)/c
			fE4[0] = ((-1/12.)*Ey[2]+(2./3.)*Ey[1]-(2./3.)*left_E1+(1./12.)*left_E2)/(dx)
			fE4[1] = ((-1/12.)*Ey[3]+(2./3.)*Ey[2]-(2./3.)*Ey[0]+(1./12.)*left_E1)/(dx)
			fB4[0] = ((-1/12.)*Bz[2]+(2./3.)*Bz[1]-(2./3.)*left_B1+(1./12.)*left_B2)/(dx)
			fB4[1] = ((-1/12.)*Bz[3]+(2./3.)*Bz[2]-(2./3.)*Bz[0]+(1./12.)*left_B1)/(dx)
			for i in range(2,N-2):
				fE4[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
				fB4[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
			for i in range(0,N):
				Eyt[i] = Ey[i] - ((-3./8.)*fB1[i]+(37./24.)*fB2[i]+(-59./24.)*fB3[i]+(55./24.)*fB4[i])*c*c*dt
				Bzt[i] = Bz[i] - ((-3./8.)*fE1[i]+(37./24.)*fE2[i]+(-59./24.)*fE3[i]+(55./24.)*fE4[i])*dt
			for i in range(0,N):
				fE1[i]=fE2[i]
				fE2[i]=fE3[i]
				fE3[i]=fE4[i]
				fB1[i]=fB2[i]
				fB2[i]=fB3[i]
				fB3[i]=fB4[i]
			checker = 1e-6
			while(ii<20 and checker>1e-15):
				ii = ii+1
				for i in range(0,N):
					Eytt[i] = Eyt[i]
					Bztt[i] = Bzt[i]
				Ey[0] = np.sin(k*(0)*dx-w*t*dt)
				Bz[0] = np.sin(k*(0)*dx-w*t*dt)/c
				left_E1 = np.sin(k*(-1)*dx-w*(t+1)*dt)
				left_E2 = np.sin(k*(-2)*dx-w*(t+1)*dt)
				left_B1 = np.sin(k*(-1)*dx-w*(t+1)*dt)/c
				left_B2 = np.sin(k*(-2)*dx-w*(t+1)*dt)/c
				fEy[0] = ((-1/12.)*Eyt[2]+(2./3.)*Eyt[1]-(2./3.)*left_E1+(1./12.)*left_E2)/(dx)
				fEy[1] = ((-1/12.)*Eyt[3]+(2./3.)*Eyt[2]-(2./3.)*Eyt[0]+(1./12.)*left_E1)/(dx)
				fBz[0] = ((-1/12.)*Bzt[2]+(2./3.)*Bzt[1]-(2./3.)*left_B1+(1./12.)*left_B2)/(dx)
				fBz[1] = ((-1/12.)*Bzt[3]+(2./3.)*Bzt[2]-(2./3.)*Bzt[0]+(1./12.)*left_B1)/(dx)
				for i in range(2,N-2):
					fE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					fB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(0,N):
					Eyt[i] = Ey[i] - ((1./24.)*fB1[i]+(-5./24.)*fB2[i]+(19./24.)*fB3[i]+(9./24.)*fB4[i])*c*c*dt
					Bzt[i] = Bz[i] - ((1./24.)*fE1[i]+(-5./24.)*fE2[i]+(19./24.)*fE3[i]+(9./24.)*fE4[i])*dt
				checker = 0
				for i in range(0,N):
					checker = checker + np.abs(Eytt[i]-Eyt[i]) + np.abs(Bztt[i]-Bzt[i])
				checker = checker/(2*N)
			print(t,ii,checker)
			for i in range(0,N):
				Ey[i] = Eyt[i] 
				Bz[i] = Bzt[i] 
	plt.figure(figsize=(8,5))
	plt.plot(x,Ey)
	plt.plot(x,Bz)
	plt.xlabel(r"Real Space$(\lambda)$",FontSize=14)
	plt.ylabel("Amplitude",FontSize=14)
	timer = str(T*dt*c/lamb)
	plt.text((N*dx-10*lamb)/lamb, .10, 'Period is : '+timer, {'color': 'r', 'fontsize': 12})
	plt.text((N*dx-10*lamb)/lamb, -0.10, 'Speed of Light is : '+str(c), {'color': 'r', 'fontsize': 12})
	plt.title("Point Source test",FontSize=14)
	label=[r'$E_y$',r'$B_z$']
	plt.legend(label)
	plt.savefig("Point_Source_Test_1.png")
	plt.show()
	plt.close()
elif (types == "broad"):
	r_start = 80
	r_end = r_start + 20*3
	fig = plt.figure(figsize=(8,5))
	for t in range(0,T):
		#prepape difference
		#dy/dt = f(y,t)
		#prepare k1 = dt * f(t,y)
		if(t<3):
			for i in range(r_start,r_end):
				Ey[i] = np.sin(k*(i)*dx-w*(t)*dt)
				Bz[i] = np.sin(k*(i)*dx-w*(t)*dt)/c
			for i in range(2,N-2):
				kE1[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
				kB1[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
			for i in range(r_start,r_end):
				kE1[i] = k*np.cos(k*(i)*dx-w*(t)*dt)
				kB1[i] = k*np.cos(k*(i)*dx-w*(t)*dt)/c
			for i in range(0,N):
				Eyt[i] = Ey[i] - kB1[i]*c*c*0.5*dt
				Bzt[i] = Bz[i] - kE1[i]*0.5*dt
			for i in range(2,N-2):
				kE2[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB2[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(r_start,r_end):
				kE2[i] = k*np.cos(k*(i)*dx-w*(t+0.5)*dt)
				kB2[i] = k*np.cos(k*(i)*dx-w*(t+0.5)*dt)/c
			for i in range(0,N):
				Eyt[i] = Ey[i] - kB2[i]*c*c*0.5*dt
				Bzt[i] = Bz[i] - kE2[i]*0.5*dt
			for i in range(2,N-2):
				kE3[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB3[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(r_start,r_end):
				kE3[i] = k*np.cos(k*(i)*dx-w*(t+0.5)*dt)
				kB3[i] = k*np.cos(k*(i)*dx-w*(t+0.5)*dt)/c
			for i in range(0,N):
				Eyt[i] = Ey[i] - fBz[i]*c*c*dt
				Bzt[i] = Bz[i] - fEy[i]*dt
			for i in range(2,N-2):
				kE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
				kB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
			for i in range(r_start,r_end):
				kE4[i] = k*np.cos(k*(i)*dx-w*(t+1)*dt)
				kB4[i] = k*np.cos(k*(i)*dx-w*(t+1)*dt)/c
			for i in range(0,N):
				Ey[i] = Ey[i] - (1./6.)*(kB1[i]+2*kB2[i]+2*kB3[i]+kB4[i])*c*c*dt
				Bz[i] = Bz[i] - (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])*dt
			if (t==0):
				for i in range(0,N):
					fE1[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB1[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
			elif (t==1):
				for i in range(0,N):
					fE2[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB2[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
			elif (t==2):
				for i in range(0,N):
					fE3[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB3[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
		else:
			ii=0
			for i in range(r_start,r_end):
				Ey[i] = np.sin(k*(i)*dx-w*(t)*dt)
				Bz[i] = np.sin(k*(i)*dx-w*(t)*dt)/c
			for i in range(2,N-2):
				fE4[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
				fB4[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
			for i in range(r_start,r_end):
				kE4[i] = k*np.cos(k*(i)*dx-w*(t)*dt)
				kB4[i] = k*np.cos(k*(i)*dx-w*(t)*dt)/c
			for i in range(0,N):
				Eyt[i] = Ey[i] - ((-3./8.)*fB1[i]+(37./24.)*fB2[i]+(-59./24.)*fB3[i]+(55./24.)*fB4[i])*c*c*dt
				Bzt[i] = Bz[i] - ((-3./8.)*fE1[i]+(37./24.)*fE2[i]+(-59./24.)*fE3[i]+(55./24.)*fE4[i])*dt
			for i in range(0,N):
				fE1[i]=fE2[i]
				fE2[i]=fE3[i]
				fE3[i]=fE4[i]
				fB1[i]=fB2[i]
				fB2[i]=fB3[i]
				fB3[i]=fB4[i]
			checker = 1e-6
			while(ii<20 and checker>1e-15):
				ii = ii+1
				for i in range(0,N):
					Eytt[i] = Eyt[i]
					Bztt[i] = Bzt[i]
				for i in range(2,N-2):
					fE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					fB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(r_start,r_end):
					fE4[i] = k*np.cos(k*(i)*dx-w*(t+1)*dt)
					fB4[i] = k*np.cos(k*(i)*dx-w*(t+1)*dt)/c
				for i in range(0,N):
					Eyt[i] = Ey[i] - ((1./24.)*fB1[i]+(-5./24.)*fB2[i]+(19./24.)*fB3[i]+(9./24.)*fB4[i])*c*c*dt
					Bzt[i] = Bz[i] - ((1./24.)*fE1[i]+(-5./24.)*fE2[i]+(19./24.)*fE3[i]+(9./24.)*fE4[i])*dt
				checker = 0
				for i in range(0,N):
					checker = checker + np.abs(Eytt[i]-Eyt[i]) + np.abs(Bztt[i]-Bzt[i])
				checker = checker/(2*N)
			print(t,ii,checker)
			for i in range(0,N):
				Ey[i] = Eyt[i]*f(i*dx)
				Bz[i] = Bzt[i]*f(i*dx) 
		#plt.plot(x,Ey)
		#plt.plot(x,Bz)
		#plt.plot(x,func)
		if (t%10 == 0):
			im = plt.plot(x,Ey,'r',x,Bz,'g',x,func,'b')
			plt.xlabel(r"Real Space$(\lambda)$",FontSize=14)
			plt.ylabel("Amplitude",FontSize=14)
			timer = str(T*dt*c/lamb)
			plt.text((N*dx-10*lamb)/lamb, .10, 'Period is : '+timer, {'color': 'r', 'fontsize': 12})
			plt.text((N*dx-10*lamb)/lamb, -0.10, 'Speed of Light is : '+str(c), {'color': 'r', 'fontsize': 12})
			plt.title("Point Source test",FontSize=14)
			label=[r'$E_y$',r'$B_z$',r'$ABCs$']
			plt.legend(label)
			ims.append(im)
	ani = animation.ArtistAnimation(fig, ims, interval=10, repeat_delay=1000)
	ani.save("test8.gif",writer='pillow')
	#plt.savefig("Point_Source_Test_6.png")
	#plt.show()
	#plt.close()
elif (types == "broad2"):
	r_start = 80
	r_end = r_start + 20
	for t in range(0,T):
		#prepape difference
		#dy/dt = f(y,t)
		#prepare k1 = dt * f(t,y)
		if(t<3):
			if(t==0):
				for i in range(r_start,r_end):
					Ey[i] = np.sin(k*(i)*dx-w*(t)*dt)
					Bz[i] = np.sin(k*(i)*dx-w*(t)*dt)/c
				for i in range(2,N-2):
					kE1[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
					kB1[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
				for i in range(r_start,r_end):
					kE1[i] = k*np.cos(k*(i)*dx-w*(t)*dt)
					kB1[i] = k*np.cos(k*(i)*dx-w*(t)*dt)/c
				for i in range(0,N):
					Eyt[i] = Ey[i] - kB1[i]*c*c*0.5*dt
					Bzt[i] = Bz[i] - kE1[i]*0.5*dt
				for i in range(2,N-2):
					kE2[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					kB2[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(r_start,r_end):
					kE2[i] = k*np.cos(k*(i)*dx-w*(t+0.5)*dt)
					kB2[i] = k*np.cos(k*(i)*dx-w*(t+0.5)*dt)/c
				for i in range(0,N):
					Eyt[i] = Ey[i] - kB2[i]*c*c*0.5*dt
					Bzt[i] = Bz[i] - kE2[i]*0.5*dt
				for i in range(2,N-2):
					kE3[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					kB3[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(r_start,r_end):
					kE3[i] = k*np.cos(k*(i)*dx-w*(t+0.5)*dt)
					kB3[i] = k*np.cos(k*(i)*dx-w*(t+0.5)*dt)/c
				for i in range(0,N):
					Eyt[i] = Ey[i] - fBz[i]*c*c*dt
					Bzt[i] = Bz[i] - fEy[i]*dt
				for i in range(2,N-2):
					kE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					kB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(r_start,r_end):
					kE4[i] = k*np.cos(k*(i)*dx-w*(t+1)*dt)
					kB4[i] = k*np.cos(k*(i)*dx-w*(t+1)*dt)/c
				for i in range(0,N):
					Ey[i] = Ey[i] - (1./6.)*(kB1[i]+2*kB2[i]+2*kB3[i]+kB4[i])*c*c*dt
					Bz[i] = Bz[i] - (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])*dt
			else:
				for i in range(2,N-2):
					kE1[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
					kB1[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
				for i in range(0,N):
					Eyt[i] = Ey[i] - kB1[i]*c*c*0.5*dt
					Bzt[i] = Bz[i] - kE1[i]*0.5*dt
				for i in range(2,N-2):
					kE2[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					kB2[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(0,N):
					Eyt[i] = Ey[i] - kB2[i]*c*c*0.5*dt
					Bzt[i] = Bz[i] - kE2[i]*0.5*dt
				for i in range(2,N-2):
					kE3[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					kB3[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(0,N):
					Eyt[i] = Ey[i] - fBz[i]*c*c*dt
					Bzt[i] = Bz[i] - fEy[i]*dt
				for i in range(2,N-2):
					kE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
					kB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
				for i in range(0,N):
					Ey[i] = Ey[i] - (1./6.)*(kB1[i]+2*kB2[i]+2*kB3[i]+kB4[i])*c*c*dt
					Bz[i] = Bz[i] - (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])*dt
			if (t==0):
				for i in range(0,N):
					fE1[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB1[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
			elif (t==1):
				for i in range(0,N):
					fE2[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB2[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
			elif (t==2):
				for i in range(0,N):
					fE3[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
					fB3[i] = (1./6.)*(kE1[i]+2*kE2[i]+2*kE3[i]+kE4[i])
		else:
			ii=0
			if (t%200 == 0):
				for i in range(r_start,r_end):
					Ey[i] = np.sin(k*(i)*dx-w*(t)*dt)
					Bz[i] = np.sin(k*(i)*dx-w*(t)*dt)/c
				for i in range(2,N-2):
					fE4[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
					fB4[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
				for i in range(r_start,r_end):
					kE4[i] = k*np.cos(k*(i)*dx-w*(t)*dt)
					kB4[i] = k*np.cos(k*(i)*dx-w*(t)*dt)/c
				for i in range(0,N):
					Eyt[i] = Ey[i] - ((-3./8.)*fB1[i]+(37./24.)*fB2[i]+(-59./24.)*fB3[i]+(55./24.)*fB4[i])*c*c*dt
					Bzt[i] = Bz[i] - ((-3./8.)*fE1[i]+(37./24.)*fE2[i]+(-59./24.)*fE3[i]+(55./24.)*fE4[i])*dt
				for i in range(0,N):
					fE1[i]=fE2[i]
					fE2[i]=fE3[i]
					fE3[i]=fE4[i]
					fB1[i]=fB2[i]
					fB2[i]=fB3[i]
					fB3[i]=fB4[i]
				checker = 1e-6
				while(ii<20 and checker>1e-19):
					ii = ii+1
					for i in range(0,N):
						Eytt[i] = Eyt[i]
						Bztt[i] = Bzt[i]
					for i in range(2,N-2):
						fE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
						fB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
					for i in range(r_start,r_end):
						fE4[i] = k*np.cos(k*(i)*dx-w*(t+1)*dt)
						fB4[i] = k*np.cos(k*(i)*dx-w*(t+1)*dt)/c
					for i in range(0,N):
						Eyt[i] = Ey[i] - ((1./24.)*fB1[i]+(-5./24.)*fB2[i]+(19./24.)*fB3[i]+(9./24.)*fB4[i])*c*c*dt
						Bzt[i] = Bz[i] - ((1./24.)*fE1[i]+(-5./24.)*fE2[i]+(19./24.)*fE3[i]+(9./24.)*fE4[i])*dt
					checker = 0
					for i in range(0,N):
						checker = checker + np.abs(Eytt[i]-Eyt[i]) + np.abs(Bztt[i]-Bzt[i])
					checker = checker/(2*N)
				print(t,ii,checker)
				for i in range(0,N):
					Ey[i] = Eyt[i]*f(i*dx)
					Bz[i] = Bzt[i]*f(i*dx) 
			else:
				for i in range(2,N-2):
					fE4[i] = ((-1/12.)*Ey[i+2]+(2./3.)*Ey[i+1]-(2./3.)*Ey[i-1]+(1./12.)*Ey[i-2])/(dx)
					fB4[i] = ((-1/12.)*Bz[i+2]+(2./3.)*Bz[i+1]-(2./3.)*Bz[i-1]+(1./12.)*Bz[i-2])/(dx)
				for i in range(r_start,r_end):
					kE4[i] = k*np.cos(k*(i)*dx-w*(t)*dt)
					kB4[i] = k*np.cos(k*(i)*dx-w*(t)*dt)/c
				for i in range(0,N):
					Eyt[i] = Ey[i] - ((-3./8.)*fB1[i]+(37./24.)*fB2[i]+(-59./24.)*fB3[i]+(55./24.)*fB4[i])*c*c*dt
					Bzt[i] = Bz[i] - ((-3./8.)*fE1[i]+(37./24.)*fE2[i]+(-59./24.)*fE3[i]+(55./24.)*fE4[i])*dt
				for i in range(0,N):
					fE1[i]=fE2[i]
					fE2[i]=fE3[i]
					fE3[i]=fE4[i]
					fB1[i]=fB2[i]
					fB2[i]=fB3[i]
					fB3[i]=fB4[i]
				checker = 1e-6
				while(ii<20 and checker>1e-19):
					ii = ii+1
					for i in range(0,N):
						Eytt[i] = Eyt[i]
						Bztt[i] = Bzt[i]
					for i in range(2,N-2):
						fE4[i] = ((-1/12.)*Eyt[i+2]+(2./3.)*Eyt[i+1]-(2./3.)*Eyt[i-1]+(1./12.)*Eyt[i-2])/(dx)
						fB4[i] = ((-1/12.)*Bzt[i+2]+(2./3.)*Bzt[i+1]-(2./3.)*Bzt[i-1]+(1./12.)*Bzt[i-2])/(dx)
					for i in range(0,N):
						Eyt[i] = Ey[i] - ((1./24.)*fB1[i]+(-5./24.)*fB2[i]+(19./24.)*fB3[i]+(9./24.)*fB4[i])*c*c*dt
						Bzt[i] = Bz[i] - ((1./24.)*fE1[i]+(-5./24.)*fE2[i]+(19./24.)*fE3[i]+(9./24.)*fE4[i])*dt
					checker = 0
					for i in range(0,N):
						checker = checker + np.abs(Eytt[i]-Eyt[i]) + np.abs(Bztt[i]-Bzt[i])
					checker = checker/(2*N)
				print(t,ii,checker)
				for i in range(0,N):
					Ey[i] = Eyt[i]*f(i*dx)
					Bz[i] = Bzt[i]*f(i*dx)
	plt.figure(figsize=(8,5))
	plt.plot(x,Ey)
	plt.plot(x,Bz)
	plt.plot(x,func)
	plt.xlabel(r"Real Space$(\lambda)$",FontSize=14)
	plt.ylabel("Amplitude",FontSize=14)
	timer = str(T*dt*c/lamb)
	#plt.text((N*dx-10*lamb)/lamb, .10, 'Period is : '+timer, {'color': 'r', 'fontsize': 12})
	#plt.text((N*dx-10*lamb)/lamb, -0.10, 'Speed of Light is : '+str(c), {'color': 'r', 'fontsize': 12})
	plt.title("Point Source test",FontSize=14)
	label=[r'$E_y$',r'$B_z$',r'$ABCs$']
	plt.legend(label)
	plt.savefig("Point_Source_Test_5.png")
	plt.show()
	plt.close()