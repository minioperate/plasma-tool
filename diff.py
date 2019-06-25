import numpy as np
import matplotlib.pyplot as plt
import math

def diff(f,fp,dx,BC='uniform',diff_type = 'central'):
	coeff = np.zeros((7,len(f)),dtype=int)
	if(BC == 'periodic'):
		for i in range(0,len(f)):
			temp = np.zeros(7)
			temp[0] = i-3
			temp[1] = i-2
			temp[2] = i-1
			temp[3] = i
			temp[4] = i+1
			temp[5] = i+2
			temp[6] = i+3
			for j in range(0,7):
				if (temp[j]<0):
					temp[j] = temp[j]+len(f)
				elif (temp[j]>(len(f)-1)):
					temp[j] = temp[j]-len(f)
	elif(BC == 'uniform'):
		for i in range(0,len(f)):
			#print(len(f),i)
			temp = np.zeros(7)
			temp.astype(int)
			temp[0] = i-3
			temp[1] = i-2
			temp[2] = i-1
			temp[3] = i
			temp[4] = i+1
			temp[5] = i+2
			temp[6] = i+3
			for j in range(0,7):
				if (temp[j]<0):
					temp[j] = 0
				elif (temp[j]>(len(f)-1)):
					temp[j] = len(f)-1
				coeff[j,i]=np.int16(temp[j])
				#np.int16(coeff[j,i])
				#print(temp[j],coeff[j,i])
	if(diff_type == 'central'):
		for i in range(0,len(f)):
			fp[i]=(-1./60.)*f[coeff[0,i]]+(3./20.)*f[coeff[1,i]]+(-3./4.)*f[coeff[2,i]]+(3./4.)*f[coeff[4,i]]+(-3./20.)*f[coeff[5,i]]+(1./60.)*f[coeff[6,i]]
			fp[i]=fp[i]/dx
	elif(diff_type == 'CWENO'):
		eps = 1e-6
		#f0 = (1./3.)*f[i-2]+(-7./6.)*f[i-1]+(11./6.)*f[i]
		#f1 = (-1./6.)*f[i-1]+(5./6.)*f[i]+(1./3.)*f[i+1]
		#f1 = (1./3.)*f[i]+(5./6.)*f[i+1]+(-1./6.)*f[i+2]
		for i in range(0,len(f)):
			fp_jm1 = (f[coeff[3,i]]-f[coeff[1,i]])/(2*dx)
			fp_j = (f[coeff[4,i]]-f[coeff[2,i]])/(2*dx)
			fp_jp1 = (f[coeff[5,i]]-f[coeff[3,i]])/(2*dx)
			fpp_jm1 = (f[coeff[3,i]]-2*f[coeff[2,i]]+f[coeff[1,i]])/(dx**2)
			fpp_jp1 = (f[coeff[5,i]]-2*f[coeff[4,i]]+f[coeff[3,i]])/(dx**2)
			IS0 = 13./12.*(f[coeff[1,i]]-2*f[coeff[2,i]]+f[coeff[3,i]])**2+0.25*(f[coeff[1,i]]-4*f[coeff[2,i]]+3*f[coeff[3,i]])**2
			IS1 = 13./12.*(f[coeff[2,i]]-2*f[coeff[3,i]]+f[coeff[4,i]])**2+0.25*(f[coeff[2,i]]-f[coeff[4,i]])**2
			IS2 = 13./12.*(f[coeff[3,i]]-2*f[coeff[4,i]]+f[coeff[5,i]])**2+0.25*(3*f[coeff[3,i]]-4*f[coeff[4,i]]+f[coeff[5,i]])**2
			a0 = (3./16.)/(eps+IS0)**2
			a1 = (5./8.)/(eps+IS1)**2
			a2 = (3./16.)/(eps+IS2)**2
			w0 = a0/(a0+a1+a2)
			w1 = a1/(a0+a1+a2)
			w2 = a2/(a0+a1+a2)
			fp[i]=w0*(fp_jm1+dx*fpp_jm1)+w1*fp_j+w2*(fp_jp1-dx*fpp_jp1)
	elif(diff_type == 'WENO-5'):
		for i in range(0,len(f)):
			eps = 1e-6
			fp1 = (3./8.)*f[coeff[1,i]]-(5./4.)*f[coeff[2,i]]+(15./8.)*f[coeff[3,i]]
			fp2 = (-1./8.)*f[coeff[2,i]]+(3./4.)*f[coeff[3,i]]+(3./8.)*f[coeff[4,i]]
			fp3 = (3./8.)*f[coeff[3,i]]+(3./4.)*f[coeff[4,i]]-(1./8.)*f[coeff[5,i]]
			fm1 = (3./8.)*f[coeff[0,i]]-(5./4.)*f[coeff[1,i]]+(15./8.)*f[coeff[2,i]]
			fm2 = (-1./8.)*f[coeff[1,i]]+(3./4.)*f[coeff[2,i]]+(3./8.)*f[coeff[3,i]]
			fm3 = (3./8.)*f[coeff[2,i]]+(3./4.)*f[coeff[3,i]]-(1./8.)*f[coeff[4,i]]
			bp1 = (1./3.)*(4*f[coeff[1,i]]**2-19*f[coeff[1,i]]*f[coeff[2,i]]+25*f[coeff[2,i]]+11*f[coeff[1,i]]*f[coeff[3,i]]-31*f[coeff[2,i]]*f[coeff[3,i]]+10*f[coeff[3,i]]**2)
			bp2 = (1./3.)*(4*f[coeff[2,i]]**2-13*f[coeff[2,i]]*f[coeff[3,i]]+13*f[coeff[3,i]]**2+5*f[coeff[2,i]]*f[coeff[4,i]]-13*f[coeff[3,i]]*f[coeff[4,i]]+4*f[coeff[4,i]]**2)
			bp3 = (1./3.)*(10*f[coeff[3,i]]**2-31*f[coeff[3,i]]*f[coeff[4,i]]+25*f[coeff[4,i]]**2+11*f[coeff[3,i]]*f[coeff[5,i]]-19*f[coeff[4,i]]*f[coeff[5,i]]+4*f[coeff[5,i]]**2)
			bm1 = (1./3.)*(4*f[coeff[0,i]]**2-19*f[coeff[0,i]]*f[coeff[1,i]]+25*f[coeff[1,i]]+11*f[coeff[0,i]]*f[coeff[2,i]]-31*f[coeff[1,i]]*f[coeff[2,i]]+10*f[coeff[2,i]]**2)
			bm2 = (1./3.)*(4*f[coeff[1,i]]**2-13*f[coeff[1,i]]*f[coeff[2,i]]+13*f[coeff[2,i]]**2+5*f[coeff[1,i]]*f[coeff[3,i]]-13*f[coeff[2,i]]*f[coeff[3,i]]+4*f[coeff[3,i]]**2)
			bm3 = (1./3.)*(10*f[coeff[2,i]]**2-31*f[coeff[2,i]]*f[coeff[3,i]]+25*f[coeff[3,i]]**2+11*f[coeff[2,i]]*f[coeff[4,i]]-19*f[coeff[3,i]]*f[coeff[4,i]]+4*f[coeff[4,i]]**2)
			wwp1 = (1./16.)/(eps+bp1)**2
			wwp2 = (5./8.)/(eps+bp2)**2
			wwp3 = (5./16.)/(eps+bp3)**2
			wwm1 = (1./16.)/(eps+bm1)**2
			wwm2 = (5./8.)/(eps+bm2)**2
			wwm3 = (5./16.)/(eps+bm3)**2
			temp1 = wwp1+wwp2+wwp3
			temp2 = wwm1+wwm2+wwm3
			wp1 = wwp1/temp1
			wp2 = wwp2/temp1
			wp3 = wwp3/temp1
			wm1 = wwm1/temp2
			wm2 = wwm2/temp2
			wm3 = wwm3/temp2
			fpp=wp1*fp1+wp2*fp2+wp3*fp3
			fm=wm1*fm1+wm2*fm2+wm3*fm3
			fp[i] = (fpp-fm)/dx
	elif(diff_type == 'WENO-5-z'):
		eps = 1e-6
		#f0 = (1./3.)*f[i-2]+(-7./6.)*f[i-1]+(11./6.)*f[i]
		#f1 = (-1./6.)*f[i-1]+(5./6.)*f[i]+(1./3.)*f[i+1]
		#f1 = (1./3.)*f[i]+(5./6.)*f[i+1]+(-1./6.)*f[i+2]
		for i in range(0,len(f)):
			#print(i)
			fp1 = (3./8.)*f[coeff[1,i]]-(5./4.)*f[coeff[2,i]]+(15./8.)*f[coeff[3,i]]
			fp2 = (-1./8.)*f[coeff[2,i]]+(3./4.)*f[coeff[3,i]]+(3./8.)*f[coeff[4,i]]
			fp3 = (3./8.)*f[coeff[3,i]]+(3./4.)*f[coeff[4,i]]-(1./8.)*f[coeff[5,i]]
			fm1 = (3./8.)*f[coeff[0,i]]-(5./4.)*f[coeff[1,i]]+(15./8.)*f[coeff[2,i]]
			fm2 = (-1./8.)*f[coeff[1,i]]+(3./4.)*f[coeff[2,i]]+(3./8.)*f[coeff[3,i]]
			fm3 = (3./8.)*f[coeff[2,i]]+(3./4.)*f[coeff[3,i]]-(1./8.)*f[coeff[4,i]]
			bp1 = (1./3.)*(4*f[coeff[1,i]]**2-19*f[coeff[1,i]]*f[coeff[2,i]]+25*f[coeff[2,i]]+11*f[coeff[1,i]]*f[coeff[3,i]]-31*f[coeff[2,i]]*f[coeff[3,i]]+10*f[coeff[3,i]]**2)
			bp2 = (1./3.)*(4*f[coeff[2,i]]**2-13*f[coeff[2,i]]*f[coeff[3,i]]+13*f[coeff[3,i]]**2+5*f[coeff[2,i]]*f[coeff[4,i]]-13*f[coeff[3,i]]*f[coeff[4,i]]+4*f[coeff[4,i]]**2)
			bp3 = (1./3.)*(10*f[coeff[3,i]]**2-31*f[coeff[3,i]]*f[coeff[4,i]]+25*f[coeff[4,i]]**2+11*f[coeff[3,i]]*f[coeff[5,i]]-19*f[coeff[4,i]]*f[coeff[5,i]]+4*f[coeff[5,i]]**2)
			bm1 = (1./3.)*(4*f[coeff[0,i]]**2-19*f[coeff[0,i]]*f[coeff[1,i]]+25*f[coeff[1,i]]+11*f[coeff[0,i]]*f[coeff[2,i]]-31*f[coeff[1,i]]*f[coeff[2,i]]+10*f[coeff[2,i]]**2)
			bm2 = (1./3.)*(4*f[coeff[1,i]]**2-13*f[coeff[1,i]]*f[coeff[2,i]]+13*f[coeff[2,i]]**2+5*f[coeff[1,i]]*f[coeff[3,i]]-13*f[coeff[2,i]]*f[coeff[3,i]]+4*f[coeff[3,i]]**2)
			bm3 = (1./3.)*(10*f[coeff[2,i]]**2-31*f[coeff[2,i]]*f[coeff[3,i]]+25*f[coeff[3,i]]**2+11*f[coeff[2,i]]*f[coeff[4,i]]-19*f[coeff[3,i]]*f[coeff[4,i]]+4*f[coeff[4,i]]**2)
			taup = abs(bp1-bp3)
			taum = abs(bm1-bm3)
			bp1_n = (bp1+eps)/(bp1+taup+eps)
			bp2_n = (bp2+eps)/(bp2+taup+eps)
			bp3_n = (bp3+eps)/(bp3+taup+eps)
			bm1_n = (bm1+eps)/(bm1+taum+eps)
			bm2_n = (bm2+eps)/(bm2+taum+eps)
			bm3_n = (bm3+eps)/(bm3+taum+eps)
			wwp1 = (1./16.)/(eps+bp1_n)**2
			wwp2 = (5./8.)/(eps+bp2_n)**2
			wwp3 = (5./16.)/(eps+bp3_n)**2
			wwm1 = (1./16.)/(eps+bm1_n)**2
			wwm2 = (5./8.)/(eps+bm2_n)**2
			wwm3 = (5./16.)/(eps+bm3_n)**2
			temp1 = wwp1+wwp2+wwp3
			temp2 = wwm1+wwm2+wwm3
			wp1 = wwp1/temp1
			wp2 = wwp2/temp1
			wp3 = wwp3/temp1
			wm1 = wwm1/temp2
			wm2 = wwm2/temp2
			wm3 = wwm3/temp2
			fpp=wp1*fp1+wp2*fp2+wp3*fp3
			fm=wm1*fm1+wm2*fm2+wm3*fm3
			fp[i] = (fpp-fm)/dx