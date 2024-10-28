import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib.ticker import MultipleLocator
import matplotlib.image as mpimg
mpl.rcParams['font.family'] = 'FreeSerif'
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rc('axes', linewidth=1.5)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
mpl.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
class Publication_Drawer(object):
	"""docstring for Publication_Drawer"""
	def __init__(self, types,num_xlabel,num_ylabel,num_title):
		super(Publication_Drawer, self).__init__()
		# types = 'one_column','two_column','onehalf_column'
		self.type = types
		self.tlfs = mpl.rcParams['xtick.labelsize']
		if(self.type == "one_column"):
			self.figure_x = 3.5
			self.fontsize = 12
			self.tlfs = self.tlfs/2
		elif(self.type == "two_column"):
			self.figure_x = 7.5
			self.fontsize = 20
		elif(self.type == "onehalf_column"):
			self.figure_x = 5.5
			self.fontsize = 18
		else:
			self.figure_x = 5
			self.fontsize = 30
		self.dpi  = 600
		self.num_ylabel = num_ylabel
		self.num_xlabel = num_xlabel
		self.num_title  = num_title
		self.marker     = ['ro', 'gv', 'b8', 'ys','mp','c*']
		self.color      = ['r-', 'g-', 'b-', 'y-','m-','c-']
		self.color2     = ['b-', 'r--', 'b-', 'y-','m-','c-']
		self.cmap       = 'jet' 
	def set_figure(self,m,n,label_type,ratio):
		if(label_type == 'contour'):
			mpl.rc('axes', linewidth=1)
			self.L        = self.num_ylabel*self.fontsize/72+2*self.tlfs/72
			self.D        = self.num_xlabel*self.fontsize/72+2*self.tlfs/72
			self.U        = self.num_title*self.fontsize/72
			s_f_x         = (self.figure_x - self.L)/(1.1*m-0.1)
			x_size        = (s_f_x-2*self.tlfs/72)/1.1
			y_size        = ratio*x_size
			self.figure_y = self.D+self.U+(1.25*n-0.25)*y_size
			self.pox      = self.L/self.figure_x
			self.poy      = self.D/self.figure_y
			self.xpanel   = x_size/self.figure_x
			self.ypanel   = y_size/self.figure_y
			self.cpox     = (self.L+1.05*x_size)/self.figure_x
			self.cpoy     = self.D/self.figure_y
			self.cxpanel  = 0.07*x_size/self.figure_x
			self.cypanel  = y_size/self.figure_y
			self.po       = np.zeros((2,m,n))
			self.cpo      = np.zeros((2,m,n))
			for i in range(m):
				for j in range(n):
					self.po[0,i,j]  = self.pox+i*1.1*(s_f_x/self.figure_x)
					self.po[1,i,j]  = self.poy+j*1.25*self.ypanel
					self.cpo[0,i,j] = self.cpox+i*1.1*(s_f_x/self.figure_x)
					self.cpo[1,i,j] = self.cpoy+j*1.25*self.ypanel
		if(label_type == 'contour2'):
			mpl.rc('axes', linewidth=1)
			self.L        = self.num_ylabel*self.fontsize/72+2*self.tlfs/72
			self.D        = self.num_xlabel*self.fontsize/72+2*self.tlfs/72
			self.U        = self.num_title*self.fontsize/72
			s_f_x         = (self.figure_x - self.L-0.3)/(1.1*m-0.1)
			x_size        = (s_f_x)/1.05
			y_size        = ratio*x_size
			self.figure_y = self.D+self.U+(1.15*n-0.15)*y_size+0.2
			self.pox      = self.L/self.figure_x
			self.poy      = self.D/self.figure_y
			self.xpanel   = x_size/self.figure_x
			self.ypanel   = y_size/self.figure_y
			self.cpox     = (self.L)/self.figure_x
			self.cpoy     = self.D/self.figure_y
			self.cxpanel  = 0.05*x_size/self.figure_x
			self.cypanel  = 2.15*y_size/self.figure_y
			self.cypanel  = 1.0*y_size/self.figure_y
			self.po       = np.zeros((2,m,n))
			self.cpo      = np.zeros((2,m,n))
			for i in range(m):
				for j in range(n):
					self.po[0,i,j]  = self.pox+i*1.1*self.xpanel
					self.po[1,i,j]  = self.poy+j*1.15*self.ypanel
					self.cpo[0,i,j] = self.po[0,i,j]+1.05*self.xpanel
					self.cpo[1,i,j] = self.po[1,i,j-1]
		elif(label_type == 'line'):
			dx            = 1.8*(self.fontsize+1.1*self.tlfs)/72
			dy            = 1.05*(self.fontsize+2.5*self.tlfs)/72
			s_f_x         = self.figure_x/(1.03*m-0.03)
			x_size        = s_f_x - dx
			y_size        = ratio*x_size
			s_f_y         = y_size+dy
			self.figure_y = (1.03*n-0.03)*s_f_y
			self.pox      = dx/self.figure_x
			self.poy      = dy/self.figure_y
			self.xpanel   = x_size/self.figure_x-0.02
			self.ypanel   = y_size/self.figure_y-0.02
			self.po       = np.zeros((2,m,n))
			for i in range(m):
				for j in range(n):
					self.po[0,i,j] = self.pox+i*1.03*(s_f_x/self.figure_x)
					self.po[1,i,j] = self.poy+j*1.2*self.ypanel
		fig = plt.figure(figsize=(self.figure_x,self.figure_y),dpi=self.dpi)
		return fig

	def line_plot(self,fig,x,y,m,n,x_name,y_name,labels,bo):
		ax = fig.add_axes([self.po[0,m,n],self.po[1,m,n],self.xpanel,self.ypanel])
		plt.sca(ax)
		plt.yscale("log")
		plt.ylim([1e-16,1e-1])
		plt.xlim([np.amin(x),np.amax(x)])
		for i in range(len(y)):
			if( bo == 0):
				plt.plot(x,y[i,:],self.color[i],linewidth=2)
			else:
				plt.plot(x,y[i,:],self.marker[i],ms=4)
		plt.ylabel(y_name,fontsize=self.fontsize)
		if(m==0 and n == 0):
			if(bo != 0):
				handles, label = ax.get_legend_handles_labels()
				plt.legend(labels,loc='best')
		if(n == 0):
			plt.xlabel(x_name,fontsize=self.fontsize)
	def line_plot_1(self,fig,x,y,m,n,x_name,y_name,labels):
		ax = fig.add_axes([self.po[0,m,n],self.po[1,m,n],self.xpanel,self.ypanel])
		plt.sca(ax)
		plt.yscale("log")
		plt.ylim([1e-17,1e0])
		plt.xlim([np.amin(x),np.amax(x)])
		plt.grid(axis='y',color='0.7',linestyle='--')
		#for i in range(len(y)):
		#	plt.plot(x,y[i,:],self.color[i],linewidth=2)
		plt.plot(x,y[:],'b',linewidth=2)
		plt.ylabel(y_name,fontsize=self.fontsize)
		#if(m==0 and n == 1):
			#plt.legend(labels,loc='best')
		if(n == 0):
			plt.xlabel(x_name,fontsize=self.fontsize)
	def line_plot_m(self,fig,x,y,m,n,x_name,y_name,labels):
		ax = fig.add_axes([self.po[0,m,n],self.po[1,m,n],self.xpanel,self.ypanel])
		plt.sca(ax)
		plt.yscale("log")
		plt.ylim([1e-17,1e0])
		plt.xlim([np.amin(x),np.amax(x)])
		plt.grid(axis='y',color='0.7',linestyle='--')
		for i in range(len(y)):
			plt.plot(x,y[i,:],self.color2[i],linewidth=2)
		#plt.plot(x,y[:],'b',linewidth=2)
		plt.ylabel(y_name,fontsize=self.fontsize)
		plt.legend(labels,loc='best',fontsize = self.fontsize-2)
		if(n == 0):
			plt.xlabel(x_name,fontsize=self.fontsize)
	def line_plot_2(self,fig,x1,y1,x2,y2,m,n,x_name,y_name,labels):
		ax = fig.add_axes([self.po[0,m,n],self.po[1,m,n],self.xpanel,self.ypanel])
		plt.sca(ax)
		plt.yscale("log")
		plt.ylim([1e-6,1e-5])
		plt.xlim([np.amin(x1),np.amax(x1)])
		for i in range(len(y1)):
			plt.plot(x1,y1[i,:],self.color[i],linewidth=2)
		img = []
		for i in range(len(y2)):
			im,=plt.plot(x2,y2[i,:],self.marker[i],ms=6)
			img.append(im)
		plt.ylabel(y_name,fontsize=self.fontsize)
		if(m==0 and n == 0):
			ax.legend(handles=img,labels=labels)
		if(n == 0):
			plt.xlabel(x_name,fontsize=self.fontsize)
	def contour_plot(self,fig,data,m,n,x_vis,y_vis,p_info,c_info,test):
		x_min  = p_info[0]
		x_max  = p_info[1]
		y_min  = p_info[2]
		y_max  = p_info[3]
		cbmin = c_info[0]
		cbmax = c_info[1]
		if x_vis == 0:
			all = True
		else:
			all = False
		if y_vis == 0:
			alb = True
		else:
			alb = False
		ax     = fig.add_axes([self.po[0,m,n],self.po[1,m,n],self.xpanel,self.ypanel])
		plt.xlim(x_min,x_max)
		plt.ylim(y_min,y_max)
		asp    = self.figure_y*self.ypanel*(x_max-x_min)/(self.figure_x*self.xpanel*(y_max-y_min))
		if(test == 0):
			im     = plt.imshow(data, cmap=self.cmap, vmin=cbmin, vmax=cbmax, origin='lower',aspect=asp, 
                    interpolation='bicubic',extent=[x_min, x_max, y_min,y_max])
		else:
			plt.sca(ax)
			plt.plot([x_min,x_max],[y_min,y_max],color = '0.7',linewidth=1)
			plt.plot([x_min,x_max],[y_max,y_min],color = '0.7',linewidth=1)
			for pos in ['top', 'bottom', 'right', 'left']:
				ax.spines[pos].set_edgecolor('0.7')
			ax.tick_params(axis='x', colors='0.7')
			if(m!=0):
				ax.tick_params(axis='y', colors='0.7')
		#plt.xlim(x_min,x_max)
		plt.tick_params(
	        axis='x',        # changes apply to the x-axis
	        which='both',    # both major and minor ticks are affected
	        bottom='on',     # ticks along the bottom edge are off
	        top='on',        # ticks along the top edge are off
	        direction='out',# Puts ticks inside the axes, outside the axes, or both
	        labelbottom=alb) # labels along the bottom edge are off
		#plt.ylim(y_min,y_max)
		plt.tick_params(
	        axis='y',        # changes apply to the y-axis
	        which='both',    # both major and minor ticks are affected
	        left='on',       # ticks along the bottom edge are off
	        right='on',      # ticks along the top edge are off
	        direction='out',# Puts ticks inside the axes, outside the axes, or both
	        labelleft=all)   # labels along the left edge are off
		plt.minorticks_on()
		if(test == 0):
			cb = fig.add_axes([self.cpo[0,m,n],self.cpo[1,m,n],self.cxpanel,self.cypanel])
			cbar = plt.colorbar(im, cax=cb)
			cbar.ax.tick_params(labelsize=self.tlfs)
			plt.sca(ax)
	def contour_plot_no_cb(self,fig,data,m,n,x_vis,y_vis,p_info,c_info,test,test2):
		x_min  = p_info[0]
		x_max  = p_info[1]
		y_min  = p_info[2]
		y_max  = p_info[3]
		cbmin = c_info[0]
		cbmax = c_info[1]
		if x_vis == 0:
			all = True
		else:
			all = False
		if y_vis == 0:
			alb = True
		else:
			alb = False
		ax     = fig.add_axes([self.po[0,m,n],self.po[1,m,n],self.xpanel,self.ypanel])
		plt.xlim(x_min,x_max)
		plt.ylim(y_min,y_max)
		asp    = self.figure_y*self.ypanel*(x_max-x_min)/(self.figure_x*self.xpanel*(y_max-y_min))
		if(test == 0):
			im     = plt.imshow(data, cmap=self.cmap, vmin=cbmin, vmax=cbmax, origin='lower',aspect=asp, 
                    interpolation='bicubic',extent=[x_min, x_max, y_min,y_max])
			plt.minorticks_on()
		else:
			plt.sca(ax)
			plt.plot([x_min,x_max],[y_min,y_max],color = '0.7',linewidth=1)
			plt.plot([x_min,x_max],[y_max,y_min],color = '0.7',linewidth=1)
			plt.minorticks_on()
			for pos in ['top', 'bottom', 'right', 'left']:
				ax.spines[pos].set_edgecolor('0.7')
			ax.tick_params(axis="x", which='both', colors="0.7")
			ax.tick_params(axis="y", which='both', colors="0.7")
			ax.set_xticks([0, 10],color='k')
			ax.set_xticklabels([0, 10],color='k')
			ax.set_yticks([-10,0, 10],color='k')
			ax.set_yticklabels([-10,0, 10],color='k')
		#plt.xlim(x_min,x_max)
		plt.tick_params(
	        axis='x',        # changes apply to the x-axis
	        which='both',    # both major and minor ticks are affected
	        bottom='on',     # ticks along the bottom edge are off
	        top='on',        # ticks along the top edge are off
	        direction='out',# Puts ticks inside the axes, outside the axes, or both
	        labelbottom=alb) # labels along the bottom edge are off
		#plt.ylim(y_min,y_max)
		plt.tick_params(
	        axis='y',        # changes apply to the y-axis
	        which='both',    # both major and minor ticks are affected
	        left='on',       # ticks along the bottom edge are off
	        right='on',      # ticks along the top edge are off
	        direction='out',# Puts ticks inside the axes, outside the axes, or both
	        labelleft=all)   # labels along the left edge are off
		if(test2 == 3):
			im.cmap.set_under('w')
			im.cmap.set_over('k')
			cb = fig.add_axes([self.cpo[0,m,n]-0.01,self.cpo[1,m,n],self.cxpanel,self.cypanel])
			cbar = plt.colorbar(im, cax=cb,extend='both', format='%.1f')
			cbar.ax.tick_params(labelsize=12)
			plt.sca(ax)
	def set_xlabel(self,fig,xtitle):
		temp  = self.L/self.figure_x
		temp1 = (self.D-2*self.tlfs/72)/self.figure_y
		temp2 = self.U/self.figure_y
		#x     = self.po[0,1,0]+0.5*self.xpanel
		x     = 0.5
		y     = 0
		fig.text(x, y, xtitle, horizontalalignment='center', verticalalignment='bottom',fontsize=self.fontsize,)
	def set_ylabel(self,fig,ytitle):
		temp  = self.fontsize/72/self.figure_x
		temp1 = self.D/self.figure_y
		temp2 = self.U/self.figure_y
		x     = temp/2
		y     = temp1+(1-temp1-temp2)/2
		fig.text(x, y, ytitle, horizontalalignment='center', verticalalignment='center',fontsize=self.fontsize, rotation=90)
	def set_ylabel2(self,fig,n,name):
		temp  = 3*self.fontsize/72/self.figure_x
		temp1 = self.D/self.figure_y
		temp2 = self.U/self.figure_y
		x     = temp/2
		y     = self.po[1,0,n]+0.5*self.ypanel
		fig.text(x, y, name, horizontalalignment='center', verticalalignment='center',fontsize=12, rotation=90)
	def set_title(self,fig,m,name):
		temp  = self.L/self.figure_x
		temp1 = self.D/self.figure_y
		temp2 = self.U/self.figure_y
		x     = self.po[0,m,-1]+0.5*self.xpanel
		y     = self.po[1,m,-1]+self.ypanel+0.01
		fig.text(x, y, name, horizontalalignment='center', verticalalignment='bottom',fontsize=12)
	def set_text(self,fig,x,y,string):
		fig.text(x, y, string,fontsize=16)

