import cv2
import numpy as np
import scipy.interpolate as si
import os
from scipy.linalg import block_diag
from skimage.segmentation import active_contour
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

class lane_detector :

	def __init__ (self, image,skipSize=250, partition_size=[250,20,25,30,30,30,30,30,35], \
					thr=[5]*10,maxLineGap=1,all_lines=False,midline=False,blur=None, group_size=8, warehouse=False) : 
		self.image = image
		self.height, self.width = image.shape[:2]
		self.skipSize = skipSize
		self.partition_size = partition_size
		self.n = len(self.partition_size)
		self.threshold = thr
		self.maxLineGap = maxLineGap
		self.horizon = None
		self.lines = [] 			# all lines found
		self.best_left = [] 		# most right, left line; in reversed order in regards to partitions
		self.best_right = [] 		# most left, right line; in reversed order in regards to partitions
		self.group_size = group_size		# for grouping rows in finding the horizon
		self.k = None				#
		self.xm = []				# midline x coordinates
		self.ym = []				# midline y coordinates
		self.Qx = []				# control points x coordinates
		self.Qy = []				# control points y coordinates
		self.midline = []			# midline points
		self.all = all_lines		# draw all lines
		self.draw_midline = midline	
		self.blur = blur
		self.warehouse = warehouse

	def detect(self) :

		self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
		if self.blur is not None :
			self.gray = cv2.GaussianBlur(self.gray,self.blur , 0,0)
		self.edges = cv2.Canny(self.gray,10,150)
		self.edges = self.select_region(self.edges)
		self.partition = self.partition_image(self.edges, self.partition_size)

		self.find_horizon()
		self.determine_best_lines()
		self.merge_partitions()
		self.determine_midline_and_k();
		self.determine_control_points()
		if (len(self.midline) != 0) :
			self.b_snake()
		
	def partition_image(self, img, partition_size) :
		if (np.sum(partition_size) != self.height) :
			print "partition_size: ", np.sum(partition_size), " not equal height: ", self.height
			exit(1)
		y = 0
		ret = []
		for size in partition_size : 
			ret.append(img[y:y+size,0:self.width])
			y += size
		return ret

	def select_region(self, image):
		rows, cols = image.shape[:2]
		if self.warehouse :
			u,d,ul,ur,dl,dr = 0.05, 0.07, 0.3, 0.3, 0.1, 0.1
		else :
			u,d,ul,ur,dl,dr = 0.5, 0.05, 0.46, 0.42, 0.15, 0.1
		bottom_left  = [cols*dl, rows*(1.-d)]
		bottom_right = [cols*(1-dr), rows*(1.-d)]
		top_left     = [cols*ul, rows*u]
		top_right    = [cols*(1-ur), rows*u] 
		vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
		mask = np.zeros_like(image)
		cv2.fillPoly(mask, vertices, 255)
		return cv2.bitwise_and(image, mask)

	def find_horizon(self) :
		horizons = np.zeros((self.height*2,1))
		origin_height = 0 

		flag = True
		for size, im, thr in zip(self.partition_size, self.partition, self.threshold) : 
			if (self.skipSize is not None and flag) :
				origin_height += size; self.lines.append(None)
				flag = False
				continue

			# Hough transform
			self.lines.append(cv2.HoughLinesP(im,1,np.pi/180, thr, 10, self.maxLineGap))
			section = self.lines[-1]

			if section is not None : # it can happen that there are no lines in a section
				for i in range(len(section)) :
					for j in range(i+1,len(section)) :
						x,y = self.intersection(section[i][0],section[j][0])
						if (x == 100000) :
							continue
						vpoint = origin_height + y
						if vpoint > -self.height and vpoint < self.height :
							horizons[self.height+vpoint] += 1
				
				# optional -- draw all lines 
				if (self.all) :
					for line in self.lines[-1] :
						for x1,y1,x2,y2 in line :
							cv2.line(im,(x1,y1),(x2,y2),(255,255,255),5)
			origin_height += size

		best_group = np.argmax(np.sum(horizons.reshape(-1, self.group_size), axis=1))
		self.horizon = best_group*self.group_size - self.height #+ int(self.group_size/2)

	def determine_best_lines(self) :
		origin_height = self.height
		for section,size,im in zip(reversed(self.lines),reversed(self.partition_size),reversed(self.partition)):
			origin_height -= size
			left = -1
			right = -1

			if section is None :
				self.best_right.append(right)
				self.best_left.append(left)
				continue
			
			for i in range(len(section)) :
				for j in range(i+1,len(section)) :
					x,y = self.intersection(section[i][0],section[j][0])
					if x == 100000 :
						continue
					vp = origin_height + y
					if self.horizon <= vp and vp <= self.horizon+self.group_size :
						left, right = self.update(section,left,right,i,j)
			self.best_right.append(right)
			self.best_left.append(left)

	def merge_partitions(self) :
		t=0
		for size, im in zip(self.partition_size, self.partition):
			self.gray[t:t+size, 0:self.width] = im
			t += size

	def mirror(self, line) :
		x1,y1,x2,y2 = line
		if self.isRight(line) :
			d = self.k*(y1-self.horizon)
			x1 = x1 - d
			d = self.k*(y2-self.horizon)
			x2 = x2-d
		else :
			d = self.k*(y1-self.horizon)
			x1 = x1 + d
			d = self.k*(y2-self.horizon)
			x2 = x2 + d
		return [x1,y1,x2,y2]

	def find_midpoint(self,left_line, right_line, origin_height, size=0) :

		if origin_height < self.horizon :
			origin_height = self.horizon
		y = size
		if left_line[0] == left_line[2] :
			xl = left_line[0]
		else :
			k,b = self.line_parametersP(left_line)
			if k == 0 :
				xl = 0
			else :
				xl = (y-b)/k
		if right_line[0] == right_line[2] :
			xr = right_line[0]
		else :
			k,b = self.line_parametersP(right_line)
			if k == 0 :
				xr = self.width
			else :
				xr = (y-b)/k
			

		if size != 0 and self.nonempty_sections == 0:
			self.k = (xr-xl)/(y+origin_height-self.horizon)
			return int(0.5*(xr-xl) + xl), y + origin_height
		
		self.nonempty_sections += 1
		return int(0.5*(xr-xl) + xl), y + origin_height

	def determine_midline_and_k(self) : 
		p = None
		origin_height = self.height
		flag = True
		self.nonempty_sections = 0
		for bl,br,i,section in zip(self.best_left, self.best_right, reversed(range(self.n)),reversed(self.lines)) :
			origin_height -= self.partition_size[i]
			
			if br == -1 or bl == -1 :
				self.xm.append(None)
				self.ym.append(None)
				continue
			left_line = section[bl][0]
			right_line = section[br][0]
			if self.nonempty_sections == 0 :
				x,y = self.find_midpoint(left_line,right_line,origin_height, size=self.partition_size[i])
				self.xm.append(x);self.ym.append(y);

			x,y = self.find_midpoint(left_line,right_line,origin_height)
			self.xm.append(x);self.ym.append(y);
		
		# find sections without lines
		origin_height = self.height-self.partition_size[self.n-1]
		flag = 0
		last_found = 0
		for j, size in zip(range(1,self.n), reversed(self.partition_size[:self.n-1])) :
			if j+flag < len(self.xm) and self.xm[j+flag] is not None and self.xm[j+flag-1] is None:
				i = j-1
				current = origin_height
				left_line = self.lines[self.n-i-2][self.best_left[i+1]][0]
				right_line = self.lines[self.n-i-2][self.best_right[i+1]][0] 
				for k,size in zip(reversed(range(last_found,i+1)),reversed(self.partition_size[self.n-i-1:self.n-last_found])) :
					x,y = self.find_midpoint(left_line,right_line,current, size=size)
					self.xm[k+flag] = x; self.ym[k+flag] = y
					current += size

				if flag == 0: 
					flag = 1
				last_found = j
			elif j+flag < len(self.xm) and self.xm[j+flag] is not None :
				last_found = j+1	
			origin_height -= size

		while len(self.xm) and self.xm[-1] is None :
			self.xm.pop()
			self.ym.pop()
		while len(self.xm) and self.ym[-1] == self.ym[-2] and self.ym[-1] == self.horizon :
			self.xm.pop()
			self.ym.pop()

	def determine_control_points(self) :

		if len(self.xm) == 0 :
			return
		elif len(self.xm) > 3 :
			x1, y1 = self.xm[-2], self.ym[-2]
			x2, y2 = self.xm[-3], self.ym[-3]

			vp1x, vp1y = self.intersection([self.xm[-1], self.ym[-1],x1,y1], [0,self.horizon, self.width, self.horizon], True)
			vp2x, vp2y = self.intersection([x1,y1,x2,y2], [0,self.horizon, self.width, self.horizon], True)
			vp3x,vp3y = self.intersection([self.xm[-4], self.ym[-4],x2,y2], [0,self.horizon, self.width, self.horizon], True)

			b1 = vp3x-vp2x
			b2 = vp2x-vp1x
			if b2 == 0 :
				px = x2; py = y2
			elif b1 == 0 :
				px = x1; py = y1
			else :
				px = int((x1+x2)/2.); py = int((y1+y2)/2.)

			qx = 1.5*px - 0.25*(self.xm[0] + self.xm[-1])
			qy = 1.5*py - 0.25*(self.ym[0] + self.ym[-1])

			self.Qx = np.array([self.xm[-1],self.xm[-1],self.xm[-1], qx,self.xm[0],self.xm[0],self.xm[0]]).reshape((7,1))
			self.Qy = np.array([self.ym[-1],self.ym[-1],self.ym[-1], qy,self.ym[0],self.ym[0],self.ym[0]]).reshape((7,1))
			self.midline = np.rint(self.bspline(np.column_stack((self.Qx,self.Qy)))).astype(int)
		else :
			px = self.xm[-2]; py = self.ym[-2]

			qx = 1.5*px - 0.25*(self.xm[0] + self.xm[-1])
			qy = 1.5*py - 0.25*(self.ym[0] + self.ym[-1])

			self.Qx = np.array([self.xm[-1],self.xm[-1],self.xm[-1], qx,self.xm[0],self.xm[0],self.xm[0]]).reshape((7,1))
			self.Qy = np.array([self.ym[-1],self.ym[-1],self.ym[-1], qy,self.ym[0],self.ym[0],self.ym[0]]).reshape((7,1))
			self.midline = np.rint(self.bspline(np.column_stack((self.Qx,self.Qy)))).astype(int)

	def b_snake(self) :
		p = np.asarray(self.edges).astype('int8')
		dy, dx = np.gradient(p)
		gamma = 0.2e-3
		s_max = float(len(self.midline))
		n = 2
		m = 5
		C = np.array([[-1/6.,0.5,-0.5,1/6.],[0.5,-1,0.5,0],[-0.5,0,0.5,0],[1/6.,2/3.,1/6.,0]])
		done = False
		kk = 0
		while(not done) :
			done = True
			kk += 1
			ek = 0
			Qx = np.zeros(self.Qx.shape)
			Qy = np.zeros(self.Qx.shape)
			for i in range (-1, 3) :
				ii = i+1
				mi = np.zeros((m,4))
				eix = np.zeros((m,1)); eiy = np.zeros((m,1))
				for j in range(m) :
					init = ii*40 + 8*j
					s = (init+4)/s_max
					mi[j] = np.array([s**3,s**2,s,1])
					for l in range(init,init+8) :
						x,y = self.midline[l]
						d = int(self.k*(y-self.horizon)/2.)
						if x+d >= self.width :
							d = self.width-x-2
						if x-d < 0 :
							d = x
						if y < 0 :
							y = 0 
						if y >= self.height :
							y = self.height-1
						eix[j] += dx[y][x+d] + dx[y][x-d]
						eiy[j] += dy[y][x+d] + dy[y][x-d]
						ek += dx[y][x-d] - dx[y][x+d]
				mi = np.dot(mi,C)
				mi = np.dot(np.linalg.inv(np.dot(mi.T,mi)), mi.T) # 4x5
				dqx = np.dot(mi, eix) #4x1
				dqy = np.dot(mi,eiy)
				Qx[ii:ii+4] = self.Qx[ii:ii+4] + dqx*gamma
				Qy[ii:ii+4] = self.Qy[ii:ii+4] + dqy*gamma
			dk = ek*0.5e-2
			self.midline = np.rint(self.bspline(np.column_stack((Qx,Qy)))).astype(int)
			if (dk > 0.01 or np.sum(np.abs(Qx-self.Qx)+np.abs(Qy-self.Qy)) > 17):
				done = False
			else :
				self.k += dk
				self.Qx = Qx
				self.Qy = Qy
		
	def snakes(self,alpha=0.1, beta=10, gamma=0.001,w_edge=0) :
		init1 = [] ; init2 = []
		for i in range(len(self.midline)) :
 			d = np.array([int(self.k*(self.midline[i][1]-self.horizon)/2),0])
 			init1.append(self.midline[i]+d)
 			init2.append(self.midline[i]-d)
 		init1 = np.array(init1); init2 = np.array(init2)
		self.line1 = active_contour(self.image, init1, alpha=1, beta=100, gamma=0.0001,w_edge=0).astype(int)
		self.line2 = active_contour(self.image, init2, alpha=1, beta=100, gamma=0.0001,w_edge=0).astype(int)



	def line_parameters(self, line) :
		rho, th = line
		a = np.cos(th)
		b = np.sin(th)
		k = -a/b
		b = rho/b
		return (k,b)

	def line_parametersP(self, line) :
		x1 = line[0]
		y1 = line[1]
		x2 = line[2]
		y2 = line[3]
		k = float(y2-y1)/float(x2-x1)
		b = float(y1) - float(k)*float(x1) 
		return (k,b)

	def intersection(self, l1, l2, horizontal=False) :

		if l1[0] - l1[2] == 0 :
			if l2[0] - l2[2] == 0 :
				return (100000,100000)
			else :
				k, b = self.line_parametersP(l2)
				if k == 0 :
					return (100000,100000)
				y = k*l1[0]+b
				if y > l2[1] and y > l2[3] :
					return (100000,100000)
				return (l1[0], int(y))
		elif l2[0] - l2[2] == 0 :
			k, b = self.line_parametersP(l1)
			if k == 0 :
				return (100000,100000)
			y = k*l2[0]+b
			if y > l1[1] and y > l1[3] :
				return (100000,100000)
			return (l2[0], int(y))
		else :
			k1, b1 = self.line_parametersP(l1)
			k2, b2 = self.line_parametersP(l2)
			if (k1 == k2) :
				return (100000,100000)
			if (k1 == 0 or k2 == 0) and not horizontal:
				return (100000,100000) 
			x = float(b2-b1)/float(k1-k2)
			y = k1*x+b1

			if (x < -300 or x > self.width+300 or (y > l1[1] and y > l1[3]) or (y > l2[1] and y > l2[3])) :
				return (100000,100000)
			return (int(x),int(y))

	def isRight(self, line) :
		x1,y1,x2,y2 = line
		if (x1+x2)/2 < self.width*0.46 :
			return False
		else: return True

	def update(self, section, left, right, i, j) :
		if self.isRight(section[i][0]) and not self.isRight(section[j][0]) :
			if right == -1 :
				right = i; left = j
			elif self.better(section[i][0], section[right][0], True) \
					and self.better(section[j][0], section[left][0], False) :
				right = i; left = j
		elif self.isRight(section[j][0]) and not self.isRight(section[i][0]) :
			if right == -1 :
				right = j; left = i
			elif self.better(section[j][0], section[right][0], True) \
					and self.better(section[i][0], section[left][0], False) :
				right = j; left = i
		return left, right

	def better(self, line1, line2, right) :
		t1 = [line1[0],line1[2]]
		t2 = [line2[0],line2[2]]
		if right :
			if np.min(t1) < np.min(t2) :
				return True
			else :
				return False
		else :
			if np.max(t1) > np.max(t2) :
				return True
			else :
				return False

	def bspline(self, cv, n=200, degree=3, periodic=False):
		cv = np.asarray(cv)
		count = len(cv)

		if periodic:
			factor, fraction = divmod(count+degree+1, count)
			cv = np.concatenate((cv,) * factor + (cv[:fraction],))
			count = len(cv)
			degree = np.clip(degree,1,degree)

		# If opened, prevent degree from exceeding count-1
		else:
			degree = np.clip(degree,1,count-1)


		# Calculate knot vector
		kv = None
		if periodic:
			kv = np.arange(0-degree,count+degree+degree-1,dtype='int')
		else:
			kv = np.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')

		# Calculate query range
		u = np.linspace(periodic,(count-degree),n)


		# Calculate result
		arange = np.arange(len(u))
		points = np.zeros((len(u),cv.shape[1]))
		for i in xrange(cv.shape[1]):
			points[arange,i] = si.splev(u, (kv,cv[:,i],degree))

		return points

	def draw(self) :
		if self.draw_midline :
			for i in range(len(self.xm)-1) :
				cv2.line(self.image,(self.xm[i],self.ym[i]),(self.xm[i+1],self.ym[i+1]),(255,255,255),1)
 		for i in range(len(self.midline)-1) :
 			d1 = np.array([int(self.k*(self.midline[i][1]-self.horizon)/2),0])
 			d2 = np.array([int(self.k*(self.midline[i+1][1]-self.horizon)/2),0])
 			cv2.line(self.image,tuple(self.midline[i]+d1),tuple(self.midline[i+1]+d2),(255,255,255),5)
 			cv2.line(self.image,tuple(self.midline[i]-d1),tuple(self.midline[i+1]-d2),(255,255,255),5)
 			# cv2.line(self.image,tuple(self.line1[i]),tuple(self.line1[i+1]),(255,255,255),7)
 			# cv2.line(self.image,tuple(self.line2[i]),tuple(self.line2[i+1]),(255,255,255),7)

		#plt.imshow(self.image,)
		# plt.show()

	def draw_edges(self) :

		for best in [reversed(self.best_right),reversed(self.best_left)] :
			origin_height = 0
			for k,size,section in zip(best, self.partition_size, self.lines) :
				if k == -1 :
					origin_height += size
					continue
				x1,y1,x2,y2 = section[k][0]
				cv2.line(self.gray,(x1,y1+origin_height),(x2,y2+origin_height),(255,255,255),3)
				origin_height += size
	
	def get_image(self) :
		self.draw()
		return self.image
	
	def get_gray(self) :
		self.draw_edges()
		return self.gray

def main() :

	files = [f for f in os.listdir('./test/outdoor') if f.endswith(".jpg") ]
	for f, i in zip(files,range(len(files))) :
		print i+1
		detector = lane_detector(cv2.imread("./test/outdoor/" + f),skipSize=260, partition_size=[260,40,50,60,70], \
					thr=[10]*10,maxLineGap=6, blur=None,all_lines=False,group_size=20)
		detector.detect()
		plt.imshow(detector.get_image())
		plt.savefig("test/" + str(f))

	files = [f for f in os.listdir('./test/indoor') if f.endswith(".jpg") ]
	for f, i in zip(files,range(len(files))) :
		print i+1
		detector = lane_detector(cv2.imread("./test/indoor/"+f),skipSize=None, partition_size=[80,100,100,100,100], \
					thr=[50]*10,maxLineGap=10, blur=None,all_lines=False,group_size=10,warehouse=True)
		detector.detect()
		plt.imshow(detector.get_image())
		plt.savefig("test/" + str(f))

if __name__ == "__main__":
	main()


