#!/usr/bin/python

from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import argv
from math import pi, sin, cos, atan, acos

def normalize(vector):
    vector = vector/linalg.norm(vector)
    return vector

class SpaceLims(object):
	def __init__(self, lo, hi):
		self.x = [lo[0], hi[0]]
		self.y = [lo[1], hi[1]]
		self.z = [lo[2], hi[2]]
	
	def toPoints(self):
		return ([self.x[0], self.y[0], self.z[0]], [self.x[1], self.y[1], self.z[1]])	
		
class Trajectory(object):
	def __init__(self, obj):
		self.obj = obj
		self.step = 0
		
	def bounce(self,**kwargs):
		# lims
		lims = kwargs['lims']
		self.computeDirectionMatrix(lims)
		self.obj.pos = self.obj.pos + self.obj.vel*self.obj.dm
		self.step = self.step + 1
		
	def computeDirectionMatrix(self,lims):
		# lims
		(lo,hi) = lims.toPoints()
		for i in range(0,3):
			new_pos = self.obj.pos[i] + self.obj.vel[i]
			if new_pos >= hi[i] or new_pos <= lo[i]:
				self.obj.dm[i] = -1*self.obj.dm[i]

	def moveInCircle(self,**kwargs):
		# c, r, t_start, t_step, phi_start, phi_step
		c = kwargs['c']
		r = kwargs['r']
		t_start = kwargs['t_start']
		t_step = kwargs['t_step']
		phi_start = kwargs['phi_start']
		phi_step = kwargs['phi_step']
		t = radians(t_start + self.step*t_step)
		phi = radians(phi_start + self.step*phi_step)
		x = c[0] + r*sin(phi)*cos(t)
		y = c[1] + r*sin(phi)*sin(t)
		z = c[2] + r*cos(phi)
		self.obj.vel = array([x,y,z]) - self.obj.pos
		self.obj.pos = self.obj.pos + self.obj.vel
		self.step = self.step + 1
		
	def moveInTrefoil(self, x_scale = 1., y_scale = 1., z_scale = 1, **kwargs):
		# c, r ,t_start, t_step, x_scale = 1., y_scale = 1., z_scale = 1.
		c = kwargs['c']
		r = kwargs['r']
		t_start = kwargs['t_start']
		t_step = kwargs['t_step']
		r = r/3.
		t = radians(t_start + self.step*t_step)
		x = (c[0] + r*(sin(t) + 2*sin(2*t)))*x_scale
		y = (c[1] + r*(cos(t) - 2*cos(2*t)))*y_scale
		z = (c[2] - r*sin(3*t))*z_scale
		self.obj.vel = array([x,y,z]) - self.obj.pos
		self.obj.pos = self.obj.pos + self.obj.vel
		self.step = self.step + 1
		
	def randomMove(self, **kwargs):
		# lo, hi, vlo, vhi
		self.obj.vel = [random.randint(vlo,vhi), random.randint(vlo, vhi), random.randint(vlo, vhi)]/100.
		self.bounce(lo,hi)		

class Universe(object):
	def __init__(self, lims):
		self.players = {}
		self.lims = lims
		self.collectionClasses = ['Swarm','Membrane','ElasticMembrane','RigidHollowSphere', 'ElasticHollowSphere', 'QuadricSurface', 'ElasticQuadricSurface']
		self.influenceClasses = ['Attractant','Repellent','Predator']
	
	def __iter__(self):
		for p in self.players.values():
			yield p
	
	def randomPoint(self,lims):
		return array([random.randint(lims.x[0],lims.x[1]), random.randint(lims.y[0],lims.y[1]), random.randint(lims.z[0],lims.z[1])])
	
	def randomPosVelDm(self,p,v):
		dm = array([random.choice([-1,1]), random.choice([-1,1]), random.choice([-1,1])])
		pos = self.randomPoint(p)
		vel = self.randomPoint(v)/100.
		return(pos,vel,dm)
	
	def addAtt(self, n, p, v = SpaceLims([50,50,50], [100, 100, 100])):
		for i in range(0,n):
			pvd = self.randomPosVelDm(p, v)
			o = Attractant()
			o.pos = pvd[0]
			o.vel = pvd[1]
			o.dm = pvd[2]
			o.tag = "att_%i" %i
			o.activate()
			for s in self:
				if s.__class__.__name__ in self.collectionClasses:
					s.addInfluence(o)
			self.players[o.tag] = o
	
	def addRep(self, n, p, v = SpaceLims([50,50,50], [100, 100, 100])):
			for i in range(0,n):
				pvd = self.randomPosVelDm(p,v)	
				o = Repellent()
				o.pos = pvd[0]
				o.vel = pvd[1]
				o.dm = pvd[2]
				o.tag = "rep_%i" %i
				o.activate()
				for s in self:
					if s.__class__.__name__ in self.collectionClasses:
						s.addInfluence(o)
				self.players[o.tag] = o
					
	def addPred(self, n, p):
		for i in range(0,n):
			pos = self.randomPoint(p)
			o = Predator()
			o.pos = pos
			o.tag = "pred_%i" %i
			o.activate()
			for s in self:
				if s.__class__.__name__ in self.collectionClasses:
					s.addInfluence(o)	
			self.players[o.tag] = o
	
	def addCollectionOfMovingObjects(self, como):
		self.players[como.tag] = como
		for inf in como.influences.values():
			self.addPlayer(inf)
		
	def addPlayer(self,player):
		self.players[player.tag] = player
		
	def dump(self):
		for p in self:
			p.dump()
			
	def plot(self, fig, elev, azim):
		cmaps = ['Blues','Reds','Greys','jet','hot','hsv']
		fig.clf()
		ax = fig.add_subplot(111, projection='3d')
		ax.view_init(elev = elev, azim = azim)
		ax.set_autoscale_on(False)
		ax.set_xlim3d(uni.lims.x)
		ax.set_ylim3d(uni.lims.y)
		ax.set_zlim3d(uni.lims.z)
	
		for p in uni.players.values():
			if p.__class__.__name__ in self.collectionClasses :
				print p.__class__.__name__
				c_map = p.color
				if c_map in cmaps:
					(bx,by,bz,bvel,bsize,bcol,balpha) = p.getElementsData()
					ax.scatter(bx, by, bz, c=bvel, cmap = plt.get_cmap(c_map), s = bsize)
				else:
					for e in p:
						ax.scatter(e.pos[0], e.pos[1], e.pos[2], c=e.color, s = e.size, alpha = e.alpha)
					
			if p.__class__.__name__ in self.influenceClasses:
				if p.active == True:
					ix = p.pos[0]
					iy = p.pos[1]
					iz = p.pos[2]
					ax.scatter(ix,iy,iz, c=p.color, s=p.size, alpha = p.alpha)
					
	def step(self,steps,fig,x_rev,y_rev, d):
		elev = 30
		azim = 30
		azim_incr = x_rev*360./steps
		elev_incr = y_rev*360./steps
	
		for step in range(0,steps):
			print "Step %i" % step
			for p in self:			
				p.step()
				p.dump()

			self.plot(fig, elev, azim)
			azim = azim + azim_incr
			elev = elev + elev_incr
			fname  = "frame_%04d.png" % step
			fig.savefig(fname, dpi = d)
						
		
class StaticObject(object):
	def __init__(self, pos = zeros((3))):
		self.pos = pos
		self.color = None
		self.size = None
		self.alpha = 1
		self.tag = hash(self)
	
	def __str__(self):
		s = '%s\n' % self.__class__.__name__
		for (key,value) in sorted(self.__dict__.items()):
			s = s + '\t%s %r\n' % (key,value)
		return s	
	
	def	dump(self):
		print self

class MovingObject(StaticObject):
	def __init__(self, pos = zeros((3)), vel = zeros((3)), maxVel = 1):
		self.vel = vel
		self.maxVel = maxVel
		super(MovingObject,self).__init__(pos)
		
	def limitVel(self):
		if linalg.norm(self.vel) > self.maxVel:
			self.vel = (self.vel/linalg.norm(self.vel))*self.maxVel
		
class MoWithBehaviour(MovingObject):
	def __init__(self, pos = zeros((3)), vel = zeros((3)), maxVel = 1, minSep = 10, wAtt = 0.05, wAvdRep = 1, minRepSep = 15):
		self.wAtt = wAtt
		self.minRepSep = minRepSep
		self.wAvdRep = wAvdRep
		self.minSep = minSep
		super(MoWithBehaviour,self).__init__(pos, vel, maxVel)
		
	# attractants	
	def attract(self, att):
		return (att.pos - self.pos)*self.wAtt
		
	# repellent avoidance
	def repell(self, rep):
		v5 = zeros((3))
		difference = rep.pos - self.pos
		distance = linalg.norm(difference)
		if distance < self.minRepSep:
			v5 = (v5 - difference)*self.wAvdRep
		return v5			

class MoWithTrajectory(MoWithBehaviour):
	def __init__(self, pos, vel, maxVel, dm, active, mode):
		self.dm = dm
		self.active = active
		self.mode = mode
		self.traj = Trajectory(self)
		self.traj_data = None
		super(MoWithTrajectory,self).__init__(pos, vel, maxVel)
		
	def activate(self):
		self.active = True
		
	def deactivate(self):
		self.active = False
		
	def step(self):
		if self.active == True:
			if self.mode == 'random':
				self.traj.randomMove(**self.traj_data)
			elif self.mode == 'circle':
				self.traj.moveInCircle(**self.traj_data)
			elif self.mode == 'trefoil':
				self.traj.moveInTrefoil(**self.traj_data)
			elif self.mode == 'bounce':
				self.traj.bounce(**self.traj_data)
			elif self.mode == 'wait':
				self.vel = zeros((3))
			else:
				pass				

class Attractant(MoWithTrajectory):
	def __init__(self, pos = zeros((3)), vel = zeros((3)), maxVel = 1, dm = zeros((3)), active = False, mode = 'wait'):
		super(Attractant,self).__init__(pos, vel, maxVel, dm, active, mode)
		self.size = 100
		self.color = 'r'
		self.alpha = 0.5
		
class Repellent(MoWithTrajectory):
	def __init__(self, pos = zeros((3)), vel = zeros((3)), maxVel = 1, dm = zeros((3)), active = False, mode = 'wait'):
		super(Repellent,self).__init__(pos, vel, maxVel, dm, active, mode)
		self.size = 100
		self.color = 'g'
		self.alpha = 0.5

class Predator(Repellent):
	def __init__(self, pos = zeros((3)), vel = zeros((3)), maxVel = 1, dm = zeros((3)), active = False, mode = 'wait', pred_range = 100):
		super(Predator,self).__init__(pos, vel, maxVel, dm, active, mode)
		self.target = None
		self.wAtt = 0.5
		self.maxVel = 4
		self.pred_range = pred_range
		self.size = 100
		self.color = 'y'

	# attack if in range	
	def r1(self):
		t_com = self.target.centerOfMass()
		dist = linalg.norm(t_com - self.pos)
		print dist
		if dist <= self.pred_range:
			return (t_com -  self.pos)*self.wAtt
		else:
			return zeros((3))
			
	def step(self):
		super(Predator,self).step()
		if self.mode == 'attack':
			self.vel = self.r1()
			self.limitVel()	
		self.pos = self.pos + self.vel

class CollectionOfMovingObjects(object):
	def __init__(self):
		self.tag = hash(self)
		self.elements = {}
		self.influences = {}
		self.ne = 0
		self.color = None
	
	def __iter__(self):
		for e in self.elements.values():
			yield e
		
	def dump(self):
		print self.__class__.__name__
		for (key,value) in sorted(self.__dict__.items()):
			print '\t%s %s' %(key,value)
		
		for e in self:
			e.dump()
		for i in self.influences.values():
			i.dump()	
						
	def addElement(self,e):
		self.elements[e.tag] = e
		self.ne = self.ne + 1	

	def removeElement(self,e):
		del self.elements[e.tag]
		self.ne = self.ne - 1
		
	def addInfluence(self, inf):
		self.influences[inf.tag] = inf	
		
	def removeInfluence(self, inf):
		del self.influences[inf]	
	
	def centerOfMass(self):
		com = zeros((3))
		for e in self:
			com += e.pos		
		com /= self.ne
		return com
		
	def centerOfVel(self):
		cov = zeros((3))   
		for e in self:
			cov += e.vel	
		cov /= self.ne
		return cov
		
	def getElementsData(self):
		ex, ey, ez, evel, esize, ecol, ealpha = [],[],[],[],[],[],[]
		for e in self:
			pos = e.pos
			ex.append(e.pos[0])
			ey.append(e.pos[1])
			ez.append(e.pos[2])
			vel = linalg.norm(e.vel)
			evel.append(vel)
			esize.append(e.size)
			ecol.append(e.color)
			ealpha.append(e.alpha)
		return (ex,ey,ez,evel,esize,ecol,ealpha)
	
	def step(self):
		raise NotImplementedError("method needs to be defined by sub-class")		

class Swarm(CollectionOfMovingObjects):
	def __init__(self, wMass= 0.03, wSep = 1, wAlign = 0.1):
		super(Swarm, self).__init__()
		self.wMass = wMass
		self.wSep = wSep
		self.wAlign = wAlign
	
	# move together	
	def r1(self, e, com):
		return (com - e.pos)*self.wMass
	
	# avoid collisions	
	def r2(self,e):
		v2 = zeros((3))
		for ei in self:
			if ei.tag != e.tag:
				difference = ei.pos - e.pos
				distance = linalg.norm(difference)
				if distance < e.minSep:
					v2 = v2 - normalize(difference)/distance
		return v2*self.wSep	

	# sync speed
	def r3(self, e, cov):
		return (cov - e.vel)*self.wAlign
	
	def step(self):
		com = self.centerOfMass()
		cov = self.centerOfVel()
		
		for e in self:
			v1 = self.r1(e, com)
			v2 = self.r2(e)
			v3 = self.r3(e,cov)
			
			v4 = zeros((3))
			v5 = zeros((3))
			
			for inf in self.influences.values():
				if inf.active == True:
					if inf.__class__.__name__ == 'Attractant':
						v4 = v4 + e.attract(inf)
					elif inf.__class__.__name__ == 'Repellent' or inf.__class__.__name__ == 'Predator':
						v5 = v5 + e.repell(inf)	
						
			e.vel = v1 + v2 + v3 + v4 + v5
			e.limitVel()
			e.pos = e.pos + e.vel

class GeomPrimitive(CollectionOfMovingObjects):
	def __init__(self, obj_limit):
		super(GeomPrimitive, self).__init__()
		self.deltas = {}
		self.object_limit = obj_limit
		
	def computeDeltas(self):
		com = self.centerOfMass()
		for e in self:
			self.deltas[e.tag] = e.pos - com
			
	def populate(self,ne):
		for i in range(0,ne):
			e = MoWithBehaviour()
			e.size = 100
			e.color = 'g'
			e.alpha = 0.25
			self.addElement(e)			
		
class ElasticGeomPrimitive(GeomPrimitive):
	def computeTransitionMatrix(self):
		tm = {}
		com = super(ElasticGeomPrimitive,self).centerOfMass()
		for e in self:
			tm[e.tag] = com + self.deltas[e.tag]
		return tm				
			
	def preserveShape(self,e,tm):
		return (tm[e.tag] - e.pos)*self.rigidity
	
	def attract(self, e, att):
		v = zeros((3))
		difference = att.pos - e.pos
		distance = linalg.norm(difference)
		if distance < self.minAttDist:
			v = (v + difference)*self.wAtt
		return v
		
	def step(self):
		for e in self:
			v1 = zeros((3))
			v2 = zeros((3))
			v3 = zeros((3))
			tm = self.computeTransitionMatrix()
			v1 = self.preserveShape(e,tm)
			for inf in self.influences.values():
				if inf.active == True:
					if inf.__class__.__name__ == 'Attractant':
						v2 = v2 + self.attract(e,inf)
					elif inf.__class__.__name__ == 'Repellent' or inf.__class__.__name__ == 'Predator':
						v3 = v3 + e.repell(inf)
						
			e.vel = v1 + v2 + v3
			e.limitVel()
			e.pos = e.pos + e.vel			

class QuadricSurface(GeomPrimitive):
	def __init__(self, xs, ys, surf_type, a, b, c):
		self.xs = xs.tolist()
		self.ys = ys.tolist()
		self.a = a
		self.b = b
		self.c = c
		self.surf_type = surf_type
		self.added_elements = 0
		super(QuadricSurface,self).__init__(len(self.xs)*len(self.ys))
		
	def addElement(self,e):
		r = int(self.added_elements/len(self.xs))
		c = self.added_elements % len(self.xs)
		self.added_elements = self.added_elements + 1
		x = self.xs[c]
		y = self.ys[r]
		z = self.calcZ(x,y)
		if z != None:
			e.pos = array([x,y,z])
			super(QuadricSurface,self).addElement(e)
			self.computeDeltas()
				
	def calcZ(self,x,y):
		if self.surf_type == 'paraboloid':
			return self.a*pow(x,2) + self.b*pow(y,2)
		
		if self.surf_type == 'elipsoid':
			sq = 1 - pow(x,2)/pow(self.a,2) - pow(y,2)/pow(self.b,2)
			if sq >= 0:
				return self.c*sqrt(sq)
			else:
				return None
		
		if self.surf_type == 'double_cone':
			sq = self.a*pow(x,2) + self.b*pow(y,2)
			if sq >= 0:
				return sqrt(sq)
			else:
				return None
				
		if self.surf_type == 'hyperboloid_1':
			sq = pow(x,2)/pow(a,2) + pow(y,2)/pow(b,2) - 1
			if sq >= 0:
				return self.c*sqrt(sq)
			else:
				return None
				
		if self.surf_type == 'hyperboloid_2':
			sq = pow(x,2)/pow(a,2) + pow(y,2)/pow(b,2) + 1
			if sq >= 0:
				return self.c*sqrt(sq)
			else:
				return None											

class ElasticQuadricSurface(QuadricSurface, ElasticGeomPrimitive):
	def __init__(self, xs, ys, surf_type, a, b, c, rigidity = 0.5, minAttDist = 15, wAtt = 1):
		super(ElasticQuadricSurface, self).__init__(xs, ys, surf_type, a, b, c)
		self.minAttDist = minAttDist
		self.wAtt = wAtt
		self.rigidity = rigidity

class Sphere(GeomPrimitive):
	def __init__(self, center, radius, obj_limit):
		super(Sphere,self).__init__(obj_limit)
		self.center = center
		self.radius = radius

class HollowSphere(Sphere):
	def __init__(self, center, radius, ps = 16, ms = 16):
		self.ps = range(1,ps/2) + range(ps/2+1, ps)
		self.ms = ms
		self.phi_step = 2*pi/(len(self.ps) + 2)
		self.t_step = 2*pi/self.ms
		super(HollowSphere,self).__init__(center,radius,self.ms*len(self.ps))
		
	def addElement(self,e):
		par_pos = int(self.ne/self.ms)
		par_pos = self.ps[par_pos]
		mer_pos = self.ne % self.ms
		t = mer_pos*self.t_step
		phi = par_pos*self.phi_step
		x = self.center[0] + self.radius*sin(phi)*cos(t)
		y = self.center[1] + self.radius*sin(phi)*sin(t)
		z = self.center[2] + self.radius*cos(phi)
		e.pos = array([x,y,z])
		super(HollowSphere,self).addElement(e)
		self.computeDeltas()
		
class ElasticHollowSphere(HollowSphere, ElasticGeomPrimitive):
	def __init__(self, 	center, radius, ps = 16, ms = 16, rigidity = 0.5, minAttDist = 15, wAtt = 1):
		super(ElasticHollowSphere, self).__init__(center, radius, ps, ms)
		self.minAttDist = minAttDist
		self.wAtt = wAtt
		self.rigidity = rigidity

class RigidSolidSphere(Sphere):
	def __init__(self, center, radius, n_elements):
		super(RigidSolidSphere, self).__init__(center, radius)
		self.n_elements = n_elements
		
class ElasticSolidSphere(RigidSolidSphere):
	def __init__(self, center, radius, n_elements):
		super(ElasticSolidSphere, self).__init__(center, radius, n_elements)

class Membrane(GeomPrimitive):
	def __init__(self, nrow, ncol, p1, p2,orientation):
		self.nrow = nrow
		self.ncol = ncol
		self.p1 = p1
		self.p2 = p2
		self.orientation = orientation	
		
		self.row_spacing = (self.p2 - self.p1)/float(self.nrow)
		self.col_spacing = (self.p2 - self.p1)/float(self.ncol)
		self.rowDelta = None
		self.colDelta = None
		super(Membrane,self).__init__(self.nrow*self.ncol)
		
		self.setDeltas()
		
	def setDeltas(self):
		rx = self.row_spacing[0]
		ry = self.row_spacing[1]
		rz = self.row_spacing[2]
		
		cx = self.col_spacing[0]
		cy = self.col_spacing[1]
		cz = self.col_spacing[2]
		
		if self.orientation == 'xy':
			self.rowDelta = array([0, ry, rz])
			self.colDelta = array([cx ,0 ,cz])
		if self.orientation == 'yx':
			self.rowDelta = array([rx, 0, rz])
			self.colDelta = array([0 ,cy, cz])	
		if self.orientation == 'xz':
			self.rowDelta = array([0, ry, rz])
			self.colDelta = array([cx, cy, 0])
		if self.orientation == 'zx':
			self.rowDelta = array([rx, ry, 0])
			self.colDelta = array([0, cy, cz])
		if self.orientation == 'yz':
			self.rowDelta = array([rx, 0, rz])
			self.colDelta = array([cx, cy, 0])
		if self.orientation == 'zy':
			self.rowDelta = array([rx, ry, 0])
			self.colDelta = array([cx, 0, cz])
		
	def addElement(self, e):
		row = int(self.ne/self.ncol)
		col = self.ne % self.ncol
		delta = row*self.rowDelta + col*self.colDelta
		e.pos = self.p1 + delta
		super(Membrane,self).addElement(e)
		self.computeDeltas()
			
class ElasticMembrane(Membrane, ElasticGeomPrimitive):
	def __init__(self, nrow, ncol, p1, p2, orientation, rigidity = 0.5, minAttDist = 15, wAtt = 1):
		super(ElasticMembrane, self).__init__(nrow, ncol,p1, p2, orientation)
		self.minAttDist = minAttDist
		self.wAtt = wAtt
		self.rigidity = rigidity
			
#########################################################################    		
#########################################################################

lims = SpaceLims([-100,-100,-100], [100, 100, 100])
fig = plt.figure()
uni = Universe(lims)
sp = ElasticHollowSphere([50,50,50],30,20,20)
sp.populate(sp.object_limit)

# xs, ys, surf_type, a, b, c
eqs = ElasticQuadricSurface(linspace(-50,50,15), linspace(-50,50,15), 'double_cone',1, 1, 1)
eqs.populate(eqs.object_limit)
uni.addCollectionOfMovingObjects(eqs)
uni.addCollectionOfMovingObjects(sp)	

uni.addRep(10, uni.lims, SpaceLims([100,100,100], [150, 150, 150]) )
for i in range(10):
	t = 'rep_%i' %i
	uni.players[t].traj_data = {'lims':uni.lims}
	uni.players[t].mode = 'bounce'

uni.step(1,fig,0,0, 100)
