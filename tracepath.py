#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt,\
                  square, linspace, zeros, array,\
                  concatenate, delete, row_stack, column_stack

from numpy.random import random, normal, randint, shuffle
from scipy import interpolate
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from scipy.spatial import cKDTree
import gtk, gobject

np.random.seed(2)

PI = pi
TWOPI = pi*2.
PIHALF = pi*0.5

SIZE = 5000
ONE = 1./SIZE

BACK = 1.

X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

DIST_NEAR_INDICES = np.inf
NUM_NEAR_INDICES = 30
SHIFT_INDICES = 5

W = 0.9
PIX_BETWEEN = 2

START_X = (1.-W)*0.5
START_Y = (1.-W)*0.5
#START_X = 0.5
#START_Y = 0.5

NUMMAX = int(2*SIZE)
NUM_LINES = int(SIZE*W/PIX_BETWEEN)
#NUM_LINES = 150
H = W/NUM_LINES

FILENAME = 'aa'

TURTLE_ANGLE_NOISE = pi*0.05
INIT_TURTLE_ANGLE_NOISE = 0.0

def myrandom(size):

  #res = normal(size=size)
  res = 1.-2.*random(size=size)

  return res


def turtle(sthe,sx,sy,steps):

  XY = zeros((steps,2),'float')
  THE = zeros(steps,'float')

  XY[0,0] = sx
  XY[0,1] = sy
  THE[0] = sthe
  the = sthe

  noise = myrandom(size=steps)*INIT_TURTLE_ANGLE_NOISE
  for k in xrange(1,steps):

    x = XY[k-1,0] + cos(the)*ONE
    y = XY[k-1,1] + sin(the)*ONE
    XY[k,0] = x
    XY[k,1] = y
    THE[k] = the
    the += noise[k]

    #if x>X_MAX or x<X_MIN or y>Y_MAX or y<Y_MIN:
      #XY = XY[:k,:]
      #THE = THE[:k]
      #break

  return THE, XY

class Render(object):

  def __init__(self,n):

    self.n = n
    self.__init_cairo()

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.n,self.n)
    ctx = cairo.Context(sur)
    ctx.scale(self.n,self.n)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx

  def line(self,xy):


    self.ctx.move_to(xy[0,0],xy[0,1])
    for (x,y) in xy[1:]:
      self.ctx.line_to(x,y)
    self.ctx.stroke()

  def circle(self,xy,r):

    self.ctx.arc(xy[0],xy[1],r,0,TWOPI)
    self.ctx.stroke()


class Path(object):

  def __init__(self,xy):

    self.xy = xy
    self.tree = cKDTree(xy)

  def trace(self,r,the):

    all_near_inds = self.tree.query_ball_point(self.xy,r)

    near_last = []
    circles = []
    for k,inds in enumerate(all_near_inds):

      ii = set(inds)
      isect = ii.intersection(near_last)

      if len(isect)<1:
        near_last = inds
        circles.append((k,inds))

    alphas = []
    for k,inds in circles:

      inds_s = array(sorted(inds),'int')
      xy_diff_sum = self.xy[inds_s[-1],:]-self.xy[inds_s[0],:]

      #xy_diff = self.xy[inds_s[1:],:]-self.xy[inds_s[:-1],:]
      #xy_diff_sum = np.sum(xy_diff,axis=0)

      alpha = arctan2(xy_diff_sum[1],xy_diff_sum[0])
      alphas.append(alpha)

    #noise = myrandom(len(alphas))*0.5
    alphas = array(alphas) + the

    dx = cos(alphas)
    dy = sin(alphas)

    xy_circles = row_stack([self.xy[k,:] for k,_ in circles])

    xy_new = zeros(xy_circles.shape,'float')
    xy_new[:,:] = xy_circles[:,:]

    xy_new[:,0] += dx*r
    xy_new[:,1] += dy*r

    return xy_circles, xy_new


def main():

  render = Render(SIZE)

  render.ctx.set_source_rgb(1,0,0)
  render.ctx.set_line_width(ONE)

  the,xy = turtle(0.5*PI,START_X,START_Y,NUMMAX)
  render.line(xy)

  for i in xrange(NUM_LINES):

    if not i%10:
      print i, NUM_LINES

    path = Path(xy)
    circles,xy = path.trace(PIX_BETWEEN*ONE,-PIHALF)

    tck,u = interpolate.splprep([xy[:,0],xy[:,1]],s=0)
    unew = np.linspace(0,1,NUMMAX)
    out = interpolate.splev(unew,tck)

    xy = column_stack(out)

    render.ctx.set_source_rgb(0,0,0)
    render.line(xy)

    alpha_noise = myrandom(len(xy))*pi
    xy_noise = column_stack([cos(alpha_noise),sin(alpha_noise)])*ONE*0.5
    xy += xy_noise

    if not i%100:

      fn = '{:s}_{:05d}.png'.format(FILENAME,i)
      print fn
      render.sur.write_to_png(fn)

  render.sur.write_to_png('{:s}_final.png'.format(FILENAME))


if __name__ == '__main__':
  if False:
    import pstats
    import cProfile
    OUT = 'profile'
    pfilename = '{:s}.profile'.format(OUT)
    cProfile.run('main()',pfilename)
    p = pstats.Stats(pfilename)
    p.strip_dirs().sort_stats('cumulative').print_stats()
  else:
    main()
