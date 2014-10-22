#!/usr/bin/python
# -*- coding: utf-8 -*-

from itertools import count
from time import time as time

import Image
import numpy as np
import cairocffi as cairo
from scipy import interpolate
from scipy.spatial import cKDTree

## fun parameters
SIZE = 1000 # half the size, in pixels
W = 0.9 # border
PIX_BETWEEN = 10 # spatial frequency of lines
BACK = [np.random.random()/2.0, 1, np.random.random(), 1] # background color, RGBA
FRONT = [0, 0, 0, 0.5] # line color, RGBA
NOISE = 0.2 # deviation of straight lines; try 1.1
DXR, DYR = 1., 1. # does weird swirly things as time goes on

ONE = 1./SIZE
NUMMAX = int(2*SIZE)

X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

START_X = (1.-W)*0.5
STOP_X = 1.-START_X
START_Y = (1.-W)*0.5
STOP_Y = 1.-START_Y

PI = np.pi
TWOPI = np.pi*2.
PIHALF = np.pi*0.5

COLOR_PATH = './img/slopes.png' # currently unused

class Render(object):
    def __init__(self,n):
        self.n = n
        self.__init_cairo()

    def __init_cairo(self):
        sur = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.n, self.n)
        ctx = cairo.Context(sur)
        ctx.scale(self.n, self.n)
        self.sur = sur
        self.ctx = ctx
        self.clear_canvas()
        self.__get_colors(COLOR_PATH)
        self.n_colors = len(self.colors)

    def clear_canvas(self):
        self.ctx.set_source_rgba(*BACK)
        self.ctx.rectangle(0, 0, 1, 1)
        self.ctx.fill()

    def __get_colors(self, f):
        scale = 1./255.
        im = Image.open(f)
        w,h = im.size
        rgbim = im.convert('RGB')
        res = []
        for i in xrange(0,w):
            for j in xrange(0,h):
                r,g,b = rgbim.getpixel((i,j))
                res.append((r*scale,g*scale,b*scale))

        np.random.shuffle(res)
        self.colors = res

    def line(self, xy):
        cx = self.ctx
        cx.move_to(xy[0,0],xy[0,1])
        for (x,y) in xy[1:]:
            cx.line_to(x,y)
        cx.stroke()

    def circles(self, xy, rr):
        cx = self.ctx
        for r, (x,y) in zip(rr, xy):
            cx.arc(x,y,r,0,TWOPI)
            cx.fill()

    def circle(self, xy, r):
        self.ctx.arc(xy[0],xy[1],r,0,TWOPI)
        self.ctx.stroke()

class Path(object):

    def __init__(self, xy, r):

        self.xy = xy
        self.r = r
        self.tree = cKDTree(xy)

    def trace(self, the):

        r = self.r
        numxy = len(self.xy)

        all_near_inds = self.tree.query_ball_point(self.xy,r)

        near_last = []
        circles = []
        for k, inds in enumerate(all_near_inds):

            ii = set(inds)
            isect = ii.intersection(near_last)

            if len(isect)<5:
                near_last = inds
                circles.append((k,inds))

        ## attach last node.
        if circles[-1][0] < numxy-1:
            circles.append((numxy-1,all_near_inds[-1]))

        alphas = []
        for k, inds in circles:
            ## TODO: test average angle?
            inds_s = np.array(sorted(inds),'int')
            xy_diff_sum = self.xy[inds_s[-1],:] - self.xy[inds_s[0], :]

            # xy_diff = self.xy[inds_s[1:],:] - self.xy[inds_s[:-1], :]
            # xy_diff_sum = np.sum(xy_diff, axis=0)

            alpha = np.arctan2(xy_diff_sum[1], xy_diff_sum[0])
            alphas.append(alpha)

        alphas = np.array(alphas) + the
        xy_circles = np.row_stack([self.xy[k,:] for k,_ in circles])
        dx = np.cos(alphas*DXR)
        dy = np.sin(alphas*DYR)
        xy_new = xy_circles[:,:] + np.column_stack((dx*r,dy*r))

        self.xy_circles = xy_circles
        self.xy_new = xy_new

    def noise(self):
        # rnd = lambda s: np.random.normal(size=s)
        rnd = lambda s: 1.-2.*np.random.random(size=s)
        alpha_noise = rnd(len(self.xy_new))*np.pi
        noise = np.column_stack([np.cos(alpha_noise), np.sin(alpha_noise)])*self.r*NOISE
        self.xy_new += noise

    def interpolate(self, num_p_multiplier):
        num_points = len(self.xy_circles)*num_p_multiplier
        tck,u = interpolate.splprep([self.xy_new[:,0], self.xy_new[:,1]],s=0)
        unew = np.linspace(0, 1, num_points)
        out = interpolate.splev(unew,tck)
        self.xy_interpolated = np.column_stack(out)

def get_limit_indices(xy, top, bottom):
    start = 0
    stop = len(xy)

    top_ymask = (xy[:,1]<top).nonzero()[0]
    if top_ymask.any():
        start = top_ymask.max()

    bottom_ymask = (xy[:,1]>bottom).nonzero()[0]
    if bottom_ymask.any():
        stop = bottom_ymask.min()

    return start, stop

def main(infile='./img/img'):
    # pix = lambda i: np.sqrt(1+i)*ONE # gradient-like distribution of lines
    pix = lambda i: PIX_BETWEEN*ONE # linear distribution of lines

    render = Render(SIZE)
    render.ctx.set_source_rgba(*FRONT)
    render.ctx.set_line_width(ONE) # doesn't seem to do anything

    xy = np.column_stack((np.ones(NUMMAX)*START_X, np.linspace(START_Y, STOP_Y, NUMMAX)))
    draw_start, draw_stop = get_limit_indices(xy, top=START_Y, bottom=STOP_Y)
    last_xy = xy[draw_start:draw_stop,:]

    for i in count():

        path = Path(xy, pix(i))
        path.trace(-PIHALF)
        path.noise()
        path.interpolate(int(pix(i)/ONE)*2)
        xy = path.xy_interpolated

        ## remove nodes above and below canvas
        canvas_start, canvas_stop = get_limit_indices(xy, top=0., bottom=1.)
        xy = xy[canvas_start:canvas_stop, :]

        ## render nodes above STOP_Y and below START_Y
        draw_start, draw_stop = get_limit_indices(xy, top=START_Y, bottom=STOP_Y)
        render.circles(xy[draw_start:draw_stop,:], np.ones(draw_stop-draw_start)*ONE)

        ## io and loop stuff
        xmax = xy[:,0].max()
        if (xmax > STOP_X):
            break
        print 'num', i, 'points', len(path.xy_circles), 'x', xmax
        if i % 50 == 0:
            if i > 0:
                fn = '{:s}_{:05d}.png'.format(infile, i)
                print fn
                render.sur.write_to_png(fn)

    fn = '{:s}_final.png'.format(infile)
    print fn
    render.sur.write_to_png(fn)


if __name__ == '__main__':
    main()
