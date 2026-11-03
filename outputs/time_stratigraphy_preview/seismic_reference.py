"""Shared model/PSF methods frozen from the local scenario implementation.
Image-domain approximation, not full wave propagation.
"""
import json
from pathlib import Path
import numpy as np
from scipy.fft import rfftn,irfftn,rfftfreq,fftfreq,next_fast_len,fftshift

def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def voxelize_fan(z_interfaces,q_layers,x,y,z,scale,architecture,reference,datum_elevation=None,burial_depth=1400.):
    nx,ny=len(x),len(y);nz=len(z);dz=float(z[1]-z[0]);nc=nx*ny
    if architecture=='original_relief':
        if datum_elevation is None:raise ValueError('Original-relief geometry requires a common pre-depositional regional datum')
        # Positive elevation is upwards; positive model depth is downwards.
        # Only one fitted regional initial plane is removed. The final top of
        # each column is never used to flatten or anchor that column.
        d=burial_depth-scale*(z_interfaces-np.asarray(datum_elevation).reshape(-1,1))
    elif architecture=='topographic':d=1400.+scale*(reference-z_interfaces)
    else:
        # A separately labelled accommodation-space interpretation: column
        # tops aligned to one horizon; thickness and vertical order preserved.
        xx,yy=np.meshgrid(x,y,indexing='ij')
        top=550.+5*np.sin(2*np.pi*xx/6000)+3*np.sin(2*np.pi*yy/4000)
        d=top.ravel()[:,None]+scale*(z_interfaces[:,0,None]-z_interfaces)
    if np.min(d)<30 or np.max(d)>z[-1]-50:
        raise ValueError('Scaled fan does not fit scenario depth range; adjust scale/depth')
    edges=np.r_[z-dz/2,z[-1]+dz/2]
    qvol=np.zeros((nc,nz),dtype='f4');occ=np.zeros_like(qvol)
    for col in range(nc):
        h=np.maximum(0,np.diff(d[col]));s=np.r_[0,np.cumsum(h*q_layers[col])]
        v=np.diff(np.interp(edges,d[col],s,left=0,right=s[-1]))
        qvol[col]=np.clip(v/dz,0,1)
        occupied=np.maximum(0,np.minimum(edges[1:],d[col,-1])-np.maximum(edges[:-1],d[col,0]))
        occ[col]=occupied/dz
    return qvol.reshape(nx,ny,nz),occ.reshape(nx,ny,nz),d[:,0].reshape(nx,ny),d[:,-1].reshape(nx,ny),d

def rock_model(q,xx,yy,z,seed):
    rng=np.random.default_rng(seed)
    edges=[-100.]
    while edges[-1]<z[-1]+100:edges.append(edges[-1]+float(rng.uniform(14,38)))
    edges=np.array(edges);n=len(edges)+1
    dv=rng.normal(0,55,n);dr=.25*dv+rng.normal(0,10,n)
    strong=np.arange(4,n,11);dv[strong]+=rng.choice([-1,1],len(strong))*90;dr[strong]+=.25*dv[strong]
    warp=9*np.sin(2*np.pi*xx/3500+.3*np.cos(2*np.pi*yy/4400))+5*np.cos(2*np.pi*(xx/7000+yy/5300))
    strat=z[None,None,:]-warp[:,:,None]*(.3+.7*z[None,None,:]/1800)
    ids=np.searchsorted(edges,strat).astype('u2')
    # Mud porosity/clay/silt variability is an explicit stratified rock-physics
    # hypothesis, added before reflectivity; no seismic texture is superposed.
    mud_weight=(1-q)
    vp=2200.+.35*z[None,None,:]+400*q+dv[ids]*mud_weight
    rho=2050.+.12*z[None,None,:]+160*q+dr[ids]*mud_weight
    return vp.astype('f4'),rho.astype('f4'),ids,edges

def psf_image(reflectivity,spacing,vref,f0,angle0=25.,angle1=40.):
    # Frequencies are cycles/metre: exploding-reflector dispersion gives
    # equivalent temporal f = vref * |spatial frequency| / 2.
    original=reflectivity.shape;pad=[16,16,48]
    shape=tuple(next_fast_len(original[i]+2*pad[i]) for i in range(3))
    work=np.zeros(shape,dtype='f4');slices=tuple(slice(pad[i],pad[i]+original[i]) for i in range(3));work[slices]=reflectivity
    spectrum=rfftn(work,workers=2);del work
    fx=fftfreq(shape[0],spacing[0]).astype('f4')[:,None,None]
    fy=fftfreq(shape[1],spacing[1]).astype('f4')[None,:,None]
    fz=rfftfreq(shape[2],spacing[2]).astype('f4')[None,None,:]
    kh=np.sqrt(fx*fx+fy*fy)
    freq=vref/2*np.sqrt(kh*kh+fz*fz)
    theta=np.degrees(np.arctan2(kh,np.maximum(fz,1e-20)))
    taper=np.clip((theta-angle0)/(angle1-angle0),0,1)
    aperture=(.5*(1+np.cos(np.pi*taper))).astype('f4')
    u=(freq/f0)**2
    # Scale to a unit-peak depth Ricker on a horizontally constant reflector.
    scale=vref/(np.sqrt(np.pi)*f0*spacing[2]*np.e)
    transfer=(np.e*u*np.exp(-u)*aperture*scale).astype('f4')
    image=irfftn(spectrum*transfer,s=shape,workers=2)[slices].copy()
    kernel=fftshift(irfftn(transfer,s=shape,workers=2))
    center=np.array(shape)//2;half=[min(20,center[0]-1),min(20,center[1]-1),min(48,center[2]-1)]
    ks=tuple(slice(center[i]-half[i],center[i]+half[i]+1) for i in range(3))
    compact=kernel[ks].copy()
    offsets=[np.arange(-half[i],half[i]+1)*spacing[i] for i in range(3)]
    return image.astype('f4'),compact,offsets
