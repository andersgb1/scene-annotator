#!/usr/bin/env python

import argparse, numpy, os, yaml

import sys
sys.path.append('../covis/build/lib')

from covis import *

# Setup program options
po = argparse.ArgumentParser()

po.add_argument("root", default="/path/to/sixd", type=str, help="root path of your dataset")

po.add_argument("--object", '-o', default="models/obj_01.ply", type=str, help="relative path to object file")
po.add_argument("--scene-dir", '-s', default="test/01", type=str, help="relative path to directory with scenes")
po.add_argument("--output-dir", default="./output", type=str, help="specify output directory")

args = po.parse_args()

print('Loading object file {}...'.format(args.root + '/' + args.object))
model = util.load(args.root + '/' + args.object)
idot = args.object.rfind('.')
objid_string = args.object[idot-2:idot]
objid = int(objid_string)
print('\tObject ID: {}'.format(objid))

scenedir = args.root + '/' + args.scene_dir
assert os.path.isdir(scenedir)

scenedirrgb = scenedir + '/rgb'
print('Loading scene file list from directory {}...'.format(scenedirrgb))
assert os.path.isdir(scenedirrgb)
rgblist = sorted(os.listdir(scenedirrgb))
print('\tGot {} files from {} to {}'.format(len(rgblist), rgblist[0], rgblist[-1]))

gtfile = scenedir + '/gt.yml'
print('Loading GT poses from {}...'.format(gtfile))
assert os.path.isfile(gtfile)
with open(gtfile, 'r') as f:
    gtdata = yaml.load(f)

infofile = scenedir + '/info.yml'
print('Loading camera info from {}...'.format(infofile))
assert os.path.isfile(infofile)
with open(infofile, 'r') as f:
    camdata = yaml.load(f)

if not os.path.isdir(args.output_dir):
    print('Creating output directory {}...'.format(args.output_dir))
    os.makedirs(args.output_dir)

assert len(rgblist) == len(gtdata) == len(camdata)

seqid_string = args.scene_dir[-2:]
print('Traversing scenes in sequence {}...'.format(seqid_string))
for i in range(len(rgblist)):
    # Load scene image
    from scipy import misc
    rgbfile = scenedirrgb + '/' + rgblist[i]
    assert os.path.isfile(rgbfile)
    img = misc.imread(rgbfile)

    # Load intrinsic matrix
    assert camdata[i]['depth_scale'] == 1
    K = camdata[i]['cam_K']
    fx,cx,fy,cy = K[0],K[2],K[4],K[5]

    # Load GT poses for object
    Tlist = gtdata[i]
    tmp = []
    for T in Tlist:
        if T['obj_id'] == objid:
            tmp.append(T)
    Tlist = tmp

    # Generate masks for each object instance, put them into this binary image (0 for bg, 255 for fg)
    img_masked = numpy.zeros_like(img)
    for T in Tlist:
        R = numpy.asarray(T['cam_R_m2c']).reshape((3,3))
        t = numpy.asarray(T['cam_t_m2c']).reshape((3,1))
        Ti = numpy.vstack((numpy.hstack((R,t)), numpy.array([0,0,0,1])))
        xyz = model.cloud.array()[0:3,:]
        xyz = numpy.matmul(R, xyz) + numpy.tile(t, xyz.shape[1])
        xy = xyz[0:2,:] / xyz[2,:] # Normalize by z
        xy[0,:] = fx * xy[0,:] + cx
        xy[1, :] = fy * xy[1, :] + cy
        xy = numpy.round(xy).astype(int)
        xy[0, :] = numpy.clip(xy[0, :], 0, img.shape[1] - 1)
        xy[1, :] = numpy.clip(xy[1, :], 0, img.shape[0] - 1)
        img_masked[xy[1,:], xy[0,:]] = 255 # Row index (y) comes first

    # Remove pepper noise
    from skimage import morphology
    img_masked = morphology.closing(img_masked)

    # Save result
    outfile = args.output_dir + '/' + objid_string + '_' + seqid_string + '_' + rgblist[i]
    print('\tSaving output file {} with {} annotated instances...'.format(outfile, len(Tlist)))
    misc.imsave(outfile, img_masked)
