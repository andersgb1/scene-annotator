#!/usr/bin/env python

import argparse, numpy, os, yaml

import sys
sys.path.append('../covis/build/lib')

from covis import *

# Setup program options
po = argparse.ArgumentParser()

po.add_argument("root", default="/path/to/sixd/dataset", type=str, help="root path of your dataset")

po.add_argument("--object", '-o', default="models/obj_01.ply", type=str, help="relative path to object file")
po.add_argument("--scene-dir", '-s', default="test/01", type=str, help="relative path to directory with scenes")
po.add_argument("--test-set", '-e', default="", type=str, help="relative path to test set list (set to e.g. test_set_v1.yml to enable)")
po.add_argument("--threshold", '-t', default=10, type=float, help="occlusion threshold, set to zero or less to disable occlusion reasoning")
po.add_argument("--output-dir", default="./output", type=str, help="specify output directory root")

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

scenedird = scenedir + '/depth'
assert os.path.isdir(scenedird)
dlist = sorted(os.listdir(scenedird))
assert len(rgblist) == len(dlist)

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

if args.test_set:
    testfile = args.root + '/' + args.test_set
    print('Loading test set list from {}...'.format(testfile))
    with open(testfile, 'r') as stream:
        try:
            test_set_data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        scnid_string = args.scene_dir[-2:]
        scnid = int(scnid_string)
        test_set = test_set_data[scnid]
        print('\tGot {} scenes in test set'.format(len(test_set)))
        gtdata = [gtdata[i] for i in test_set]
        camdata = [camdata[i] for i in test_set]

print('GT/cam/rgb list length: {}/{}/{}'.format(len(gtdata), len(camdata), len(rgblist)))
assert len(gtdata) == len(camdata) == len(rgblist)

outdir = args.output_dir + '/' + args.root[args.root.rfind('/')+1:] + '/' + args.scene_dir[-2:] + '/mask_gt/' + objid_string
if not os.path.isdir(outdir):
    print('Creating output directory {}...'.format(outdir))
    os.makedirs(outdir)

seqid_string = args.scene_dir[-2:]
print('Traversing scenes in sequence {}...'.format(seqid_string))
for i in range(len(gtdata)):
    # Load scene images
    from scipy import misc
    rgbfile = scenedirrgb + '/' + rgblist[i]
    assert os.path.isfile(rgbfile)
    dfile = scenedird + '/' + dlist[i]
    assert os.path.isfile(dfile)
    img = misc.imread(rgbfile)
    depth = misc.imread(dfile, mode='I')
#    from skimage.viewer import ImageViewer
#    viewer = ImageViewer(depth.astype(float))
#    viewer.show()
#    depth = depth.astype(float)

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
    img_masked = numpy.zeros((img.shape[0],img.shape[1]), dtype=numpy.uint8)
    for T in Tlist:
        # Format pose
        R = numpy.asarray(T['cam_R_m2c']).reshape((3,3))
        t = numpy.asarray(T['cam_t_m2c']).reshape((3,1))
        Ti = numpy.vstack((numpy.hstack((R,t)), numpy.array([0,0,0,1])))
        # Transform the object coordinates into the scene
        xyz = model.cloud.array()[0:3,:]
        xyz = numpy.matmul(R, xyz) + numpy.tile(t, xyz.shape[1])
        # Project to pixel coordinates
        xy = xyz[0:2,:] / xyz[2,:] # Normalize by z
        xy[0, :] = fx * xy[0, :] + cx
        xy[1, :] = fy * xy[1, :] + cy
        xy = numpy.round(xy).astype(int)
        z = xyz[2,:] # Maintain depth for use below
        
        # Remove pixels beyond image borders
        mask = numpy.logical_and(numpy.logical_and(xy[0, :] >= 0, xy[0, :] <= img.shape[1] - 1),
                                 numpy.logical_and(xy[1, :] >= 0, xy[1, :] <= img.shape[0] - 1))
        xy = xy[:,mask]
        z = z[mask]
        
        # Remove pixels behind scene data
        if args.threshold > 0:
            mask = numpy.zeros_like(z, dtype=bool)
            for j in range(z.shape[0]):
                if depth[xy[1,j], xy[0,j]] > 0:
                    if abs(depth[xy[1,j], xy[0,j]] - z[j]) < args.threshold:           
#                    if depth[xy[1,j], xy[0,j]] < z[j] - args.threshold:
                        mask[j] = True
            xy = xy[:, mask]
            z = z[mask]
        
        img_masked[xy[1,:], xy[0,:]] = 255 # Row index (y) comes first

    # Remove pepper noise
    from skimage import morphology
    img_masked = morphology.closing(img_masked)

    # Save result
    outfile = outdir + '/' + rgblist[i]
    print('\tSaving output file {} with {} annotated instance(s)...'.format(outfile, len(Tlist)))
    misc.imsave(outfile, img_masked)

