from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os
import time
from argparse import ArgumentParser, Namespace
import sys
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    start = time.time()
    model = load_model(model_path, device)
    print(f'model load time:{time.time()-start}')
    start = time.time()
    # load_images can take a list of images or a directory
    img_path = args.img_path
    images,img_list = load_images(img_path,size=None)
    pairs = make_pairs(images, scene_graph='swin', prefilter=None, symmetrize=False) #build image pairs
    print(f'data process time:{time.time()-start}')
    start = time.time()
    output = inference(pairs, model, device, batch_size=batch_size)
    print(f'inference time:{time.time()-start}')
    start = time.time()
    print(list(output.keys()))
    print(list(output['pred1'].keys()))
    print(list(output['view1'].keys()))
    print(list(output['pred2'].keys()))
    print(list(output['view2'].keys()))
    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    print(f'Alignment time:{time.time()-start},loss:{loss}')
    start = time.time()

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    intrinsics = scene.get_intrinsics()
    confidence_masks = scene.get_masks()
    print(f'Alignment inference time:{time.time()-start}')
    start = time.time()
    print(poses.shape,intrinsics.shape,len(pts3d))
    print(pts3d[0].shape)
    import json
    json.dump('pose_dust3r.json',)

    #for i,img_name in enumerate(img_list)

    # visualize reconstruction
    #scene.show()

    # find 2D-2D matches between the two images
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    print(f'Matches finiding time:{time.time()-start}')
    start = time.time()
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]