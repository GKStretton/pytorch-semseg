import os
from PIL import Image
import torch
import argparse
import numpy as np
import scipy.misc as misc
from skimage import io
import matplotlib.pyplot as plt


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict
from ptsemseg.metrics import runningScore

import time

start = time.time()

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


def decode_segmap(label_mask):
    label_colours = np.asarray([[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 255], [0, 255, 0]])
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for l in range(5):
        r[label_mask == l] = label_colours[l, 0]
        g[label_mask == l] = label_colours[l, 1]
        b[label_mask == l] = label_colours[l, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def test(args):
    print("Starting...")

    base = "/home/greg/datasets/rootset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]


    data_loader = get_loader(args.dataset)
    loader = data_loader(root=base, mode=args.mode, is_transform=True, img_norm=args.img_norm, test_mode=True)
    n_classes = loader.n_classes

    # Setup image
    img_path = args.image + ".png"
    print("Read Input Image from : {}".format(base + "/images/" + img_path))
    img = io.imread(base + "/images/" + img_path)
    if n_classes == 2:
        gt = io.imread(base + "/segmentation8/" + img_path, 0)
    elif n_classes == 5:
        gt = io.imread(base + "/combined/" + img_path, 0)
    #gt = np.array(gt, dtype=np.int8)
    #if n_classes == 2:
    #    gt[gt == -1] = 1
        


    running_metrics = runningScore(n_classes)

    #resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp="bicubic")

    orig_size = img.shape[:-1]
    #if model_name in ["pspnet", "icnet", "icnetBN", "root"]:
        # uint8 with RGB mode, resize width and height which are odd numbers
    #img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
    resized_image = np.array(Image.fromarray(img).resize((orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1)))
    #else:
    #    img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if args.img_norm:
        img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path, map_location='cpu')["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    print("Running network...")
    outputs = model(images)

    if args.dcrf:
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        unary = np.ascontiguousarray(unary)

        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = args.out_path[:-4] + "_drf.png"
        misc.imsave(dcrf_path, decoded_crf)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        pred = pred.astype(np.float32)
        # float32 with F mode, resize back to orig_size
        pred = misc.imresize(pred, orig_size, "nearest", mode="F")

    running_metrics.update(gt, pred)

    decoded = loader.decode_segmap(pred)
    decoded = decoded.astype(np.float32)
    origimg = io.imread(base + "/images/" + img_path)
    origimg = origimg.astype(np.float32) / 255.0
    gtimg = decode_segmap(gt)
    
    stack = np.hstack((origimg, gtimg, decoded[:origimg.shape[0],:origimg.shape[1]]))
    print("Done!")
    print("Time taken:", int((time.time() - start)*10)/10.0, "seconds")

    if n_classes == 5:
        score, class_iou = running_metrics.get_scores()

        print("\n===IoU===")
        for i in range(1,n_classes):
            classnames = ["Background", "Root", "Seed", "Lateral tips", "Primary tips"]
            print("%s:\t%.2f" % (classnames[i], class_iou[i]))

    plt.imshow(stack)
    plt.show()
    tosave = (255*decoded).astype(np.uint8)
    io.imsave("final/output/" + img_path, tosave)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--mode",
        nargs="?",
        type=str,
        default="multi",
        help="mode: foreback/multi",
    )
    parser.add_argument(
        "--image",
        default="1317",
        help="Image number e.g. 1317"
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="pascal",
        help="Dataset to use ['pascal, camvid, ade20k etc']",
    )

    parser.add_argument(
        "--img_norm",
        dest="img_norm",
        action="store_true",
        help="Enable input image scales normalization [0, 1] \
                              | True by default",
    )
    parser.add_argument(
        "--no-img_norm",
        dest="img_norm",
        action="store_false",
        help="Disable input image scales normalization [0, 1] |\
                              True by default",
    )
    parser.set_defaults(img_norm=True)

    parser.add_argument(
        "--dcrf",
        dest="dcrf",
        action="store_true",
        help="Enable DenseCRF based post-processing | \
                              False by default",
    )
    parser.add_argument(
        "--no-dcrf",
        dest="dcrf",
        action="store_false",
        help="Disable DenseCRF based post-processing | \
                              False by default",
    )
    parser.set_defaults(dcrf=False)

    parser.add_argument(
        "--imgs", nargs="?", type=str, default=None, help="Path of the input image"
    )
    parser.add_argument(
        "--out_path", nargs="?", type=str, default=None, help="Path of the output segmap"
    )
    args = parser.parse_args()
    test(args)
