#!/usr/bin/env python
#coding:utf-8


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import glob
import os

from PIL import Image
import torch
from torchvision import utils, datasets, transforms
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import explained_variance_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from vae import VAE
#from vae_inception import VAE
from utils import load_model


def main(args):
    transform = transforms.Compose([transforms.Resize([64, 64], 1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([.5], [.5])])

    vae = VAE(img_size=[3, 64, 64], z_dim=args.z_dim)
    pt_file = load_model(args.model_load_path, "*200.pt")
    vae.load_state_dict(torch.load(pt_file))
    vae.eval()

    if args.extract_z:
        dirs = glob.glob(os.path.join(args.data_path, "*"))
        dirs.sort()

        for d in dirs:
            data = []
            files = glob.glob(os.path.join(d, "*.png"))
            files.sort()
            print(d.split("/")[-1])
            for f in files:
                img = Image.open(f)
                img = transform(img)
                data.append(vae.reparam(*vae.encoder(img[None, :, :, :])).detach())
            data = torch.cat(data).cpu().numpy()
            #np.savetxt(d.split("/")[-1] + ".txt", data * 0.8 / 4, delimiter=" ")
            np.savetxt(d.split("/")[-1] + ".txt", data, delimiter=",")


    if args.vis_z:
        dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
        data, label = [], []
        for i, j in dataset:
            data.append(i)
            label.append(j)
        data = torch.stack(data)
        label = torch.tensor(label)

        mu, log_var = vae.encoder(data)
        z = vae.reparam(mu, log_var)

        pca = PCA(n_components=3)
        #pca = TSNE(n_components=3)
        #pca = KernelPCA(n_components=3, kernel="rbf", fit_inverse_transform=True)
        z = z.detach().numpy()
        if z.shape[-1] > 3:
            comp_z = pca.fit_transform(z)
        else:
            comp_z = z
        #print(explained_variance_score(z.detach().numpy(), pca.inverse_transform(comp_z)))
        #print("evr:", pca.explained_variance_ratio_)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection="3d")
        cmap = plt.get_cmap("tab20")
        for i in range(label.max() + 1):
            ax.scatter(comp_z[label == i, 0], comp_z[label == i, 1],comp_z[label == i, 2], 
                       marker="${}$".format(i), color=cmap(i), label=dataset.classes[i])
            plt.legend(loc="best")
        plt.show()



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--z_dim", type=int, default=7)
    #parse.add_argument("--data_path", type=str, default="./data/20200605/validation/env/goal000/")
    parse.add_argument("--data_path", type=str, default="../result/goal000to003/gen_goal/")
    parse.add_argument("--model_load_path", type=str, default="./result/result200706_z10b1/VAE/model/")
    parse.add_argument("--extract_z", type=bool, default=False)
    parse.add_argument("--vis_z", type=bool, default=True)

    main(parse.parse_args())

