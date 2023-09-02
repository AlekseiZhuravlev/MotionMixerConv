import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch


def visualize_batch_single_ais(batch_full):

    xs = batch_full[:, ::3]
    ys = batch_full[:, 1::3]
    zs = batch_full[:, 2::3]

    KPS_PARENT=[-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    joints_ais = [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
        "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
    ]
    j=0
    ax.clear()
    ax.scatter(xs[j], ys[j],
               zs[j])

    for i in range(len(joints_ais)):
        ax.text(xs[j, i], ys[j, i],
               zs[j, i], joints_ais[i], size=8, zorder=1, color='k')

    ax.set_xlim3d([xs.min(), xs.max()])
    ax.set_xlabel('X')
    #
    ax.set_ylim3d([ys.min(), ys.max()])
    ax.set_ylabel('Y')
    #
    ax.set_zlim3d([zs.min(), zs.max()])
    ax.set_zlabel('Z')

    ax.set_title('Skeleton model, AIS dataset')

    for kp_idx in range(1, len(xs[j])):
        lxs = torch.stack([xs[j, KPS_PARENT[kp_idx]], xs[j, kp_idx]])
        lys = torch.stack([ys[j, KPS_PARENT[kp_idx]], ys[j, kp_idx]])
        lzs = torch.stack([zs[j, KPS_PARENT[kp_idx]], zs[j, kp_idx]])

        ax.plot(lxs, lys, lzs)

def make_animation_ais(batch_full, add_title, add_joint_names):
    xs = batch_full[:, ::3]
    ys = batch_full[:, 1::3]
    zs = batch_full[:, 2::3]

    KPS_PARENT=[-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    joints_ais = [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee",
        "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"
    ]

    images = []
    for j in range(batch_full.shape[0]):
        ax.clear()
        ax.scatter(xs[j], ys[j],
                   zs[j])

        if add_joint_names:
            for i in range(len(joints_ais)):
                ax.text(xs[j, i], ys[j, i],
                        zs[j, i], joints_ais[i], size=8, zorder=1, color='k')

        ax.set_xlim3d([xs.min(), xs.max()])
        ax.set_xlabel('X')
        #
        ax.set_ylim3d([ys.min(), ys.max()])
        ax.set_ylabel('Y')
        #
        ax.set_zlim3d([zs.min(), zs.max()])
        ax.set_zlabel('Z')

        if add_title:
            ax.set_title('Skeleton model, AIS dataset')

        for kp_idx in range(1, len(xs[j])):
            lxs = torch.stack([xs[j, KPS_PARENT[kp_idx]], xs[j, kp_idx]])
            lys = torch.stack([ys[j, KPS_PARENT[kp_idx]], ys[j, kp_idx]])
            lzs = torch.stack([zs[j, KPS_PARENT[kp_idx]], zs[j, kp_idx]])

            ax.plot(lxs, lys, lzs)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image_from_plot)

    # destroy the figure so that the next one can be drawn
    plt.close(fig)

    # show all images as animation
    fig = plt.figure()
    plt.axis('off')
    ims = [[plt.imshow(image, animated=True)] for image in images]
    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=0, blit=True)

    # save animation as gif
    ani.save('/home/azhuavlev/PycharmProjects/MotionMixerConv/conv_mixer/animation_ais.gif', writer='pillow')

    # save animation as mp4
    # ani.save('/home/azhuavlev/PycharmProjects/MotionMixerConv/conv_mixer/animation_ais.mp4', writer='ffmpeg')

    plt.close()