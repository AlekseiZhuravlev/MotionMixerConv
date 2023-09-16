import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch


def visualize_batch_single_ais(batch_full):

    xs = batch_full[:, ::3] / 1000
    ys = batch_full[:, 1::3] / 1000
    zs = batch_full[:, 2::3] / 1000

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


def make_animation_ais(batch_full, add_title, add_joint_names, save_name):
    xs = batch_full[:, ::3] / 1000
    ys = batch_full[:, 1::3] / 1000
    zs = batch_full[:, 2::3] / 1000

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
    ani.save(f'/home/azhuavlev/PycharmProjects/MotionMixerConv/conv_mixer/{save_name}.gif', writer='pillow')

    # save animation as mp4
    # ani.save('/home/azhuavlev/PycharmProjects/MotionMixerConv/conv_mixer/animation_ais.mp4', writer='ffmpeg')

    plt.close()



def visualize_batch_ais(batch_full, save_path, batch_gt, batch_train):
    # ignore constant joints and joints at same position with other joints

    dimensions_to_use = np.arange(57)


    KPS_PARENT=[-1, 0, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16]

    xs = batch_full[:, ::3].numpy() #/ 1000
    ys = batch_full[:, 1::3].numpy() #/ 1000
    zs = batch_full[:, 2::3].numpy() #/ 1000

    if batch_gt is not None:
        xs_gt = batch_gt[:, ::3].numpy() #/ 1000
        ys_gt = batch_gt[:, 1::3].numpy() #/ 1000
        zs_gt = batch_gt[:, 2::3].numpy() #/ 1000

    if batch_train is not None:
        xs_train = batch_train[:, ::3].numpy() #/ 1000
        ys_train = batch_train[:, 1::3].numpy() #/ 1000
        zs_train = batch_train[:, 2::3].numpy() #/ 1000

    plt.rcParams['grid.linewidth'] = 0.2
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    images = []

    # draw training data
    if batch_gt is not None:
        for j in range(batch_train.shape[0]):
            ax.clear()

            if batch_train is not None and batch_gt is not None:
                # find min and max x in xs, xs_train, xs_gt
                min_x = min(xs.min(), xs_train.min(), xs_gt.min())
                max_x = max(xs.max(), xs_train.max(), xs_gt.max())

                min_y = min(ys.min(), ys_train.min(), ys_gt.min())
                max_y = max(ys.max(), ys_train.max(), ys_gt.max())

                min_z = min(zs.min(), zs_train.min(), zs_gt.min())
                max_z = max(zs.max(), zs_train.max(), zs_gt.max())

                ax.set_xlim3d([min_x, max_x])
                ax.set_ylim3d([min_y, max_y])
                ax.set_zlim3d([min_z, max_z])
            else:
                ax.set_xlim3d([xs.min(), xs.max()])
                ax.set_ylim3d([ys.min(), ys.max()])
                ax.set_zlim3d([zs.min(), zs.max()])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.scatter(xs_train[j, dimensions_to_use // 3], ys_train[j, dimensions_to_use // 3],
                          zs_train[j, dimensions_to_use // 3], c='#07529a', label='gt')

            for kp_idx in range(1, len(xs_train[j])):
                lxs = np.stack([xs_train[j, KPS_PARENT[kp_idx]], xs_train[j, kp_idx]])
                lys = np.stack([ys_train[j, KPS_PARENT[kp_idx]], ys_train[j, kp_idx]])
                lzs = np.stack([zs_train[j, KPS_PARENT[kp_idx]], zs_train[j, kp_idx]])

                ax.plot(lxs, lys, lzs, c='#07529a')

            ax.set_title('Input')

            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image_from_plot)

    # draw predicted data
    for j in range(batch_full.shape[0]):
        ax.clear()

        # plot ground truth
        if batch_gt is not None:
            ax.scatter(xs_gt[j, dimensions_to_use // 3], ys_gt[j, dimensions_to_use // 3],
                          zs_gt[j, dimensions_to_use // 3], c='#07529a', label='gt')

            for kp_idx in range(1, len(xs_gt[j])):
                lxs = np.stack([xs_gt[j, KPS_PARENT[kp_idx]], xs_gt[j, kp_idx]])
                lys = np.stack([ys_gt[j, KPS_PARENT[kp_idx]], ys_gt[j, kp_idx]])
                lzs = np.stack([zs_gt[j, KPS_PARENT[kp_idx]], zs_gt[j, kp_idx]])

                ax.plot(lxs, lys, lzs, c='#07529a')

        # plot prediction
        ax.scatter(xs[j, dimensions_to_use // 3], ys[j, dimensions_to_use // 3],
                   zs[j, dimensions_to_use // 3], c='#eab90c')
        for kp_idx in range(1, len(xs[j])):
            lxs = np.stack([xs[j, KPS_PARENT[kp_idx]], xs[j, kp_idx]])
            lys = np.stack([ys[j, KPS_PARENT[kp_idx]], ys[j, kp_idx]])
            lzs = np.stack([zs[j, KPS_PARENT[kp_idx]], zs[j, kp_idx]])

            ax.plot(lxs, lys, lzs, c='#eab90c')

        # set the scales
        if batch_train is not None and batch_gt is not None:
            ax.set_xlim3d([min_x, max_x])
            ax.set_ylim3d([min_y, max_y])
            ax.set_zlim3d([min_z, max_z])
        else:
            ax.set_xlim3d([xs.min(), xs.max()])
            ax.set_ylim3d([ys.min(), ys.max()])
            ax.set_zlim3d([zs.min(), zs.max()])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title('Prediction')

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
    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=200, blit=True)

    # save animation as gif
    ani.save(save_path, writer='pillow')

    # save animation as mp4
    # ani.save('/home/azhuavlev/PycharmProjects/MotionMixerConv/conv_mixer/animation.mp4', writer='ffmpeg')

    plt.close()