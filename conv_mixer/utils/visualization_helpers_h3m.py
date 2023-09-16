import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch

def visualize_batch_single_h3m(batch_full):
    # ignore constant joints and joints at same position with other joints
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)
    joints_to_use = np.setdiff1d(np.arange(32), joint_to_ignore)

    xs_raw = batch_full[:, ::3] / 1000
    ys_raw = batch_full[:, 1::3] / 1000
    zs_raw = batch_full[:, 2::3] / 1000

    xs = xs_raw
    ys = -zs_raw
    zs = ys_raw

    connect = np.array([
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
        (0, 1), (0, 6),
        (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
        (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (24, 25), (24, 17),
        (24, 14), (14, 15)
    ])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    joint_name_h3m = np.array([
        "Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg", "LeftFoot",
        "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm", "LeftForeArm",
        "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm", "RightForeArm",
        "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"
    ])

    for j in range(1):
        ax.clear()
        ax.scatter(xs[j, dimensions_to_use // 3], ys[j, dimensions_to_use // 3],
                   zs[j, dimensions_to_use // 3])

        for i in range(len(joints_to_use)):
            ax.text(xs[j, joints_to_use[i]], ys[j, joints_to_use[i]],
                   zs[j, joints_to_use[i]], joint_name_h3m[joints_to_use[i]], size=8, zorder=1, color='k')



        ax.set_xlim3d([xs.min(), xs.max()])
        ax.set_xlabel('X')
        #
        ax.set_ylim3d([ys.min(), ys.max()])
        ax.set_ylabel('Y')
        #
        ax.set_zlim3d([zs.min(), zs.max()])
        ax.set_zlabel('Z')

        ax.set_title('Skeleton model, H3.6M dataset')

        for i in range(len(connect)):
            ax.plot(xs[j][connect[i]], ys[j][connect[i]], zs[j][connect[i]])



def visualize_batch(batch_full, save_path, batch_gt, batch_train):
    # ignore constant joints and joints at same position with other joints
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)

    xs_raw = batch_full[:, ::3] / 1000
    ys_raw = batch_full[:, 1::3] / 1000
    zs_raw = batch_full[:, 2::3] / 1000

    xs = xs_raw.numpy()
    ys = -zs_raw.numpy()
    zs = ys_raw.numpy()

    if batch_gt is not None:
        xs_raw_gt = batch_gt[:, ::3] / 1000
        ys_raw_gt = batch_gt[:, 1::3] / 1000
        zs_raw_gt = batch_gt[:, 2::3] / 1000

        xs_gt = xs_raw_gt.numpy()
        ys_gt = -zs_raw_gt.numpy()
        zs_gt = ys_raw_gt.numpy()

    if batch_train is not None:
        xs_raw_train = batch_train[:, ::3] / 1000
        ys_raw_train = batch_train[:, 1::3] / 1000
        zs_raw_train = batch_train[:, 2::3] / 1000

        xs_train = xs_raw_train.numpy()
        ys_train = -zs_raw_train.numpy()
        zs_train = ys_raw_train.numpy()

    connect = np.array([
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
        (0, 1), (0, 6),
        (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
        (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (24, 25), (24, 17),
        (24, 14), (14, 15)
    ])

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

            ax.set_xlim3d([xs.min(), xs.max()])
            ax.set_xlabel('X')
            #
            ax.set_ylim3d([ys.min(), ys.max()])
            ax.set_ylabel('Y')
            #
            ax.set_zlim3d([zs.min(), zs.max()])
            ax.set_zlabel('Z')

            ax.scatter(xs_train[j, dimensions_to_use // 3], ys_train[j, dimensions_to_use // 3],
                          zs_train[j, dimensions_to_use // 3], c='#07529a', label='gt')
            for i in range(len(connect)):
                ax.plot(xs_train[j][connect[i]], ys_train[j][connect[i]], zs_train[j][connect[i]], c='#07529a')
                # make legend

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
            for i in range(len(connect)):
                ax.plot(xs_gt[j][connect[i]], ys_gt[j][connect[i]], zs_gt[j][connect[i]], c='#07529a')

        # plot prediction
        ax.scatter(xs[j, dimensions_to_use // 3], ys[j, dimensions_to_use // 3],
                   zs[j, dimensions_to_use // 3], c='#eab90c')
        for i in range(len(connect)):
            ax.plot(xs[j][connect[i]], ys[j][connect[i]], zs[j][connect[i]], c='#eab90c')

        # set the scales
        ax.set_xlim3d([xs.min(), xs.max()])
        ax.set_xlabel('X')
        #
        ax.set_ylim3d([ys.min(), ys.max()])
        ax.set_ylabel('Y')
        #
        ax.set_zlim3d([zs.min(), zs.max()])
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