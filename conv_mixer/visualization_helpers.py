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
        # print(xs.shape, j, KPS_PARENT[kp_idx], kp_idx)
        # print(kp_idx, len(KPS_PARENT), len(xs))
        lxs = torch.stack([xs[j, KPS_PARENT[kp_idx]], xs[j, kp_idx]])
        lys = torch.stack([ys[j, KPS_PARENT[kp_idx]], ys[j, kp_idx]])
        lzs = torch.stack([zs[j, KPS_PARENT[kp_idx]], zs[j, kp_idx]])

        # print('lxs', lxs, lxs.shape)
        # print('lys', lys, lys.shape)
        # print('lzs', lzs, lzs.shape)
        ax.plot(lxs, lys, lzs)

    # for i in range(len(connect)):
    #     ax.plot(xs[j][connect[i]], ys[j][connect[i]], zs[j][connect[i]])



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



def visualize_batch(batch_full):
    # ignore constant joints and joints at same position with other joints
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)


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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    images = []
    for j in range(batch_full.shape[0]):
        ax.clear()
        ax.scatter(xs[j, dimensions_to_use // 3], ys[j, dimensions_to_use // 3],
                   zs[j, dimensions_to_use // 3])

        ax.set_xlim3d([xs.min(), xs.max()])
        ax.set_xlabel('X')
        #
        ax.set_ylim3d([ys.min(), ys.max()])
        ax.set_ylabel('Y')
        #
        ax.set_zlim3d([zs.min(), zs.max()])
        ax.set_zlabel('Z')

        for i in range(len(connect)):
            ax.plot(xs[j][connect[i]], ys[j][connect[i]], zs[j][connect[i]])

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
    ani.save('/home/azhuavlev/PycharmProjects/MotionMixerConv/conv_mixer/animation.gif', writer='pillow')

    # save animation as mp4
    ani.save('/home/azhuavlev/PycharmProjects/MotionMixerConv/conv_mixer/animation.mp4', writer='ffmpeg')

    plt.close()