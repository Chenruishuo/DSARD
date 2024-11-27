import sys
import os
import re
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from sklearnex import patch_sklearn
# patch_sklearn()
from scipy.spatial import cKDTree
import sunpy.map
import astropy.units as u
import sklearn.cluster
import time
import math
from math import radians, sin, cos, sqrt, atan2

data_item = r"hmi.M_45s.20221112_231200_TAI.2.magnetogram.fits"
save_cls_pic = True


def get_pic_from_fits(data_item):
    source = fits.open(data_item)
    i = 0
    while not np.any(source[i].data):
        i += 1
    return np.array(source[i].data)


def get_center_from_fits(data_item):  # get the suncenter from fits
    source = fits.open(data_item)
    i = 0
    while not np.any(source[i].data):
        i += 1
    return source[i].header["CRPIX1"], source[i].header["CRPIX2"]

def corrected_area(points, x0, y0):
    if points is None:
        return 0
    else:

        r = 1912
        size=points.shape[0]
        x_mean, y_mean = np.mean(points[:, 0]), np.mean(points[:, 1])
        x = x_mean - x0
        y = y_mean - y0
        z = math.sqrt(r**2 - x**2 - y**2)
        cos = z / r
        size_corrected = size / (cos * 4 * 3.14159265 * 1950 * 1950) * 1000000 * 6.09

        return size_corrected

def threshold(pic, thre=0):  # get the pixels above threshold
    points = np.column_stack(np.where(np.abs(pic) > thre))
    pixel_values = pic[points[:, 0], points[:, 1]]
    points_with_value = np.concatenate([points, pixel_values[:, np.newaxis]], axis=1)
    return points_with_value


def tell_pos_and_neg(points):  # divide points into positive and negative
    pos_points = points[np.where(points[:, 2] > 0)]
    neg_points = points[np.where(points[:, 2] <= 0)]
    return pos_points, neg_points


def separate(clusters, i, cls):  # separate a cluster into several clusters by cls
    for j in range(max(cls) + 1):
        ith = clusters[i][0][cls == j]
        pos, neg = tell_pos_and_neg(ith)
        clusters.append([ith, pos, neg])
    clusters.pop(i)


def merge(clusters, i, j):  # merge ith and jth cluster
    for k in range(3):
        if not np.any(clusters[i][k]):
            clusters[i][k] = clusters[j][k]
        else:
            clusters[i][k] = np.vstack((clusters[i][k], clusters[j][k]))
    clusters.pop(j)


def small_or_singlepole(cluster, min_size, ratio, x0, y0):  # tell the type of the cluster
    minratio, maxratio = sorted([ratio, 1 / ratio])
    if (
        not (cluster[2].shape[0] > 0)
        or cluster[1].shape[0] / (cluster[2].shape[0]) > maxratio
    ):
        return 1
    if (
        not (cluster[2].shape[0] > 0)
        or cluster[1].shape[0] / (cluster[2].shape[0]) < minratio
    ):
        return 2
    if (
        corrected_area(cluster[0],x0,y0) < min_size
    ):
        return 3
    return 4


def separate_large_cluster(data_item, clusters, max_size, eps, min_samples, min_ar):
    x0, y0 = get_center_from_fits(data_item)
    sec_dbs = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
    if len(clusters) > min_ar:
        i = 0
        while True:
            if corrected_area(clusters[i][0],x0,y0) > max_size:
                sec_dbs.fit(clusters[i][0][:, :-1])
                if max(sec_dbs.labels_) == 0:
                    i += 1
                else:
                    separate(clusters, i, sec_dbs.labels_)
            else:
                i += 1
            if i == len(clusters):
                break
    return len(clusters)


def coordinate_transform(x, y, data_item_map):  # transform pixel to lat and lon
    helioproj_coord = data_item_map.pixel_to_world(y * u.pix, x * u.pix)
    heliographic_coord = helioproj_coord.transform_to("heliographic_stonyhurst")
    latitude = heliographic_coord.lat.to_value(unit=u.deg)
    longitude = heliographic_coord.lon.to_value(unit=u.deg)
    return latitude, longitude


# def dis_points(point_set1, point_set2):  # define the distance between two points sets
#     point_set2, point_set1 = sorted(
#         [point_set1, point_set2], key=lambda s: len(s))
#     tree = cKDTree(point_set1[:, :-1])
#     distances, _ = tree.query(point_set2[:, :-1], k=1)  # k=1表示只查找最近的一个邻居
#     # calculate the minimum distance between two point sets
#     min_distance = np.min(distances)
#     return min_distance


def spherical_distance(lat1, lon1, lat2, lon2):
    R = 696340
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # Distance in kilometers
    distance = R * c
    return distance


def dis_points(point_set1, point_set2, data_item_map, modify):
    point_set2, point_set1 = sorted([point_set1, point_set2], key=lambda s: len(s))
    tree = cKDTree(point_set1[:, :-1])
    distances, indices = tree.query(point_set2[:, :-1], k=1)
    if modify:
        min_index = np.argmin(distances)
        closest_point_from_set2 = point_set2[min_index]
        closest_point_from_set1 = point_set1[indices[min_index]]
        # start_time = time.time()
        lat1, lon1 = coordinate_transform(
            closest_point_from_set2[0], closest_point_from_set2[1], data_item_map
        )
        lat2, lon2 = coordinate_transform(
            closest_point_from_set1[0], closest_point_from_set1[1], data_item_map
        )
        # end_time_1 = time.time()
        sph_distance = spherical_distance(lat1, lon1, lat2, lon2)
        # end_time_2 = time.time()
        # print(end_time_1-start_time, end_time_2-end_time_1)
        return sph_distance
    else:
        return np.min(distances)


# define distance between two clusters
def distance(cluster1, cluster2, occ, ratio, min_size, data_item_map, modify, x0, y0):
    if occ == 1:
        if small_or_singlepole(cluster2, ratio=ratio, min_size=min_size,x0=x0,y0=y0) != 2:
            return dis_points(cluster1[1], cluster2[1], data_item_map, modify)
        else:
            return dis_points(cluster1[1], cluster2[2], data_item_map, modify)
    if occ == 2:
        if small_or_singlepole(cluster2, ratio=ratio, min_size=min_size,x0=x0,y0=y0) != 1:
            return dis_points(cluster1[2], cluster2[2], data_item_map, modify)
        else:
            return dis_points(cluster1[2], cluster2[1], data_item_map, modify)
    if occ == 3 or 4:
        return dis_points(cluster1[0], cluster2[0], data_item_map, modify)


def merge_bad(
    clusters, ratio, min_size, dimi_dis, min_ar, data_item, modify
):  # merge bad clusters
    i = 0
    data_item_map = sunpy.map.Map(data_item)
    x0,y0 = get_center_from_fits(data_item)
    while len(clusters) > min_ar:
        i_kind = small_or_singlepole(clusters[i], ratio=ratio, min_size=min_size, x0=x0, y0=y0)
        dis_set = np.array(
            [
                distance(
                    clusters[i],
                    clusters[j],
                    i_kind,
                    ratio=ratio,
                    min_size=min_size,
                    data_item_map=data_item_map,
                    modify=modify,
                    x0=x0,
                    y0=y0
                )
                for j in range(len(clusters))
                if j != i
            ]
        )
        if not len(dis_set):  # if there remains only one cluster
            if i_kind == 4:  # and it is good
                # print(i,k)
                return len(clusters)
            # if it is unipolar and not so small
            elif i_kind < 3 and corrected_area(clusters[i][0],x0,y0) > 4 * min_size:
                # print(i,k,'only one')
                return len(clusters)
            else:  # else
                clusters.pop(i)
                # print(i,k,'kill!')
                return len(clusters)
        neibour = np.argmin(dis_set)  # find the nearest cluster
        min_dis = np.min(dis_set)
        if neibour >= i:
            neibour += 1  # because we skip cluster[i] itself in dis_set
        nei_kind = small_or_singlepole(
            clusters[neibour], ratio=ratio, min_size=min_size, x0=x0, y0=y0
        )
        if i_kind < 3:  # if cluster i is unipolar
            if nei_kind != i_kind:
                if nei_kind + i_kind == 3:
                    # if neibour is heteropolar, consider distance between them and loosen restriction
                    if min_dis < 2 * dimi_dis:
                        merge(clusters, neibour, i)
                        # print(i,k,neibour)
                    elif corrected_area(clusters[i][0],x0,y0) > 4 * min_size:
                        # if not so near, then consider the size of this unipolar cluster
                        # print(i,k)
                        i += 1
                    else:
                        clusters.pop(i)
                        # print(i,k,"kill!")
                else:
                    # if neibour is not heteropolar
                    if min_dis < dimi_dis:
                        merge(clusters, neibour, i)
                        # print(i,k,neibour)
                    else:
                        clusters.pop(i)
                        # print(i,k,"kill!")
            else:
                # if neibour is homopolar, loosen restricion
                if min_dis < 2 * dimi_dis:
                    merge(clusters, neibour, i)
                    # print(i,k,neibour)
                else:
                    clusters.pop(i)
                    # print(i,k,"kill!")
        elif i_kind == 3:
            # if cluster i is too small
            if min_dis < dimi_dis:
                merge(clusters, neibour, i)
                # print(i,k,neibour)
            else:
                clusters.pop(i)
                # print(i,k,"kill!")
        else:
            # if cluster i is good
            minratio, maxratio = sorted([ratio, 1 / ratio])
            if nei_kind == 4:
                # if neibour is good too
                if (
                    clusters[i][0].shape[0] / clusters[neibour][0].shape[0] > maxratio
                    or clusters[i][0].shape[0] / clusters[neibour][0].shape[0]
                    < minratio
                ) and min_dis < dimi_dis:
                    # if neibour is near cluster i and they're not too far apart in size
                    # (shape[0]!=0)
                    merge(clusters, neibour, i)
                    # print(i,k,neibour)
                else:
                    # print(i,4)
                    i += 1
            else:
                # if neibour is not good
                # print(i,4)
                i += 1
        if i == len(clusters):
            break
    return len(clusters)


def kill_the_circle(points, data_item, r, yes_or_no=True):
    if yes_or_no == False:
        return points, 1
    else:
        points_inside_circle = []
        suncenter_x = get_center_from_fits(data_item)[0]
        suncenter_y = get_center_from_fits(data_item)[1]
        points_inside_circle = points[
            np.where(
                (points[:, 0] - suncenter_x) ** 2 + (points[:, 1] - suncenter_y) ** 2
                <= np.power(r, 2)
            )
        ]
        return points_inside_circle, 0


# we do not need clusters whose centers are in the outer ring


def kill_the_annulus(clusters, r, data_item):
    if len(clusters) > 0:
        suncenter_x = get_center_from_fits(data_item)[0]
        suncenter_y = get_center_from_fits(data_item)[1]
        num_ar = len(clusters)
        mask = [
            (np.mean(clusters[i][0][:, 0]) - suncenter_x) ** 2
            + (np.mean(clusters[i][0][:, 1]) - suncenter_y) ** 2
            <= r**2
            for i in range(num_ar)
        ]
        new_clusters = [x for (x, m) in zip(clusters, mask) if m]
    else:
        new_clusters = clusters
    return len(new_clusters), new_clusters


def get_the_clusters(
    data_item, save=False, print_info=False, r=None, modify=True
):  # get the clusters in a pic
    if save:
        filename = os.path.basename(data_item)
        regex_pattern = r"(\d{8})"
        match = re.search(regex_pattern, filename)
        if match:
            date = match.group(1)
            formatted_dir_name = f"HMI{date[:8]}"
            save_dir = formatted_dir_name
        else:
            print(f"Cannot find the date in the filename! {filename}")
            save_dir = "HMI"
    pic = get_pic_from_fits(data_item=data_item)
    points = threshold(thre=150, pic=pic)
    points, min_ar = kill_the_circle(points=points, data_item=data_item, r=1732)
    if save:
        save_threshold_pic(pic, points, save_dir)
    dbs = sklearn.cluster.DBSCAN(eps=30, min_samples=200)
    dbs.fit(points[:, :-1])
    num_ar = max(dbs.labels_) + 1
    if num_ar > 0:
        clusters = []
        for i in range(num_ar):
            ith = points[dbs.labels_ == i]
            pos, neg = tell_pos_and_neg(ith)
            clusters.append([ith, pos, neg])
    else:
        clusters = []
    if save:
        save_clusters_pic(pic, num_ar, clusters, "after_dbscan", data_item, save_dir)
    num_ar = separate_large_cluster(
        data_item, clusters, max_size=3000, eps=30, min_samples=500, min_ar=min_ar
    )
    if save:
        save_clusters_pic(pic, num_ar, clusters, "after_separate", data_item, save_dir)
    num_ar = merge_bad(
        clusters,
        ratio=10,
        min_size=75,
        dimi_dis=120 if not modify else 50000,
        min_ar=min_ar,
        data_item=data_item,
        modify=modify,
    )
    if save:
        save_clusters_pic(pic, num_ar, clusters, "after_merge", data_item, save_dir, label=True)
    if r is not None:
        num_ar, clusters = kill_the_annulus(clusters, r, data_item)
    if print_info:
        print(f"num_ar={num_ar-min_ar}")
        for i in range(num_ar):
            print(f"cluster {i+1}:", clusters[i][1].shape[0], clusters[i][2].shape[0])
    return num_ar - min_ar, clusters


# plot the pic and clusters with bounding box
def show_pos_and_neg(pic, num_ar, clusters, data_item, r=None, label=False):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.imshow(pic, cmap="Greys")
    for i in range(num_ar):
        if np.any(clusters[i][1]):
            ax.scatter(
                clusters[i][1][:, 1],
                clusters[i][1][:, 0],
                c="white",
                alpha=0.5,
                s=1,
                marker="o",
            )
        if np.any(clusters[i][2]):
            ax.scatter(
                clusters[i][2][:, 1],
                clusters[i][2][:, 0],
                c="black",
                alpha=0.5,
                s=1,
                marker="o",
            )
        top = max(clusters[i][0][:, 0])
        bottom = min(clusters[i][0][:, 0])
        left = min(clusters[i][0][:, 1])
        right = max(clusters[i][0][:, 1])
        rect = patches.Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            linewidth=1.5,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        if label:
            ax.annotate(
                f"{i+1}",
                ((left + right) / 2, (top + bottom) / 2),
                color="yellow",
                # bbox=dict(
                #     facecolor="none",
                #     edgecolor="yellow",
                #     boxstyle="circle,pad=0.1",
                #     linewidth=1,
                # ),
                fontsize=15,
                ha="center",
                va="center",
            )
        if r is not None:
            suncenter_x = get_center_from_fits(data_item)[0]
            suncenter_y = get_center_from_fits(data_item)[1]
            circle = patches.Circle(
                (suncenter_x, suncenter_y),
                r,
                color="lightgreen",
                linestyle="--",
                fill=False,
            )
            ax.add_patch(circle)
    plt.gca().invert_xaxis()
    plt.show()


# save pictures
def save_threshold_pic(pic, points_with_value, save_dir):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.imshow(pic, cmap="Greys")
    positive_points = points_with_value[points_with_value[:, 2] > 0]
    negative_points = points_with_value[points_with_value[:, 2] < 0]
    if len(positive_points) > 0:
        ax.scatter(
            positive_points[:, 1],
            positive_points[:, 0],
            c="white",
            alpha=0.5,
            s=1,
            marker="o",
        )
    if len(negative_points) > 0:
        ax.scatter(
            negative_points[:, 1],
            negative_points[:, 0],
            c="black",
            alpha=0.5,
            s=1,
            marker="o",
        )
    plt.gca().invert_xaxis()
    save_path = f"pictures/{save_dir}/after_threshold.png"
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, dpi=300)
    plt.close()


# save pictures
def save_clusters_pic(
    pic, num_ar, clusters, pic_name, data_item, save_dir, r=None, label=False
):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.imshow(pic, cmap="Greys")
    for i in range(num_ar):
        if np.any(clusters[i][1]):
            ax.scatter(
                clusters[i][1][:, 1],
                clusters[i][1][:, 0],
                c="white",
                alpha=0.5,
                s=1,
                marker="o",
            )
        if np.any(clusters[i][2]):
            ax.scatter(
                clusters[i][2][:, 1],
                clusters[i][2][:, 0],
                c="black",
                alpha=0.5,
                s=1,
                marker="o",
            )
        top = max(clusters[i][0][:, 0])
        bottom = min(clusters[i][0][:, 0])
        left = min(clusters[i][0][:, 1])
        right = max(clusters[i][0][:, 1])
        rect = patches.Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            linewidth=1.5,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        if label:
            if i==7:
                ax.annotate(
                    f"{i+1}",
                    (right-25, top+70),
                    color="yellow",
                    # bbox=dict(
                    #     facecolor="none",
                    #     edgecolor="yellow",
                    #     boxstyle="circle,pad=0.1",
                    #     linewidth=1,
                    # ),
                    fontsize=30,
                    ha="center",
                    va="center",
                )
            else:
                ax.annotate(
                    f"{i+1}",
                    (right-25, bottom-45),
                    color="yellow",
                    # bbox=dict(
                    #     facecolor="none",
                    #     edgecolor="yellow",
                    #     boxstyle="circle,pad=0.1",
                    #     linewidth=1,
                    # ),
                    fontsize=30,
                    ha="center",
                    va="center",
                )
        if r is not None:
            suncenter_x = get_center_from_fits(data_item)[0]
            suncenter_y = get_center_from_fits(data_item)[1]
            circle = patches.Circle(
                (suncenter_x, suncenter_y),
                r,
                color="lightgreen",
                linestyle="--",
                fill=False,
            )
            ax.add_patch(circle)
    plt.gca().invert_xaxis()
    save_path = f"pictures/{save_dir}/{pic_name}.png"
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    start = time.time()
    pic = get_pic_from_fits(data_item=data_item)
    num_ar, clusters = get_the_clusters(
        data_item,
        save=save_cls_pic,
        print_info=True,
        modify=True,
    )
    end = time.time()
    print(f"run time:{end-start}")
    show_pos_and_neg(pic, num_ar, clusters, data_item)
