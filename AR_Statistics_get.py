import sys
import numpy as np

# from sklearnex import patch_sklearn
# patch_sklearn()
import pandas as pd
import os
import time
import threading
import multiprocessing
import math

from AR_detection_model import *

thread = 3  # if you use multithread
process = None  # if you use multiprocess
# one fits for test
data_item = r"hmi.M_45s.20221112_231200_TAI.2.magnetogram.fits"
data_dir = r""  # data folder
butterfly_csv_write = "test.csv"
breakpoint_resume = False
arranged_shutdown = False
begin_item = "HMI20111231_235940_6173.fits"  # breakpoint resume
end_item = "HMI20100610_235956_6173.fits"  # arranged shutdown
save_fre = 30
save_pic = True
mode = "debug and plot"
"""
you can choose:
'debug and plot'
'get statistics multiprocess'
'get statistics multithread'
"""



def thread_worker(num, filenames):  # threading
    step = 1
    count = 0
    print(f"Thread{num}:Total={len(filenames)}\n")
    filenames = sorted(filenames)
    for name in filenames:
        _st = time.time()
        print(f"Thread{num}:{name}\nThread{num}:step {step}")
        data_name = data_dir + name
        sum_ar, list = get_clusters_info_list(data_item=data_name, name=name)
        final_list.extend(list)
        count += 1
        _et = time.time()
        print(f"run time:{_et-_st}\n")
        thread_save_fre = int(save_fre / thread)
        if count == thread_save_fre or step == len(filenames):
            print("save\n")
            list_array = np.array(final_list)
            dff = pd.DataFrame(list_array)
            dff.to_csv(butterfly_csv_write, mode="a", header=False)
            count = 0
            final_list.clear()
        step += 1
    return


def process_worker(name, step):  # processing
    _st = time.time()
    print(f"Process{os.getpid()}:{name}")
    print(f"step {step}")
    data_name = data_dir + name
    _, list = get_clusters_info_list(data_item=data_name, name=name)
    _et = time.time()
    print(f"run time:{_et-_st}\n")
    return list, step + 1


# get the clusters with points' absolute value


def get_the_clusters_pro(data_item):
    sum_ar, clusters_pro = get_the_clusters(data_item=data_item, save=save_pic)
    for cluster_index in range(sum_ar):
        for points_index in range(3):
            clusters_pro[cluster_index][points_index][:, -1] = np.abs(
                clusters_pro[cluster_index][points_index][:, -1]
            )
    return sum_ar, clusters_pro


# get the geometric center of latitude and the num of pixels of the clusters
def get_the_barycenter_size(points_pro, data_item_map):
    n = points_pro.shape[0]
    if n != 0:
        center_x = np.mean(points_pro[:, 0])
        center_y = np.mean(points_pro[:, 1])
        center_lat, center_lon = coordinate_transform(center_x, center_y, data_item_map)
        return center_x, center_y, center_lat, center_lon, n
    else:
        return None, None, None, None, 0


def get_the_fluxcenter_flux(points_pro, data_item_map):
    n = points_pro.shape[0]
    if n != 0:
        total_weight = np.sum(points_pro[:, 2])
        centroid_x = (
            np.sum(np.multiply(points_pro[:, 0], points_pro[:, 2])) / total_weight
        )
        centroid_y = (
            np.sum(np.multiply(points_pro[:, 1], points_pro[:, 2])) / total_weight
        )
        centroid_lat, centroid_lon = coordinate_transform(
            centroid_x, centroid_y, data_item_map
        )
        return centroid_x, centroid_y, centroid_lat, centroid_lon, total_weight
    else:
        return None, None, None, None, 0


def area_correcting(x, y, x0, y0, size):
    if (x is None) | (y is None):
        return 0
    else:

        r = 1912
        x = x - x0
        y = y - y0
        z = math.sqrt(r**2 - x**2 - y**2)
        cos = z / r
        size_corrected = size / (cos * 4 * 3.14159265 * 1950 * 1950) * 1000000

        return size_corrected


def get_clusters_info_list(data_item, name):
    data_item_map = sunpy.map.Map(data_item)
    sum_ar, clusters_pro = get_the_clusters_pro(data_item=data_item)
    if user == "crs":
        year = name[3:7]
        month = name[7:9]
        day = name[9:11]
    elif user == "lulu":
        year = name[10:14]
        month = name[14:16]
        day = name[16:18]
    list = []
    for cluster_pro in clusters_pro:
        stuff_list = [year, month, day]
        top = max(cluster_pro[0][:, 0])
        bottom = min(cluster_pro[0][:, 0])
        left = min(cluster_pro[0][:, 1])
        right = max(cluster_pro[0][:, 1])
        x0, y0 = get_center_from_fits(data_item)
        for points_pro in cluster_pro:
            center_x, center_y, center_lat, center_lon, n = get_the_barycenter_size(
                points_pro, data_item_map=data_item_map
            )
            size_corrected = area_correcting(center_x, center_y, x0, y0, n)
            centroid_x, centroid_y, centroid_lat, centroid_lon, total_weight = (
                get_the_fluxcenter_flux(points_pro, data_item_map=data_item_map)
            )
            flux_corrected = area_correcting(
                centroid_x, centroid_y, x0, y0, total_weight
            )
            stuff_list.extend(
                [center_x, center_y, center_lat, center_lon, n, size_corrected]
            )
            stuff_list.extend(
                [
                    centroid_x,
                    centroid_y,
                    centroid_lat,
                    centroid_lon,
                    total_weight,
                    flux_corrected,
                ]
            )
        top_lat, left_lon = coordinate_transform(top, left, data_item_map)
        bottom_lat, right_lon = coordinate_transform(bottom, right, data_item_map)
        stuff_list.extend(
            [top, bottom, left, right, top_lat, bottom_lat, left_lon, right_lon, x0, y0]
        )
        list.append(stuff_list)
    return sum_ar, list


if __name__ == "__main__":
    if mode == "debug and plot":
        start = time.time()
        pic = get_pic_from_fits(data_item=data_item)
        num_ar, clusters = get_the_clusters(data_item, print_info=True)
        end = time.time()
        print(f"run time:{end-start}")
        show_pos_and_neg(pic, num_ar, clusters, data_item)

    if mode == "get statistics multiprocess":
        df = pd.read_csv(
            butterfly_csv_write,
            header=None,
            names=[
                "number",
                "year",
                "month",
                "day",
                "center_x_all",
                "center_y_all",
                "center_lat_all",
                "center_lon_all",
                "num_pixels_all",
                "area_all",
                "centroid_x_all",
                "centroid_y_all",
                "centroid_lat_all",
                "centroid_lon_all",
                "total_weight_all",
                "flux_all",
                "center_x_pos",
                "center_y_pos",
                "center_lat_pos",
                "center_lon_pos",
                "num_pixels_pos",
                "area_pos",
                "centroid_x_pos",
                "centroid_y_pos",
                "centroid_lat_pos",
                "centroid_lon_pos",
                "total_weight_pos",
                "flux_pos",
                "center_x_neg",
                "center_y_neg",
                "center_lat_neg",
                "center_lon_neg",
                "num_pixels_neg",
                "area_neg",
                "centroid_x_neg",
                "centroid_y_neg",
                "centroid_lat_neg",
                "centroid_lon_neg",
                "total_weight_neg",
                "flux_neg",
                "top",
                "bottom",
                "left",
                "right",
                "top_lat",
                "bottom_lat",
                "left_lon",
                "right_lon",
                "x0",
                "y0",
            ],
        )
        df.to_csv(butterfly_csv_write, index=False)
        final_list = []
        filenames = os.listdir(data_dir)
        filenames = sorted(filenames)
        if breakpoint_resume:
            begin_index = filenames.index(begin_item)
            filenames = filenames[begin_index:]
        if arranged_shutdown:
            end_index = filenames.index(end_item)
            filenames = filenames[: end_index + 1]
        if process is not None:
            pool = multiprocessing.Pool(processes=process)
        else:
            pool = multiprocessing.Pool()
        step = 1
        for name in filenames:
            result = pool.apply_async(
                process_worker,
                (
                    name,
                    step,
                ),
            )
            list, step = result.get()
            final_list.extend(list)
            if step % save_fre == 0 or step == len(filenames):
                list_array = np.array(final_list)
                dff = pd.DataFrame(list_array)
                dff.to_csv(butterfly_csv_write, mode="a", header=False)
                print("save\n")
                final_list.clear()
        pool.close()
        pool.join()
        # print(final_list)

    if mode == "get statistics multithread":
        df = pd.read_csv(
            butterfly_csv_write,
            header=None,
            names=[
                "number",
                "year",
                "month",
                "day",
                "center_x_all",
                "center_y_all",
                "center_lat_all",
                "center_lon_all",
                "num_pixels_all",
                "area_all",
                "centroid_x_all",
                "centroid_y_all",
                "centroid_lat_all",
                "centroid_lon_all",
                "total_weight_all",
                "flux_all",
                "center_x_pos",
                "center_y_pos",
                "center_lat_pos",
                "center_lon_pos",
                "num_pixels_pos",
                "area_pos",
                "centroid_x_pos",
                "centroid_y_pos",
                "centroid_lat_pos",
                "centroid_lon_pos",
                "total_weight_pos",
                "flux_pos",
                "center_x_neg",
                "center_y_neg",
                "center_lat_neg",
                "center_lon_neg",
                "num_pixels_neg",
                "area_neg",
                "centroid_x_neg",
                "centroid_y_neg",
                "centroid_lat_neg",
                "centroid_lon_neg",
                "total_weight_neg",
                "flux_neg",
                "top",
                "bottom",
                "left",
                "right",
                "top_lat",
                "bottom_lat",
                "left_lon",
                "right_lon",
                "x0",
                "y0",
            ],
        )
        df.to_csv(butterfly_csv_write, index=False)
        final_list = []
        filenames = os.listdir(data_dir)
        filenames = sorted(filenames)
        if breakpoint_resume:
            begin_index = filenames.index(begin_item)
            filenames = filenames[begin_index:]
        if arranged_shutdown:
            end_index = filenames.index(end_item)
            filenames = filenames[: end_index + 1]
        for i in range(thread):
            thread_filenames = filenames[i::thread]
            t = threading.Thread(
                target=thread_worker,
                args=(
                    i,
                    thread_filenames,
                ),
            )
            t.start()
