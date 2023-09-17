# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime


def extract_data_csv(path_csv, sensor_id):
    with open(path_csv) as f:
        line  = f.readline().strip()
        while line:
            if line.find(sensor_id) != -1:
                return line
            line = f.readline().strip()
    return None


def extract_all_sensor_data(sensor_id):
    base_path = "/home/yohan/Documents/dataset/HistoricalData/"
    list_folder = ["20230612","20230628","20230710","20230718","20230728","20230802","20230810","20230818","20230862"]

    list_data = []
    for folder in list_folder:
        print(f"{folder}")
        curr = f"{base_path}{folder}/"
        csvfiles = [f for f in listdir(curr) if isfile(join(curr, f))]
        csvfiles.sort()
        for file in csvfiles:
            datapoint = extract_data_csv(curr+file, sensor_id)
            if datapoint:
                list_data.append(datapoint)
            # if len(list_data)> 10:
            #    break
        #break

    with open(base_path+sensor_id+".csv", "w") as f:
        f.write("MSR_Id,TimeStamp,CarFlow,LorryFlow,AnyFlow,CarSpeed,LorrySpeed,AnySpeed\n")
        for data in list_data:
            f.write(f"{data}\n")


def interpolate_data():
    base_path = "/home/yohan/Documents/dataset/HistoricalData/CH:0056.05_interpolated.csv"
    out_path = "/home/yohan/Documents/dataset/HistoricalData/CH:0056.05_interpolated_v2.csv"
    # df = pd.read_csv(base_path)
    # df = df.interpolate()
    # df.to_csv("/home/yohan/Documents/dataset/HistoricalData/CH:0342.01_interpolated.csv")
    # return

    with open(out_path, "w") as fout:
        with open(base_path) as f:
            line = f.readline().strip()
            fout.write("MSR_Id,TimeStamp,CarFlow,LorryFlow,AnyFlow,CarSpeed,LorrySpeed,AnySpeed\n")
            line = f.readline().strip()
            while line:

                id, ts, car_count, lory_count, other_count, car_speed, lory_speed, other_speed = line.split(",")
                if float(lory_count) < 1:
                    lory_speed = 0
                fout.write(f"{id},{ts},{car_count},{lory_count},{other_count},{car_speed},{lory_speed},{other_speed}\n")
                line = f.readline().strip()



def show_data():
    base_path = "/home/yohan/Documents/dataset/HistoricalData/CH:0056.05.csv"
    number, speed = [], []
    with open(base_path) as f:
        line = f.readline().strip()
        line = f.readline().strip()
        while line:
            data = line.split(",")
            if len(data[2]) > 0:
                number.append(int(data[2]))
            else:
                number.append(0)
            if len(data[5]) > 0:
                speed.append(float(data[5]))
            else:
                speed.append(0)
            line = f.readline().strip()
            if len(speed) > 1000:
                break
    plt.plot(number, color='r', label='Number of cars')
    plt.plot(speed, color='g', label='Car speed')

    # To load the display window
    plt.show()

def prepare_csv_ready():
    # We take a csv file, and for each data point, we create its full data ready to feed the NN
    # The file will be much bigger, but no need to compute for all points
    csv_in = "/home/yohan/Documents/dataset/HistoricalData/CH:0056.05_interpolated_v2.csv"
    csv_out = "/home/yohan/Documents/dataset/HistoricalData/CH:0056.05_nn_ready.csv"

    csv_in = "/home/yohan/Documents/dataset/HistoricalData/CH:0342.01_interpolated_v2.csv"
    csv_out = "/home/yohan/Documents/dataset/HistoricalData/CH:0342.01_nn_ready.csv"

    date_format = '%Y-%m-%dT%H:%M:%S.000000Z'
    trafic = pd.read_csv(csv_in)
    with open(csv_out, "w") as f:
        for i in range(len(trafic)-500):
            if i % 1000 == 999:
                print(f"{i+1}/{len(trafic)}")
            inputs = []
            for j in range(180):
                # The timestamp is an issue, the number is way to big now
                date = datetime.strptime(trafic.loc[i + j, "TimeStamp"], date_format)
                inputs.append(float(date.weekday()) / 6.0)
                inputs.append(float(date.hour) / 23.0)
                inputs.append(float(date.minute) / 60.0)
                inputs.append(float(trafic.loc[i + j, "CarFlow"]) / 1500.0)
                inputs.append(float(trafic.loc[i + j, "CarSpeed"]) / 130.0)
            for shift in [25, 85, 145]:
                mean_speed, mean_car = 0, 0
                for j in range(11):
                    mean_speed += float(trafic.loc[i + j + shift, "CarSpeed"]) / 130.0
                    mean_car += float(trafic.loc[i + j + shift, "CarFlow"]) / 1500.0
                mean_speed /= 11.0
                mean_car /= 11.0
                inputs.append(mean_speed)
                inputs.append(mean_car)
            out_line = ""
            for k in inputs:
                out_line += f"{k:.5f},"
            f.write(f"{out_line[:-1]}\n")
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # extract_all_sensor_data("CH:0056.05")
    # extract_all_sensor_data("CH:0342.01")
    # show_data()
    # interpolate_data()
    prepare_csv_ready()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
