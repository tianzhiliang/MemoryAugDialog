import sys
import math
import os
import time
import datetime

class StatRunTime():
    def __init__(self):
        self.clock_set = {}

    def get_time(self, landmark):
        if not landmark in self.clock_set:
            self.clock_set[landmark] = []
        try:
            self.clock_set[landmark].append(time.time_ns())
        except:
            self.clock_set[landmark].append(time.time()*1e9)

    def print_time(self, start_mark, end_mark):
        if not start_mark in self.clock_set:
            print("error. not start_mark in self.clock_set.", \
                    "start_mark:", start_mark)
            return ""
        if not end_mark in self.clock_set:
            print("error. not end_mark in self.clock_set.", \
                    "end_mark:", end_mark)
            return ""
        if len(self.clock_set[start_mark]) != len(self.clock_set[end_mark]):
            print("error. len of clock_set[start_mark] !=", \
                    "len of clock_set[end_mark]:", \
                    len(self.clock_set[start_mark]), "!=", \
                    len(self.clock_set[end_mark]))
            return ""

        time_diff = 0
        for s, e in zip(self.clock_set[start_mark], self.clock_set[end_mark]):
            time_diff += (e - s)
        time_sec = int(time_diff // 1e9)
        time_msec = int(time_diff % 1e9 // 1e6)
        time_usec = int(time_diff % 1e6 // 1e3)
        time_nsec = int(time_diff % 1e3)

        time = str(time_sec) + " s " + str(time_msec) + " ms " \
            + str(time_usec) + " us " + str(time_nsec) + " ns" \
            + " = " + str(time_diff / 1e9) + " s"
        print("Stat Run Time. From " + start_mark + " to " + end_mark \
            + " is: " + time)
        return time

    def clear_time_bufs(self, landmarks):
        for landmark in landmarks:
            self.clear_time_buf(landmark)

    def clear_time_buf(self, landmark):
        if not landmark in self.clock_set:
            print("error. not landmark in self.clock_set. landmark:", landmark)
        self.clock_set[landmark] = []

    def clear_all_bufs(self):
        self.clear_time_bufs(self.clock_set.keys())


def get_time():
    ts = time.time()
    now_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return str(now_time)

def print_time(mark):
    print(get_time() + "\t" + mark)

def test_cases():
    t1 = StatRunTime()
    t1.get_time("1")
    time.sleep(0.01)
    t1.get_time("2")
    time.sleep(0.02)
    t1.get_time("3")
    time.sleep(0.03)
    t1.get_time("4")
    for i in range(100):
        t1.get_time("5")
        time.sleep(0.01)
        t1.get_time("6")

    t1.print_time("1","2")
    t1.print_time("2","3")
    t1.print_time("3","4")
    t1.print_time("3","1")
    t1.print_time("5","6")
    t1.print_time("6","5")
    t1.clear_time_bufs(["5","1"])

    print("after clean 5 1")
    t1.print_time("1","2")
    t1.print_time("2","3")
    t1.print_time("3","4")
    t1.print_time("5","6")

    t1.clear_time_buf("6")
    for i in range(100):
        t1.get_time("5")
        time.sleep(0.01)
        t1.get_time("6")
    t1.print_time("5","6")

#test_cases()
