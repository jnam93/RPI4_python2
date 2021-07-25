#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import RPi.GPIO as GPIO
import os
import numpy as np

from numpy import savetxt,save

import csv

import math
import scipy
from scipy import io
import scipy.signal

from sklearn.preprocessing  import StandardScaler
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm

import serial
import time
import threading
#import signal
import logging
logging.basicConfig()
from apscheduler.schedulers.background import BackgroundScheduler
import apscheduler.schedulers.blocking #import BlockingScheduler

Time = 0  #   Time (time slice of fft)

SAMPLES = 128
PERLINE = 16
line = []   #   dump list for serial read char

exitThread = False
time_t=time.strftime("%y%m%d_%H%M%S", time.localtime(time.time()))
work_list=["Bicyclecrunch","Highkick","Highkneejack","Hollowrock","Legraise","Lunge","Sidelateralraise","Situp","Squat_ordinary","Xception"]
num = None







def work_spec():
    global Time
    global work_list
    global num
    print("work_list:\n0:Bicyclecrunch\n1:Highkick\n2:Highkneejack\n3:Hollowrock\n4:Legraise\n5:Lunge\n6:Sidelateralraise\n7:Situp\n8:Squat_ordinary\n9:Xception")
    while True:
        try:
            num = int(input("Enter 0 ~ 9: "))
            print("work: %s"%work_list[num]+"[%d]"%num)  
            break
            
        except:
            print("error")
            sys.stdout.write("\033[2F")
            continue 
     
    while True:
        try:
            Time = int(input("Enter Ticks: "))
            print("Ticks: %d"%Time)  
            Time = Time
            break
            
        except:
            print("error")
            sys.stdout.write("\033[2F")
            continue      
    


def start():
    print("---start---")
    t = threading.Thread(target=readThread, args=[ser])
    t.do_run     = True
    t.start()
    return t
'''
def run(arg):
    print("run")
    t = threading.currentThread()
    while getattr(t,'do_run',True):
        print("---%s Running---"% arg)
        time.sleep(1)
'''

def stop(t):
	
	t.do_run = False
	t.join()
	print("---stop---")
	sched.remove_job('job1')
'''
sched = BackgroundScheduler()
sched.start()

t = start()
sched.add_job(stop,'interval',seconds = 10,args =(t,), id = 'job1')
'''

'''
def handler(signum, frame):
    exitThread = True
'''
def parsing_data(data):
    
    tmp =''.join(data)
    return tmp

def read_line(ser):
    
    #while getattr(t,'do_run',True):
    #    time.sleep(1)
    global line
    line_I = []
    line_Q = []
    global exitThread
    global Time
    count = 0
    i=0
    j=0
    #print(ser.read())
    print('\n')
    while count < Time+1: 
        #print('while',i)   #   debug line
        #i=i+1
        for c in ser.read():    #   serial read for each 8bit 
            #print('for',j) #   debug line
            #j=j+1
            #if t.do_run != True:
            #    time.sleep(10)
            line.append(str(c)) #   list of 'char'
            line_s=''.join(line)    #   list to whole string
            #line_l=line_s.split()   #   split string to list of 'char'
            #print(line_s)
            
            if line_s.count("I")==2 and line_s.count("Q") == 1:  #   I,Q extract
                if count == 0:
                    del line[:-1]
                    
                else:
                    line_l=line_s.split()   #   split string to list of 'char'
                    line_a=np.array(line_l)
                    
                    [I_1st,I_2nd]=np.where(line_a=='I')[0]
                    [Q_1st]=np.where(line_a=='Q')[0]
                    #print('I_1st:',I_1st,'I_2nd:',I_2nd,'Q_1st:',Q_1st)
                    dump_I=line_a[I_1st+4:I_1st+132]
                    dump_Q=line_a[Q_1st+4:Q_1st+132]
                    
                    dump_I=np.array(dump_I)
                    dump_Q=np.array(dump_Q)
                    dump_I=dump_I.astype(np.float)   #   str to int
                    dump_Q=dump_Q.astype(np.float)  #could not convert string to float: -------------

                    #print('I:',dump_I)
                    #print('Q:',dump_Q)  
                    mean_I = np.mean(dump_I)      #   DC off
                    mean_Q = np.mean(dump_Q)
                    dump_I = dump_I - mean_I
                    dump_Q = dump_Q - mean_Q
                    line_I[count*128:(count+1)*128]=dump_I
                    line_Q[count*128:(count+1)*128]=dump_Q
                    
                    #print("I:",len(dump_I))
                    #print(dump_I)
                    #print("Q:",len(dump_Q))
                    #print(dump_Q)
                    sys.stdout.write("\033[2F")
                    print('\nTick:%d'%count)
                    
                   
                    del line[:-1]#
                count = count +1    
                
                
                
            elif line_s.count('I')==2 and line_s.count('Q') == 2:
                
                
                line_l=line_s.split()   #   split string to list of 'char'
                line_a=np.array(line_l)
                
                [I_1st,I_2nd]=np.where(line_a=='I')[0]
                [Q_1st,Q_2nd]=np.where(line_a=='Q')[0]
                #print('I_1st:',I_1st,'I_2nd:',I_2nd,'Q_1st:',Q_1st,'Q_2nd:',Q_2nd)
                
                dump_I=line_a[I_1st+4:Q_2nd-1]
                dump_Q=line_a[Q_2nd+4:I_2nd-1]
               
                dump_I=np.array(dump_I)
                dump_Q=np.array(dump_Q)
                dump_I=dump_I.astype(np.float)   #   str to int
                dump_Q=dump_Q.astype(np.float)
                #print('I:',dump_I)
                #print('Q:',dump_Q)  
                mean_I = np.mean(dump_I)      #   DC off
                mean_Q = np.mean(dump_Q)
                dump_I = dump_I - mean_I
                dump_Q = dump_Q - mean_Q
                line_I[count*128:(count+1)*128]=dump_I
                line_Q[count*128:(count+1)*128]=dump_Q
                
                #print("I:",len(dump_I))
                #print(dump_I)
                #print("Q:",len(dump_Q))
                #print(dump_Q)
                sys.stdout.write("\033[2F")
                print('\nTick:%d'%count)
                
                del line[:-1]#
                
                count = count +1
              
    
    return [line_I,line_Q]  
                    

def save_raw_data(line_I,line_Q):
    
    #raw_data =[]
    #raw_data = np.array(raw_data,dtype=complex)
    raw_data = line_I[:] + np.multiply(line_Q[:],1j)
    print('raw:')
    print(raw_data.shape[0])
    
    return raw_data
    
def stft_plt(input):
    global time_t
    global num
    global work_list
    plt.figure(2)
    stft, f, t, im =plt.specgram(input, NFFT=128, Fs=3000, detrend=None, window=np.hamming(128), noverlap=0, xextent=None, pad_to=None, sides='twosided', scale_by_freq=None, mode='default', scale='dB')
    
    
   
    
    savetxt("/home/pi/capstone/data/csv/stft"+"/%s"%work_list[num]+"/Nam_stft_full_%s"%work_list[num]+"_%s"%time_t +".csv",stft,delimiter=',')
    save("/home/pi/capstone/data/npy/stft"+"/%s"%work_list[num]+"/Nam_stft_full_%s"%work_list[num]+"_%s"%time_t +".npy",stft)
    stft_abs=np.abs(stft)
    stft_10log10 = 10*np.log10(stft_abs+0.0001)
    plt.figure(3)
    plt.pcolormesh(stft_10log10)
    return stft
                              
def stft_scipy(raw_data):
    [f, t, stft_data]=scipy.signal.spectrogram(
        raw_data,fs=3000,
        window='hamming',
        nperseg=128,
        noverlap=0,
        nfft=128,
        detrend=False,
        return_onesided=False,
        axis=-1,
        mode='complex'
        )
    abs_stft=np.abs(stft_data)
    
    print('stft_size:',abs_stft.shape)
    '''plt.pcolor(t, f, abs_stft,
        norm=colors.LogNorm(vmin=np.abs(stft_data.min()),vmax=np.abs(stft_data.max())),
        cmap='viridis')
       ''' 
        
    plt.pcolormesh(t, f, np.log10(abs_stft),
        vmin=math.log10(abs_stft.min()),vmax=math.log10(abs_stft.max()),
        cmap='viridis')      
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time')
    plt.show()
    return abs_stft
    
def roi_kcy1(stft_data):
    global Time
    #feature extract
    stft_abs = np.abs(stft_data)
    stft_abs_2 = np.square(stft_abs)
    
    col_sum = np.sum(stft_abs_2,axis = 0)    # column sum
    cum_sum = np.cumsum(col_sum)    # cumulative   
    #print("col_sum",col_sum)#
    #print("cumsum",cum_sum)#
    halfcol = stft_abs_2[0:65,:]
    halfcol_sum = np.sum(stft_abs_2)
    
    halfcol_cum_sum = np.cumsum(halfcol_sum)
    x_cum_sum = np.arange(Time)
    fig_bar = plt.figure(1)
    plt.bar(x_cum_sum,cum_sum[0:Time])
    
    detect_false = 0
    detect_flat = 0
#detect flat & false
    
    j = np.ones((Time))
    #print("j",j.shape)#
    
    for i in range(1,Time):
        detect_flat = cum_sum[i] - cum_sum[i-1]
        if detect_flat < 12500:###
            j[i] = 0
            #print("detect_flat:%d"%i)
            
            
    for iii in range(0,Time):
        if j[iii] == 0:
            detect_false = 1 + detect_false
            #print("detect_false:%d"%iii)
        else :
            if detect_false < 8:###
                for iiii in range(1,detect_false+1):
                    j[iii-iiii]=1   
            detect_false = 0
            
    
#our_roi = np.multiply(stft_data,1j)  

# cut & count
    detect_ones = 0
    count = 0
    index_start =[]
    index_end = []
    stft_data_cut=np.zeros((20,20,Time // 20))
    for cut in range(0,Time):##
        if j[cut] == 1:
            detect_ones = 1 + detect_ones
            #print("detect_ones:",detect_ones)#
        else :
            if detect_ones > 6:#6
                
                mid_index = cut - np.round(detect_ones/2)
                print("mid",mid_index)
                start_index = mid_index - 9
                end_index = mid_index + 10
                if mid_index < 10:
                    start_index = 0
                if mid_index > Time-9:
                    end_index = Time-1
                stft_data_cut[:,:end_index-start_index+1,count] = stft_data[54:74,start_index:end_index+1] ##
                count = count + 1
                index_start.append(start_index)
                index_end.append(end_index)
                
            detect_ones = 0
         
    
    #print("stft_data_cut",stft_data_cut)   
    print("start:",index_start)
    print("end:",index_end)
    return stft_data_cut , count , index_start, index_end
    
    
def plot_data(bnzd_in , count , index_start, index_end):
    global time_t
    global num
    global work_list
   
    
    stft_10log10 = 10*np.log10(np.abs(bnzd_in)+0.0001)
    print("stft_data_cut",stft_data_cut.shape)
    print("count",count)   
    fig = plt.figure(4,figsize=(10,5))
    for i in range(0,count):
        
        ax =fig.add_subplot(count // 2+1,2,i+1)
        plt.pcolormesh(stft_10log10[:,:,i])
        ax.set_title("[%d"%index_start[i]+":%d"%index_end[i]+"]")
        savetxt("/home/pi/capstone/data/csv/cut"+"/%s"%work_list[num]+"/Nam_stft_cut_%s"%work_list[num]+"%s"%time_t +"[%d]"%i+".csv",bnzd_in[:,:,i],delimiter=',')   #csv
        save("/home/pi/capstone/data/npy/cut"+"/%s"%work_list[num]+"/Nam_stft_cut_%s"%work_list[num] + "%s"%time_t +"[%d]"%i+".npy",bnzd_in[:,:,i])    #npy
    plt.show()
       
    


         
def scale_binarize(stft_data_cut,count):
    stft_abs=np.abs(stft_data_cut)
    scaler = StandardScaler()
    stft_data_cut_scaled =np.zeros((20,20,count))
    for i in range(0,count):
        stft_data_cut_scaled[:,:,i] = scaler.fit_transform(stft_abs[:,:,i])    
    
    stft_data_cut_scaled=stft_data_cut_scaled > 0
    bnzd_in = stft_data_cut_scaled.astype(int)
    print(bnzd_in.shape)      
    return bnzd_in
    
    
def binarize(stft_data_cut_scaled):
    
    bnzd_in=stft_data_cut_scaled > 0
         
    return bnzd_in

def gpio_init():
    GPIO.setwarnings(False)
    out_port = [8,10,11,12,13,15,16,18,19,21,22,23,24,26,29,31,32,33,35,36,37]  
    in_port = [3,5]
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(in_port, GPIO.IN) # 0 -> 0, 1-> 1
    GPIO.setup(out_port, GPIO.OUT, initial=GPIO.LOW) 
    
    
    return out_port,in_port

def gpio_fin():
    GPIO.cleanup()
    return 0

    
def tx_data(out_port,bnzd_in,count):
    print('TX')
   
    for j in range(0,count):
        for i in range(0,20):
            time.sleep(0.000001)
            GPIO.output(out_port[0],bnzd_in[i,0,j])
            GPIO.output(out_port[1],bnzd_in[i,1,j])
            GPIO.output(out_port[2],bnzd_in[i,2,j])
            GPIO.output(out_port[3],bnzd_in[i,3,j])
            GPIO.output(out_port[4],bnzd_in[i,4,j])
            GPIO.output(out_port[5],bnzd_in[i,5,j])
            GPIO.output(out_port[6],bnzd_in[i,6,j])
            GPIO.output(out_port[7],bnzd_in[i,7,j])
            GPIO.output(out_port[8],bnzd_in[i,8,j])
            GPIO.output(out_port[9],bnzd_in[i,9,j])
            GPIO.output(out_port[10],bnzd_in[i,10,j])
            GPIO.output(out_port[11],bnzd_in[i,11,j])
            GPIO.output(out_port[12],bnzd_in[i,12,j])
            GPIO.output(out_port[13],bnzd_in[i,13,j])
            GPIO.output(out_port[14],bnzd_in[i,14,j])
            GPIO.output(out_port[15],bnzd_in[i,15,j])
            GPIO.output(out_port[16],bnzd_in[i,16,j])
            GPIO.output(out_port[17],bnzd_in[i,17,j])
            GPIO.output(out_port[18],bnzd_in[i,18,j])
            GPIO.output(out_port[19],bnzd_in[i,19,j])
            print("i:",i,"j",j)
            time.sleep(0.000001)
            
            GPIO.output(out_port[20],GPIO.HIGH)
            time.sleep(0.000001)#clock time 20ns : 50MHz
            GPIO.output(out_port[20],GPIO.LOW)
            
    return 0    

def rx_data():
    
    return 0
    
def softmax():
    return 0    
    
def file_read():
    
    path = "/home/pi/capstone/data/npy/cut/"+"%s"%work_list[num]
    files = os.listdir(path)
    
    
    
        
      
    for i in range(files):
        plt.figure(i)
        full_path = os.path.join(path,work_list[num])
        files = os.listdir(full_path)
        f = np.load(os.path.join(full_path,files[i]))
       
        f= np.log10(f)
        #buff = float(buff)
        
        
        
        
        plt.pcolormesh(f*10)
        title("%s"%work_list[num])
        
        
    plt.show()    
    
    
if __name__ == "__main__":
    out_port,in_port=gpio_init()
    work_spec()
    #signal.signal(signal.SIGINT,handler)
    ser=serial.Serial(
        port = '/dev/ttyACM0',# +++automatic find func add please+++
        baudrate = 128004,
        parity = serial.PARITY_NONE,
        stopbits = serial.STOPBITS_ONE,
        bytesize = serial.EIGHTBITS,#char
        timeout = 1
	)
    
    #sched = apscheduler.scheduler.blocking.BackgroundScheduler('apscheduler.job_defaults.max_instances':'2') 
    #sched = BackgroundScheduler()
    #sched =BlockingScheduler()
    #sched.start()
    
    [line_I,line_Q] = read_line(ser)
    raw_data=save_raw_data(line_I,line_Q)
    #stft_scipy(raw_data)
    stft_data=stft_plt(raw_data)
    [stft_data_cut , count , index_start, index_end]=roi_kcy1(stft_data)
    bnzd_in=scale_binarize(stft_data_cut,count)
    
    
    

    tx_data(out_port,bnzd_in,count)
    plot_data(bnzd_in , count , index_start, index_end)
    gpio_fin()
    #sched.add_job(stop,'interval',seconds  = 3,args =[t],id ='job1',max_instances=2)
    
    
    
    
            
    
    
    
    
    
    
    
    
    
    
    
      
