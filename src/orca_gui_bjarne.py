# -*- coding: utf-8 -*-
"""

"""

import pyqtgraph as pg
#from pyqtgraph.Qt import QtCore, QtGui

from PyQt5 import QtGui, QtCore,QtWidgets #, uic
#from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog

import pyqtgraph.console
import numpy as np
import time
import zmq # for sync global variables between PCs

from pyqtgraph.dockarea import *

import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

import ast
import json
from collections import OrderedDict
import sys
sys.path.append(    "C:\\Users\\qopt\\Desktop\\orca_camera")

from dcam import *
import dcam_bjarne
import atexit

import faulthandler, traceback 
faulthandler.enable() 
sys.excepthook = lambda *a: traceback.print_exception(*a)

# safe exit (release usb interface)
def goingaway():
    Dcamapi.uninit()
    print("Bye!!")
    
atexit.register(goingaway)

#######
####### JREMEMBER to SETGET the exposuretime value
#######

class ImageAcquisition(QThread):
# need lib usb started by main thread (see below)

    imagesignal1 = pyqtSignal(np.ndarray)
    imagesignal2 = pyqtSignal(np.ndarray)
    imagesignal3 = pyqtSignal(np.ndarray)
    #imagesignal3 = pyqtSignal(object)
    customupdate = pyqtSignal(object)
    # --- BEGIN sorter hook (added) ---
    sorterhook = pyqtSignal(np.ndarray)   # loading frame (image 2), emitted the moment it is read
    # --- END sorter hook (added) ---
    
    def __init__(self):
        QThread.__init__(self)
        self.dcam = None
        self.settings = None
        self.softwareTrigger = True
        self.waitforupdate = False
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        
        self.globalvars = None

    #def __del__(self):
    #    self.wait()
        
    def setSettings(self,settings):
        self.settings = settings
        
    def setWaitforupdate(self, w = False):
        self.waitforupdate = w
        
    def getDeviceString(self,dcam):
        #get ID strings
        iDevice = 0
        output = 'cam#{}:'.format(iDevice)
        
        for idstr in DCAM_IDSTR:
            #print((idstr.name,idstr.value))
            val = dcam.dev_getstring(idstr.value)
            if val is False:
                output = output + ':'#'No '+idstr.name
            else:
                output = output + ':{}'.format(val)

        print(output)
        return output
        
    def program_settings(self,dcam,settings):
        
        print('SENSOR TEMPERATURE:', dcam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE))
        print('SENSORCOOLER STATUS:', dcam.prop_getvalue(DCAM_IDPROP.SENSORCOOLERSTATUS))
        
        ### SENSOR MODE

        ret = dcam.prop_setvalue(DCAM_IDPROP.SENSORMODE,DCAMPROP.SENSORMODE.AREA)
        if ret is False:
            print('-NG: sensormode setting fails with error {}'.format(dcam.lasterr()))

        ### READOUT SPEED (ultraquiet scan or normal)

        ultraquietscan = 1.0
        standardscan = 2.0
        ret = dcam.prop_setvalue(DCAM_IDPROP.READOUTSPEED,settings['scan'])
        if ret is False:
            print('-NG: readoutspeed setting fails with error {}'.format(dcam.lasterr()))

        ### TRIGGER SETTINGS
        # mode
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_MODE,DCAMPROP.TRIGGER_MODE.NORMAL)
        if ret is False:
            print('-NG: trigger mode fails with error {}'.format(dcam.lasterr()))

        # source
        #ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE,DCAMPROP.TRIGGERSOURCE.EXTERNAL)
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE,settings['trigsource'])
        if ret is False:
            print('-NG: trigger source fails with error {}'.format(dcam.lasterr()))
            
        
        TRIGGERSOURCETEST = dcam.prop_getvalue(DCAM_IDPROP.TRIGGERSOURCE)
        print('TRIGGERSOURCETEST:',TRIGGERSOURCETEST)
            
        if settings['trigsource'] == DCAMPROP.TRIGGERSOURCE.EXTERNAL:
            self.softwareTrigger = False
        else:
            self.softwareTrigger = True


        # active 
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE,settings['trigactive'])
        #ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE,DCAMPROP.TRIGGERACTIVE.LEVEL) # exposure length is control by trigger duration
        if ret is False:
            print('-NG: trigger active fails with error {}'.format(dcam.lasterr()))

        # polarity
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERPOLARITY,settings['trigpolarity']) # rising edge
        if ret is False:
            print('-NG: trigger polarity fails with error {}'.format(dcam.lasterr()))

        # nb
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERTIMES,1.0) # number of trigger events to start acquiring 1 frame
        if ret is False:
            print('-NG: trigger times fails with error {}'.format(dcam.lasterr()))

        # delay
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERDELAY,settings['trigdelay'])
        if ret is False:
            print('-NG: trigger delay fails with error {}'.format(dcam.lasterr()))

        # type of exposure
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE,settings['exposuretype']) # rolling shutter with global reset
        #ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE,DCAMPROP.TRIGGER_GLOBALEXPOSURE.DELAYED) # rolling shutter
        if ret is False:
            print('-NG: trigger globalexposure fails with error {}'.format(dcam.lasterr()))

        ### EXPOSURE TIME

        ret = dcam.prop_setgetvalue(DCAM_IDPROP.EXPOSURETIME,settings['exposuretime'])
        if ret is False:
            print('-NG: exposure time fails with error {}'.format(dcam.lasterr()))
        else:
            self.customupdate.emit(dict((('exposuretime',ret),('none',None))))
            
            
        ### ALU
        ret = dcam.prop_setvalue(DCAM_IDPROP.DEFECTCORRECT_MODE,DCAMPROP.DEFECTCORRECT_MODE.ON)
        if ret is False:
            print('-NG: defectcorrection mode fails with error {}'.format(dcam.lasterr()))

        ret = dcam.prop_setvalue(DCAM_IDPROP.HOTPIXELCORRECT_LEVEL,DCAMPROP.HOTPIXELCORRECT_LEVEL.STANDARD)
        if ret is False:
            print('-NG: hotpixelcorrect level fails with error {}'.format(dcam.lasterr()))

        ret = dcam.prop_setvalue(DCAM_IDPROP.INTENSITYLUT_MODE,DCAMPROP.INTENSITYLUT_MODE.THROUGH) # disable it
        if ret is False:
            print('-NG: intensity LUT mode fails with error {}'.format(dcam.lasterr()))

            
        ### BINNING AND ROIs

        #binning 1
        ret = dcam.prop_setvalue(DCAM_IDPROP.BINNING,settings['binning']) #._2 and ._4 available (horz and vert binning are dependent)
        if ret is False:
            print('-NG: binning fails with error {}'.format(dcam.lasterr()))

            
        # I found this in the documents:
        # "set subarray mode off. This setting is not mandatory, but you have to control the setting order of offset and size when mode is on. "
            
        # en/disable subarray (ROIs)
        #ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE,settings['subarray'])
        ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE,DCAMPROP.MODE.OFF)
        if ret is False:
            print('-NG: ROI mode fails with error {}'.format(dcam.lasterr()))
            
        #hpos : 0 to 4092 , step 4 , default 0
        #hsize: 4 to 4096 , step 4 , default 4096
        #vpos: 0 to 2300 , step 4 , default 0
        #vsize: 4 to 2304 , step 4 , default 2304

        #(always check the max sensor size to clampl hsize and vsize otherwise it fails)
        # hpos = 0
        # vpos = 0
        # hsize = 4096
        # vsize = 2304
        ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYHPOS,settings['hpos'])
        if ret is False:
            print('-NG: SUBARRAYHPOS fails with error {}'.format(dcam.lasterr()))
        
        ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVPOS,settings['vpos'])
        if ret is False:
            print('-NG: SUBARRAYVPOS fails with error {}'.format(dcam.lasterr()))
        
        ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYHSIZE,settings['hsize'])
        if ret is False:
            print('-NG: SUBARRAYHSIZE fails with error {}'.format(dcam.lasterr()))
            
        ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVSIZE,settings['vsize'])
        if ret is False:
            print('-NG: SUBARRAYVSIZE fails with error {}'.format(dcam.lasterr()))
            
        #"set subarray mode on. The combination of offset and size is checked on this timing."
        ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE,settings['subarray'])
        #ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE,DCAMPROP.MODE.ON)
        if ret is False:
            print('-NG: ROI mode fails with error {}'.format(dcam.lasterr()))
            
            
        # extra information
        # synchronous timing
    
    # TIMING_CYCLICTRIGGERPERIOD = 4206624  # 0x00403020, R/O, sec,   "TIMING CYCLIC TRIGGER PERIOD"
    

    
    # INTERNALFRAMERATE = 4208656  # 0x00403810, R/W, 1/sec,  "INTERNAL FRAME RATE"
    # INTERNAL_FRAMEINTERVAL = 4208672  # 0x00403820, R/W, sec,   "INTERNAL FRAME INTERVAL"
    # INTERNALLINERATE = 4208688  # 0x00403830, R/W, 1/sec,   "INTERNAL LINE RATE"
    # INTERNALLINESPEED = 4208704  # 0x00403840, R/W, m/sec,  "INTERNAL LINE SPEEED"
    # INTERNAL_LINEINTERVAL = 4208720  # 0x00403850, R/W, sec,    "INTERNAL LINE INTERVAL"
    
    
        ret = dcam.prop_getvalue(DCAM_IDPROP.TIMING_READOUTTIME)
        if ret is False:
            print('-NG: readouttime fails with error {}'.format(dcam.lasterr()))
        else:
            self.customupdate.emit(dict((('readouttime',ret),('none',None))))
         
        ret = dcam.prop_getvalue(DCAM_IDPROP.TIMING_MINTRIGGERINTERVAL)
        if ret is False:
            print('-NG: mintriggerinterval fails with error {}'.format(dcam.lasterr()))
        else:
            self.customupdate.emit(dict((('mintriginterval',ret),('none',None))))     
            
        # ret = dcam.prop_getvalue(DCAM_IDPROP.TIMING_MINTRIGGERBLANKING)
        # if ret is False:
            # print('-NG: TIMING_MINTRIGGERBLANKING fails with error {}'.format(dcam.lasterr()))
        # else:
            # self.customupdate.emit(dict((('mintrigblanking',ret),('none',None))))             
         
        ret = dcam.prop_getvalue(DCAM_IDPROP.TIMING_GLOBALEXPOSUREDELAY)
        if ret is False:
            print('-NG: globalexposuredelay fails with error {}'.format(dcam.lasterr()))
        else:
            self.customupdate.emit(dict((('globalexpdelay',ret),('none',None))))         
         
        # ret = dcam.prop_getvalue(DCAM_IDPROP.TIMING_EXPOSURE)
        # if ret is False:
            # print('-NG: timingexposurerolling fails with error {}'.format(dcam.lasterr()))
        # else:
            # self.customupdate.emit(dict((('exposurerolling',ret),('none',None))))   

        ret = dcam.prop_getvalue(DCAM_IDPROP.TIMING_INVALIDEXPOSUREPERIOD)
        if ret is False:
            print('-NG: TIMING_INVALIDEXPOSUREPERIOD fails with error {}'.format(dcam.lasterr()))
        else:
            self.customupdate.emit(dict((('invalidexposure',ret),('none',None))))        
            
            
        # ret = dcam.prop_getvalue(DCAM_IDPROP.INTERNALFRAMERATE)
        # if ret is False:
            # print('-NG: internalfps fails with error {}'.format(dcam.lasterr()))
        # else:
            # self.customupdate.emit(dict((('internalfps',ret),('none',None))))

        ret = dcam.prop_getvalue(DCAM_IDPROP.CONVERSIONFACTOR_COEFF)
        if ret is False:
            print('-NG: conversionfactorcoeff fails with error {}'.format(dcam.lasterr()))
        else:
            self.customupdate.emit(dict((('convfac',ret),('none',None))))                 
            
        ret = dcam.prop_getvalue(DCAM_IDPROP.CONVERSIONFACTOR_OFFSET)
        if ret is False:
            print('-NG: conversionfactoroffset fails with error {}'.format(dcam.lasterr()))
        else:
            self.customupdate.emit(dict((('convfacoffset',ret),('none',None))))             
       
        ret = dcam.prop_getvalue(DCAM_IDPROP.INTERNAL_LINEINTERVAL)
        if ret is False:
            print('-NG: INTERNAL_LINEINTERVAL    fails with error {}'.format(dcam.lasterr()))
        else:
            self.customupdate.emit(dict((('lineinterval',ret),('none',None))))             
    
        return
        
    def program_standard_settings(self,dcam):
        
        print('SENSOR TEMPERATURE:', dcam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE))
        print('SENSORCOOLER STATUS:', dcam.prop_getvalue(DCAM_IDPROP.SENSORCOOLERSTATUS))
        ### SENSOR MODE

        ret = dcam.prop_setvalue(DCAM_IDPROP.SENSORMODE,DCAMPROP.SENSORMODE.AREA)
        if ret is False:
            print('-NG: sensormode setting fails with error {}'.format(dcam.lasterr()))

        ### READOUT SPEED (ultraquiet scan or normal)

        ultraquietscan = 1.0
        standardscan = 2.0
        #ret = dcam.prop_setvalue(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.SLOWEST)
        #ret = dcam.prop_setvalue(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.FASTEST)
        ret = dcam.prop_setvalue(DCAM_IDPROP.READOUTSPEED,standardscan)
        if ret is False:
            print('-NG: readoutspeed setting fails with error {}'.format(dcam.lasterr()))

        ### TRIGGER SETTINGS

        # source
        #ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE,DCAMPROP.TRIGGERSOURCE.EXTERNAL)
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERSOURCE,DCAMPROP.TRIGGERSOURCE.SOFTWARE)
        self.softwareTrigger = True
        if ret is False:
            print('-NG: trigger source fails with error {}'.format(dcam.lasterr()))

        # mode
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_MODE,DCAMPROP.TRIGGER_MODE.NORMAL)
        if ret is False:
            print('-NG: trigger mode fails with error {}'.format(dcam.lasterr()))

        # active 
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE,DCAMPROP.TRIGGERACTIVE.EDGE)
        #ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERACTIVE,DCAMPROP.TRIGGERACTIVE.LEVEL) # exposure length is control by trigger duration
        if ret is False:
            print('-NG: trigger active fails with error {}'.format(dcam.lasterr()))

        # polarity
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERPOLARITY,DCAMPROP.TRIGGERPOLARITY.POSITIVE) # rising edge
        if ret is False:
            print('-NG: trigger polarity fails with error {}'.format(dcam.lasterr()))

        # nb
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERTIMES,1.0) # number of trigger events to start acquiring 1 frame
        if ret is False:
            print('-NG: trigger times fails with error {}'.format(dcam.lasterr()))

        # delay
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGERDELAY,0.0)
        if ret is False:
            print('-NG: trigger delay fails with error {}'.format(dcam.lasterr()))

        # type of exposure
        ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE,DCAMPROP.TRIGGER_GLOBALEXPOSURE.GLOBALRESET) # rolling shutter with global reset
        #ret = dcam.prop_setvalue(DCAM_IDPROP.TRIGGER_GLOBALEXPOSURE,DCAMPROP.TRIGGER_GLOBALEXPOSURE.DELAYED) # rolling shutter
        if ret is False:
            print('-NG: trigger globalexposure fails with error {}'.format(dcam.lasterr()))

        ### EXPOSURE TIME
        #it will be round to multiples of 7.2 us, although steps of 1us are possible

        #in ultraquiet speed: 0.0001728 to 1800.0 , step 0.00000001 , default 0.0082944
        #in standard speed: 0.0000072 to 1800.0 , step 0.00000001 , default 0.0082944
        exptime = 0.0082944#1000*7.2e-6
        ret = dcam.prop_setvalue(DCAM_IDPROP.EXPOSURETIME,exptime)
        if ret is False:
            print('-NG: exposure time fails with error {}'.format(dcam.lasterr()))

        ### SENSOR
        #(w) sensorcooler mode, NOT SURE?

        ### ALU
        ret = dcam.prop_setvalue(DCAM_IDPROP.DEFECTCORRECT_MODE,DCAMPROP.DEFECTCORRECT_MODE.ON)
        if ret is False:
            print('-NG: defectcorrection mode fails with error {}'.format(dcam.lasterr()))

        ret = dcam.prop_setvalue(DCAM_IDPROP.HOTPIXELCORRECT_LEVEL,DCAMPROP.HOTPIXELCORRECT_LEVEL.STANDARD)
        if ret is False:
            print('-NG: hotpixelcorrect level fails with error {}'.format(dcam.lasterr()))

        ret = dcam.prop_setvalue(DCAM_IDPROP.INTENSITYLUT_MODE,DCAMPROP.INTENSITYLUT_MODE.THROUGH) # disable it
        if ret is False:
            print('-NG: intensity LUT mode fails with error {}'.format(dcam.lasterr()))

            
        ### BINNING AND ROIs

        #binning 1
        ret = dcam.prop_setvalue(DCAM_IDPROP.BINNING,DCAMPROP.BINNING._1) #._2 and ._4 available (horz and vert binning are dependent)
        if ret is False:
            print('-NG: binning fails with error {}'.format(dcam.lasterr()))

        # en/disable subarray (ROIs)
        ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE,DCAMPROP.MODE.OFF)
        #ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYMODE,DCAMPROP.MODE.ON)
        if ret is False:
            print('-NG: ROI mode fails with error {}'.format(dcam.lasterr()))
            
        #if ON
        #hpos : 0 to 4092 , step 4 , default 0
        #hsize: 4 to 4096 , step 4 , default 4096
        #vpos: 0 to 2300 , step 4 , default 0
        #vsize: 4 to 2304 , step 4 , default 2304

        #(always check the max sensor size to clampl hsize and vsize otherwise it fails)
        # hpos = 0
        # vpos = 0
        # hsize = 4096
        # vsize = 2304
        # ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYHPOS,hpos)
        # ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVPOS,vpos)
        # ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYHSIZE,hsize)
        # ret = dcam.prop_setvalue(DCAM_IDPROP.SUBARRAYVSIZE,vsize)
        return
            
            
    def run(self):
        
        if self.dcam is not None:
            
            
            if self.dcam.is_opened() == True: # if not closed properly, clean and close
                #self.dcam.buf_release()
                self.dcam.dev_close()
                
        else:
            # try first device
            iDevice = 0
            self.dcam = Dcam(iDevice)
            
        
        # get camera string information
        str =  self.getDeviceString(self.dcam)
        self.customupdate.emit(dict((('status_msg',str),('none',None))))
        
        
        # camera open
        if self.dcam.is_opened() == False:
        
            ret = self.dcam.dev_open()
            if ret == False:
                print('-NG: Dcam.dev_open() fails with error {}'.format(self.dcam.lasterr()))
                self.customupdate.emit(dict((('error_msg',"Thread error: cannot open device."),('none',None))))
                return
                
        self.customupdate.emit(dict((('status_msg',"camera open."),('none',None))))
            
        # now, it should be opened
        # program settings
        self.customupdate.emit(dict((('status_msg',"Program settings.."),('none',None))))
        #self.program_standard_settings(self.dcam)
        self.program_settings(self.dcam,self.settings)
        
        
        # allocate buffers, if success, start acquisition loop
        nb_frames = 3
        timeout_milisec = 250
        max_timeoutcounts = 100
        acq_nb = 0
        frame = 0
        
        if self.dcam.buf_alloc(nb_frames) is not False:
        
            self.customupdate.emit(dict((('status_msg',"Memory buffers allocated."),('none',None))))
            
            while True: # thread loop
            
                if self.isInterruptionRequested() == True:
                    self.customupdate.emit(dict((('status_msg',"Stop request received. Waiting.."),('none',None))))
                    break


                if self.waitforupdate == True:
                    
                    self.customupdate.emit(dict((('status_msg',"Wait for update..."),('none',None))))
                    time.sleep(2)
                    
                    #if did not get:
                    #  Wait for next request from client
                    
                    try:
                        print(self.context.closed)
                        print(self.socket.closed)
                        #check for a message, this will not block
                        #self.socket.send(b"Hello")
                        message = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                        #byte_str = self.socket.recv(flags=zmq.NOBLOCK)
                        #print(byte_str)
                        
                        #dict_str = byte_str.decode("UTF-8")
                        
                        #message = ast.literal_eval(dict_str)
                        #print(message)
                        #a message has been received
                        #print("Received request: %s" % message)
                        
                        if 'newvariables' in message['request']:
                            self.globalvars = message
                            p1 = message['param1']
                            p3 = message['param3']
                            p2 = message['param2']
                            p4 = message['param4']
                            d1 = message['param1desc']
                            d2 = message['param2desc']
                            d3 = message['param3desc']
                            d4 = message['param4desc']
                            self.customupdate.emit(dict((('status_msg',"Global variables updates"),('globalvariables_val',[p1,p2,p3,p4]),('globalvariables_desc',[d1,d2,d3,d4]))))
                            self.socket.send(b"Camera program received global variables.")
                        else:
                            #  Send reply back to client
                            self.socket.send(b"Camera program wrong information")
                            continue
                        
                    except zmq.ZMQError as e:
                        print('ZMQError:',e)
                        continue
                    
                    except zmq.Again as e:
                        #No message received yet"
                        print('no message received yet')
                        continue # to be able to interrupt main thread from GUI
                    
                    except zmq.ContextTerminated as e:
                        print('no ZMQ context')
                        continue
                        
                    
                    ## global variable test
                    #p1 = np.random.randint(10)
                    #p2 = np.random.randint(10)
                    #p3 = np.random.randint(10)
                    #p4 = np.random.randint(10)
                    #d1 = d2 = d3 = d4 = 'test%d' % p1
                    #self.customupdate.emit(dict((('status_msg',"Global variables updates"),('globalvariables_val',[p1,p2,p3,p4]),('globalvariables_desc',[d1,d2,d3,d4]))))
                    
                frame = 0
                if self.dcam.cap_start() is not False:
                    
                    acq_nb += 1
                    temperature = self.dcam.prop_getvalue(DCAM_IDPROP.SENSORTEMPERATURE)
                    stat = self.dcam.cap_status()
                    capture_status = DCAMCAP_STATUS(stat).name
                    
                    self.customupdate.emit(dict((('status_msg',"Acquisition #%d" % acq_nb),('temperature',temperature),('capture_status',capture_status))))
                    
                    #print('framebundle mode:', self.dcam.prop_getvalue(DCAM_IDPROP.FRAMEBUNDLE_MODE))
                    
                    counts = 0
                    loopn = 0
                    while True:
                        
                        loopn += 1
                        print('LOOPN:', loopn)
                        
                        #USE SOFTWARE TRIGGER
                        if self.softwareTrigger == True:
                            time.sleep(0.2)
                            if self.dcam.cap_firetrigger() is False:
                               print('-NG: Dcam.cap_firetrigger() fails with error {}'.format(self.dcam.lasterr()))
                               break
                            else:
                               print('fire software trigger!')
                               
                        #there could be other or multiple events happening.. so check cap_transferinfo..
                        #DCAMWAIT_CAPEVENT_STOPPED
                        #also, it seems that there is no framebundle by default but it could become active in subarray mode? to be checked
                            
                        if self.dcam.wait_capevent_frameready(timeout_milisec) is not False:
                        
                            ##cap_transferinfo
                            ret = self.dcam.cap_transferinfo()
                            if ret is not False:
                                print('TRANSFERINFO (framecount,newframeindex):', ret.nFrameCount, ret.nNewestFrameIndex)
                            else:
                                print('-NG: Dcam.cap_transferinfo() fails with error {}'.format(self.dcam.lasterr()))
                            
                            #data = self.dcam.buf_getlastframedata()
                            #data = self.dcam.buf_getframedata(frame)
                            #if frame == 0:
                            #    self.imagesignal1.emit(data)
                            #if frame == 1:
                            #    self.imagesignal2.emit(data)
                                    
                            
                            frame+=1
                            # --- BEGIN sorter hook (added) ---
                            if frame == 2:
                                self.sorterhook.emit(self.dcam.buf_getframedata(1))
                            # --- END sorter hook (added) ---
                            if frame >= nb_frames:
                                data1 = self.dcam.buf_getframedata(0)
                                #data = self.dcam.buf_getlastframedata()
                                self.imagesignal1.emit(data1)
                                data2 = self.dcam.buf_getframedata(1)
                                self.imagesignal2.emit(data2)
                                data3 = self.dcam.buf_getframedata(2)
                                #dt = {'data' : data3}
                                #self.imagesignal3.emit(dict(dt,**self.globalvars))
                                self.imagesignal3.emit(data3)
                                break
                            else:
                                continue

                        # here, there was a wait_capevent error
                        dcamerr = self.dcam.lasterr()
                        if dcamerr.is_timeout():
                            print('===: timeout')
                            counts+=1
                            self.customupdate.emit(dict((('status_msg',"===: timeout #%d" % counts),('none',None))))

                        else:
                            print('-NG: Dcam.wait_event() fails with error {}'.format(dcamerr))
                            break
                            
                        if counts > (max_timeoutcounts-1):
                            self.customupdate.emit(dict((('status_msg',"Max timeout counts reached."),('none',None))))
                            break
                            
                        if self.isInterruptionRequested() == True:
                            self.customupdate.emit(dict((('status_msg',"Stop request received. Waiting.."),('none',None))))
                            break
                            
                    self.dcam.cap_stop()

                else:
                    print('-NG: Dcam.cap_snapshot() fails with error {}'.format(self.dcam.lasterr()))
                    break
                    
            self.dcam.buf_release()
        else:
            print('-NG: Dcam.buf_alloc() fails with error {}'.format(self.dcam.lasterr()))
            self.customupdate.emit(dict((('status_msg',"Could not allocate buffers!"),('none',None))))
        
        
         #always close properly
        if self.dcam.is_opened() == True:
            self.dcam.dev_close()
            
        return
                
##################
##################



params = [
    {'name': 'Main camera settings', 'type': 'group', 'children': [
        {'name': 'Readout speed', 'type': 'list', 'values': {"standard": 1, "ultraquiet": 2}, 'value': 1},
        
        {'name': 'Exposure settings', 'type': 'group', 'children': [
        {'name': 'Exposure type', 'type': 'list', 'values': {"global reset": 1, "no global reset": 2}, 'value': 1},
        {'name': 'Exposure time', 'type': 'float', 'value': 0.0082944, 'limits': (0.0000072, 1800.0), 'step': 0.00000001, 'siPrefix': True, 'suffix': 's','decimals': 8}
        ]},
        

        {'name': 'Trigger settings', 'type': 'group', 'children': [
        {'name': 'Trigger Source', 'type': 'list', 'values': {"software": 1, "external": 2}, 'value': 1},
        {'name': 'Trigger Polarity', 'type': 'list', 'values': {"positive": 1, "negative": 2}, 'value': 1},
        {'name': 'Trigger Active', 'type': 'list', 'values': {"edge": 1, "level": 2}, 'value': 1},
        {'name': 'Trigger Delay', 'type': 'float', 'value': 0.0000, 'limits': (0.0, 10.0), 'step': 0.000001, 'siPrefix': True, 'suffix': 's', 'decimals' : 6}
        ]},

    ]},
    {'name': 'Image settings', 'type': 'group', 'children': [
        {'name': 'Binning', 'type': 'list', 'values': {"1x1": 1, "2x2": 2, "4x4": 3}, 'value': 1},
        {'name': 'Subarray mode', 'type': 'list', 'values': {"ACTIVE": 1, "OFF": 2}, 'value': 2, 'expanded' : False, 'children': [
        {'name': 'hpos', 'type': 'int', 'value': 0, 'limits': (0, 4092), 'default': 0, 'step': 4, 'decimals': 6},
        {'name': 'vpos', 'type': 'int', 'value': 0, 'limits': (0, 2300), 'default': 0, 'step': 4, 'decimals': 6},
        {'name': 'vsize', 'type': 'int', 'value': 2304, 'limits': (4, 2304), 'default': 2304, 'step': 4,'decimals': 6},
        {'name': 'hsize', 'type': 'int', 'value': 4096, 'limits': (4, 4096), 'default': 4096, 'step': 4,'decimals': 6}]}#, 'siPrefix': False, 'suffix': 'px'},
        
    ]},
    {'name': 'File savings', 'type': 'group', 'children': [
        {'name': 'Series pathname', 'type': 'str', 'value' : "./data_folder/tweezer1_"},
        {'name': 'File format', 'type': 'list', 'values': {".npy": 1, ".png": 2}, 'value': 1},
        {'name': 'Save files?', 'type': 'list', 'values' : {"Yes" : 1, "No" : 2}, 'value': 2},
        {'name': 'Last saved file', 'type': 'str', 'value' : ""},
        {'name': 'Counter', 'type': 'int', 'value': 0, 'default': 0, 'step': 1, 'decimals': 6}
    ]},
    {'name': 'Extra information', 'type': 'group', 'children': [
        {'name': 'Readout time', 'type': 'float', 'value': 0.000, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': 's', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        {'name': 'Min. trigger interval', 'type': 'float', 'value': 0.000, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': 's', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        #{'name': 'Minimum trigger blanking', 'type': 'float', 'value': 0.000, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': 's', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        {'name': 'Global exposure delay', 'type': 'float', 'value': 0.000, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': 's', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        #{'name': 'Exposure rolling time', 'type': 'float', 'value': 0.000, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': 's', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        {'name': 'Invalid exposure period', 'type': 'float', 'value': 0.000, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': 's', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        #{'name': 'FPS', 'type': 'float', 'value': 0.000, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': 'Hz', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        {'name': 'Line interval', 'type': 'float', 'value': 0.000, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': 's', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        {'name': 'Conversion factor (e-/pxcount)', 'type': 'float', 'value': 0.11, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': '', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        {'name': 'Conversion offset (pxcount)', 'type': 'float', 'value': 200.0, 'limits': (0, 40000.0), 'siPrefix': True, 'suffix': '', 'readonly': True, 'decimals' : 4, 'step' : 1.0},
        {'name': 'Sensor temperature', 'type': 'float', 'value': -20.0, 'limits': (-100.0, 100.0), 'suffix': u"\N{DEGREE SIGN}"+'C', 'readonly': True}
    ]},
    
    {'name': 'Global variables', 'type': 'group', 'children': [
        {'name': 'Waitforupdate?', 'type': 'list', 'values' : {"Yes" : 1, "No" : 2}, 'value': 2},
        
        {'name': 'param1','type': 'float', 'value': 0.000, 'limits': (-1e-10, 1e10), 'siPrefix': True, 'readonly': True, 'decimals' : 4, 'expanded' : False, 'children': [
        {'name': 'description', 'type': 'str', 'value': ''},
        ]},
        
        {'name': 'param2','type': 'float', 'value': 0.000, 'limits': (-1e-10, 1e10), 'siPrefix': True, 'readonly': True, 'decimals' : 4, 'expanded' : False, 'children': [
        {'name': 'description', 'type': 'str', 'value': ''},
        ]},
        
        {'name': 'param3','type': 'float', 'value': 0.000, 'limits': (-1e-10, 1e10), 'siPrefix': True, 'readonly': True, 'decimals' : 4, 'expanded' : False, 'children': [
        {'name': 'description', 'type': 'str', 'value': ''},
        ]},
        
        {'name': 'param4','type': 'float', 'value': 0.000, 'limits': (-1e-10, 1e10), 'siPrefix': True, 'readonly': True, 'decimals' : 4, 'expanded' : False, 'children': [
        {'name': 'description', 'type': 'str', 'value': ''},
        ]}
    ]}
    
]


class MainWindow(pg.Qt.QtWidgets.QMainWindow):
    def __init__(self, dcamapistat = False, *args, **kwargs):
        #super(self.__class__, self).__init__()
        super(MainWindow, self).__init__(*args, **kwargs)
        
        self.dcamapistat = dcamapistat
        
        self.nx = 40
        self.ny = 40
        self.image1 = np.random.randint(1, 100, size=(self.nx, self.ny))
        
        #print('received:',self.dcamapistat)
        # Enable antialiasing for prettier plots, row-major to be compatible with matplotlib and others
        pg.setConfigOptions(antialias=False,imageAxisOrder='row-major') 
        
        
        area = DockArea()
        self.setCentralWidget(area)
        self.resize(1000,500)
        self.setWindowTitle('new camera GUI')
        
        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still have to
        ## fill the entire dock area and obey the limits of their internal widgets.
        self.d1 = Dock("D1 - Process", size=(1, 1))     ## give this dock the minimum possible size
        self.d2 = Dock("D2 - Cam Settings", size=(500,400))#, closable=True)
        self.d3 = Dock("Dock3", size=(500,400))
        self.d4 = Dock("D4 - Image 2", size=(500,200))
        self.d5 = Dock("D5 - Image 1", size=(1000,200))
        self.d6 = Dock("D6 - Image 3", size=(500,200))
        #self.d7 = Dock("D7 - Plot", size=(500,200))
        area.addDock(self.d5,'top')
        area.addDock(self.d1,'bottom', self.d5)
        area.addDock(self.d4, 'right')
        #area.addDock(d6, 'top', d4)   ## place d5 at top edge of d4
        area.addDock(self.d6, 'above', self.d4)   ## above to stack on top
        #area.addDock(self.d7, 'above', self.d6)   ## above to stack on top
        area.addDock(self.d2, 'bottom', self.d4)
        
        
        # Create one parameter (it will be a group with subgroups etc)
        self._p = Parameter.create(name='params', type='group', children=params)
        self._p.sigTreeStateChanged.connect(self._change)
        
        self._p.param('Image settings', 'Subarray mode', 'hpos').sigValueChanged.connect(self._Hpos_Change)
        self._p.param('Image settings', 'Subarray mode', 'vpos').sigValueChanged.connect(self._Vpos_Change)
        self._p.param('Main camera settings', 'Readout speed').sigValueChanged.connect(self._Speed_Change)
        
        
        # Too lazy for recursion:
        #for child in self._p.children():
        #    child.sigValueChanging.connect(self._valueChanging)
        #    for ch2 in child.children():
        #        ch2.sigValueChanging.connect(self._valueChanging)
        
        
        ## Create one ParameterTree widget
        self._t = ParameterTree()
        # self._t .setStyleSheet("""
        # QTreeView {
            # background-color: rgb(46, 52, 54);
            # alternate-background-color: rgb(39, 44, 45);    
            # color: rgb(238, 238, 238);
        # }
        # QLabel {
            # color: rgb(238, 238, 238);
        # }
        # QTreeView::item:has-children {
            # background-color: '#212627';
            # color: rgb(233, 185, 110);
        # }
        # QTreeView::item:selected {
            # background-color: rgb(92, 53, 102);
        # }
            # """)
        self._t.setParameters(self._p, showTop=False)
        self._t.setWindowTitle('Parameter Tree')

        #### NOW attached widgets to all docks
        
        # first dock gets save/restore buttons
        w1 = pg.LayoutWidget()
        self.label = pg.Qt.QtWidgets.QLabel(""" -- New camera GUI -- I am testing a few things...""")
        self.runBtn = pg.Qt.QtWidgets.QPushButton('Run')
        saveBtn = pg.Qt.QtWidgets.QPushButton('Save settings')
        restoreBtn = pg.Qt.QtWidgets.QPushButton('Load settings')

        w1.addWidget(self.label, row=0, col=0)
        w1.addWidget(self.runBtn, row=1, col=0)
        w1.addWidget(saveBtn, row=2, col=0)
        w1.addWidget(restoreBtn, row=3, col=0)
        self.d1.addWidget(w1)
        self.state = None
        saveBtn.clicked.connect(self._save)
        restoreBtn.clicked.connect(self._load)
        self.runBtn.clicked.connect(self._run)
        ## test save/restore
        s = self._p.saveState()
        self._p.restoreState(s)


        #w2 = pg.console.ConsoleWidget()
        #w1.addWidget(t, row=0, col=0)
        self.d2.hideTitleBar()
        self.d2.addWidget(self._t)


        ## Hide title bar on dock 3
        #d3.hideTitleBar()
        #w3 = pg.PlotWidget(title="Plot inside dock with no title bar")
        #w3.plot(np.random.normal(size=100))
        #d3.addWidget(w3)


        self.im1 = pg.ImageView()#view=pg.PlotItem())
        data_img = np.random.normal(size=(100,100))
        self.im1.setImage(data_img)
        #pos = np.array([0.0, 0.5, 1.0])
        #color = np.array([[0,0,0,255], [255,128,0,255], [255,255,0,255]], dtype=np.ubyte)
        #list the available colormaps
        #print(Gradients.keys())
        #map = pg.ColorMap(pos, color)
        map = pg.ColorMap(*zip(*Gradients["viridis"]["ticks"]))

        self.im1.setColorMap(map)
        self.d5.hideTitleBar()
        self.d5.addWidget(self.im1)

        #w4 = pg.PlotWidget(title="Dock 4 plot")
        #w4.plot(np.random.normal(size=100))
        self.im2 = pg.ImageView()
        self.im2.setImage(np.random.normal(size=(1000,1000)))
        self.im2.setColorMap(map)
        #d4.hideTitleBar()
        self.d4.addWidget(self.im2)

        #w6 = pg.PlotWidget(title="Dock 6 plot")
        #w6.plot(np.random.normal(size=100))
        self.im3 = pg.ImageView()
        self.im3.setImage(np.random.normal(size=(1000,1000)))
        self.im3.setColorMap(map)
        #d6.hideTitleBar()
        self.d6.addWidget(self.im3)

        #w7 = pg.PlotWidget(title="Dock 7 plot")
        #p7 = w7.plot(np.random.normal(size=100))
        #self.d7.addWidget(w7)
        
        
        self.statusbar = pg.Qt.QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)
        
        self.thread = ImageAcquisition()
        self.thread.customupdate.connect(self._ImageAcq_msg)
        self.thread.imagesignal1.connect(self._updatefig1)
        self.thread.imagesignal2.connect(self._updatefig2)
        self.thread.imagesignal3.connect(self._updatefig3)
        self.thread.finished.connect(self._done)
        
        self.showMaximized()
        
        
    def _ImageAcq_msg(self, post_dic):
        print(post_dic)
        
        if 'error_msg' in post_dic:
            print(post_dic['error_msg'])
            QtWidgets.QMessageBox.critical(self, "Error", post_dic['error_msg'])
            
        if 'status_msg' in post_dic:
            self.statusbar.showMessage(post_dic['status_msg'])
    
        if 'msg' in post_dic:
            print("received msg:", post_dic['msg'])
        
        if 'temperature' in post_dic:
            #self.dsp_sensorTemp.setValue(post_dic['temperature'])
            print(post_dic['temperature'])
            self._p.param('Extra information','Sensor temperature').setValue(post_dic['temperature'])
            
        #if 'capture_status' in post_dic:
        #    self.le_capStat.setText(post_dic['capture_status'])
            
        #if 'sensor_status' in post_dic:
        #    self.le_sensStat.setText(post_dic['sensor_status'])
            
        if 'exposuretime' in post_dic:
            #self.dsp_expTime.setValue(float(post_dic['exposuretime']))
            self._p.param('Main camera settings','Exposure settings','Exposure time').setValue(float(post_dic['exposuretime']))
            print('received exptime:',float(post_dic['exposuretime']))
            
        if 'readouttime' in post_dic:
            self._p.param('Extra information','Readout time').setValue(float(post_dic['readouttime']))
            print('received readouttime:',float(post_dic['readouttime'])) 
            
                    
        if 'mintriginterval' in post_dic:
            self._p.param('Extra information','Min. trigger interval').setValue(float(post_dic['mintriginterval']))
            print('received mintriginterval:',float(post_dic['mintriginterval']))        
            
        # if 'mintrigblanking' in post_dic:
            # self._p.param('Extra information','Minimum trigger blanking').setValue(float(post_dic['mintrigblanking']))
            # print('received mintrigblanking:',float(post_dic['mintrigblanking']))        
                        
            
        if 'globalexpdelay' in post_dic:
            self._p.param('Extra information','Global exposure delay').setValue(float(post_dic['globalexpdelay']))
            print('received globalexpdelay:',float(post_dic['globalexpdelay']))        
            
        # if 'exposurerolling' in post_dic:
            # self._p.param('Extra information','Exposure rolling time').setValue(float(post_dic['exposurerolling']))
            # print('received exposurerolling:',float(post_dic['exposurerolling']))  
            
        if 'invalidexposure' in post_dic:
            self._p.param('Extra information','Invalid exposure period').setValue(float(post_dic['invalidexposure']))
            print('received invalidexposure:',float(post_dic['invalidexposure']))              
            
        # if 'internalfps' in post_dic:
            # self._p.param('Extra information','FPS').setValue(float(post_dic['internalfps']))
            # print('received internalfps:',float(post_dic['internalfps']))  
            
        if 'lineinterval' in post_dic:
            self._p.param('Extra information','Line interval').setValue(float(post_dic['lineinterval']))
            print('received lineinterval:',float(post_dic['lineinterval']))  
            
        if 'convfac' in post_dic:
            self._p.param('Extra information','Conversion factor (e-/pxcount)').setValue(float(post_dic['convfac']))
            print('received convfac:',float(post_dic['convfac'])) 
            
        if 'convfacoffset' in post_dic:
            self._p.param('Extra information','Conversion offset (pxcount)').setValue(float(post_dic['convfacoffset']))
            print('received convfacoffset:',float(post_dic['convfacoffset']))

        if 'globalvariables_val' in post_dic:
            self._p.param('Global variables','param1').setValue(post_dic['globalvariables_val'][0])
            self._p.param('Global variables','param2').setValue(post_dic['globalvariables_val'][1])
            self._p.param('Global variables','param3').setValue(post_dic['globalvariables_val'][2])
            self._p.param('Global variables','param4').setValue(post_dic['globalvariables_val'][3])
            
            self._p.param('Global variables','param1','description').setValue(post_dic['globalvariables_desc'][0])
            self._p.param('Global variables','param2','description').setValue(post_dic['globalvariables_desc'][1])
            self._p.param('Global variables','param3','description').setValue(post_dic['globalvariables_desc'][2])
            self._p.param('Global variables','param4','description').setValue(post_dic['globalvariables_desc'][3])
            
        return
        
    def _updatefig1(self,image):
        self.image1 = image
        self.im1.setImage(self.image1, autoRange=False,autoLevels=True,autoHistogramRange=True)
        return 
        
    def _updatefig2(self,image):
        self.image2 = image
        self.im2.setImage(self.image2, autoRange=False,autoLevels=True,autoHistogramRange=True)    
        return 
        
    def _updatefig3(self,image):
        #image = dt['data']
        self.image3 = image # self.image1.astype(np.int32)-self.image2.astype(np.int32)
        self.im3.setImage(self.image3, autoRange=False,autoLevels=True,autoHistogramRange=True)
        
        if self._p['File savings','Save files?'] == 1:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            filename = self._p['File savings','Series pathname'] + str(int(self._p['File savings','Counter'])) +'_'+timestr
            
            # collect global variables
            globa = {'param1': [self._p['Global variables','param1'],self._p['Global variables','param1','description']],
                                'param2': [self._p['Global variables','param2'],self._p['Global variables','param2','description']],
                                'param3': [self._p['Global variables','param3'],self._p['Global variables','param3','description']],
                                'param4': [self._p['Global variables','param4'],self._p['Global variables','param4','description']]
                                }
            print(globa)                    
            # globa = {
                    # 'param1': dt['param1'],
                    # 'param2': dt['param2'],
                    # 'param3': dt['param3'],
                    # 'param4': dt['param4'],
                    # 'param1desc': dt['param1desc'],
                    # 'param2desc': dt['param2desc'],
                    # 'param3desc': dt['param3desc'],
                    # 'param4desc': dt['param4desc']
                    # }
            try:
                np.save(filename, {'Images':[self.image1,self.image2,self.image3],'globalvariables':globa} )#,globalvariables=globa)
            except:
                print('Oops')
                pass
                
            self._p.param('File savings','Last saved file').setValue(filename)
            self._p.param('File savings','Counter').setValue(int(self._p['File savings','Counter'])+1)
            print('Hey')
        return

    def _saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Choose destination file","","JSON Files (*.json)", options=options)
        return fileName

    def _openFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Choose destination file","","JSON Files (*.json)", options=options)
        return fileName         
        
    def _change(self, param, changes):
        self.label.setText("info: Tree settings changed")
        # commented because it's consuming
        # print("tree changes:")
        # for param, change, data in changes:
            # path = self._p.childPath(param)
            # if path is not None:
                # childName = '.'.join(path)
            # else:
                # childName = param.name()
            # print('  parameter: %s'% childName)
            # print('  change:    %s'% change)
            # print('  data:      %s'% str(data))
            # print('  ----------')
        return
        
        
#    def _valueChanging(self,param, value):
#        print("Value changing (not finalized): %s %s" % (param, value))
#        return
        
    def _Hpos_Change(self):
        hpos = self._p['Image settings','Subarray mode','hpos']
        hsize = self._p['Image settings','Subarray mode','hsize']
        newmxhsize = 4096-hpos
        self._p.param('Image settings','Subarray mode','hsize').setLimits((4,newmxhsize))
        self._p.param('Image settings','Subarray mode','hsize').setValue(np.minimum(hsize,newmxhsize))
    
    def _Vpos_Change(self):
        vpos = self._p['Image settings','Subarray mode','vpos']
        vsize = self._p['Image settings','Subarray mode','vsize']
        newmxvsize = 2304-vpos
        self._p.param('Image settings','Subarray mode','vsize').setLimits((4,newmxvsize))
        self._p.param('Image settings','Subarray mode','vsize').setValue(np.minimum(vsize,newmxvsize))
        
        
    def _Speed_Change(self):
        #print('change speed!', index)
        index = self._p['Main camera settings', 'Readout speed']
        expTime = self._p['Main camera settings', 'Exposure settings', 'Exposure time']
        
        if index == 1: # standard
            self._p.param('Main camera settings', 'Exposure settings', 'Exposure time').setLimits((0.0000072, 1800.0))  #in standard scan mode
            self._p.param('Main camera settings', 'Exposure settings', 'Exposure time').setValue(np.maximum(expTime,0.0000072))
        
        if index == 2: # ultraquiet
            self._p.param('Main camera settings', 'Exposure settings', 'Exposure time').setLimits((0.0001728, 1800.0))#in ultraquiet scan mode
            self._p.param('Main camera settings', 'Exposure settings', 'Exposure time').setValue(np.maximum(expTime,0.0001728))  
            
     
    def _run(self):
    
        if self.thread.isRunning():
            self.thread.requestInterruption()
            return
        
        self._p.param('File savings','Counter').setValue(0)  
        
        #send settings
        scanType = self._p['Main camera settings', 'Readout speed']
        scan = 2.0
        if scanType == 1: # standard
            scan = 2.0
        else:
            scan = 1.0 # ultraquiet
            
        trigSrc = self._p['Main camera settings', 'Trigger settings', 'Trigger Source']
        trigsource = DCAMPROP.TRIGGERSOURCE.SOFTWARE
        if trigSrc == 1:
            trigsource = DCAMPROP.TRIGGERSOURCE.SOFTWARE
        else:
            trigsource = DCAMPROP.TRIGGERSOURCE.EXTERNAL
            
        trigTyp = self._p['Main camera settings', 'Trigger settings', 'Trigger Active']
        trigactive = DCAMPROP.TRIGGERACTIVE.EDGE
        if trigTyp == 1:
             trigactive = DCAMPROP.TRIGGERACTIVE.EDGE
        else:
             trigactive = DCAMPROP.TRIGGERACTIVE.LEVEL
       
        trigPol = self._p['Main camera settings', 'Trigger settings', 'Trigger Polarity']
        trigpolarity = DCAMPROP.TRIGGERPOLARITY.POSITIVE
        if trigPol == 1:
            trigpolarity = DCAMPROP.TRIGGERPOLARITY.POSITIVE
        else:
            trigpolarity = DCAMPROP.TRIGGERPOLARITY.NEGATIVE
        
        trigdelay = self._p['Main camera settings', 'Trigger settings', 'Trigger Delay']
        
        expType = self._p['Main camera settings', 'Exposure settings', 'Exposure type']
        exposuretype = DCAMPROP.TRIGGER_GLOBALEXPOSURE.GLOBALRESET
        if expType == 1:
            exposuretype = DCAMPROP.TRIGGER_GLOBALEXPOSURE.GLOBALRESET
        else:
            exposuretype = DCAMPROP.TRIGGER_GLOBALEXPOSURE.DELAYED
        
        exposuretime = self._p['Main camera settings', 'Exposure settings', 'Exposure time']
        
        bin = self._p['Image settings','Binning']
        binning = DCAMPROP.BINNING._1 
        if bin == 1:
            binning = DCAMPROP.BINNING._1
        if bin == 2:
            binning = DCAMPROP.BINNING._2
        if bin == 3:
            binning = DCAMPROP.BINNING._4
            
        mode = self._p['Image settings','Subarray mode']
        subarray = DCAMPROP.MODE.OFF
        if mode == 1:
            subarray = DCAMPROP.MODE.ON
        else:
            subarray = DCAMPROP.MODE.OFF
            
        hsize = self._p['Image settings','Subarray mode','hsize']
        vsize = self._p['Image settings','Subarray mode','vsize']
        hpos = self._p['Image settings','Subarray mode','hpos']
        vpos = self._p['Image settings','Subarray mode','vpos']
        
        settings = dict((('scan',scan),('trigsource',trigsource),('trigactive',trigactive),('trigpolarity',trigpolarity),('trigdelay',trigdelay),('exposuretype',exposuretype),('exposuretime',exposuretime),('binning',binning),('subarray',subarray),('vsize',vsize),('hsize',hsize),('vpos',vpos),('hpos',hpos)))
        
        # program settings
        self.thread.setSettings(settings)
        
        if self._p['Global variables','Waitforupdate?'] == 1:
            self.thread.setWaitforupdate(True)
        else:
            self.thread.setWaitforupdate(False)
        
        self.thread.start()
        
        self.d2.setEnabled(False)
        
        self.runBtn.setText("Stop")
        self.runBtn.setStyleSheet("background-color: pink")
    
    def _done(self):
        self.d2.setEnabled(True)
        self.runBtn.setText("Run")
        self.runBtn.setStyleSheet("background-color: lightgreen")
        self.statusbar.showMessage("done.")
        
        self._p.param('File savings','Save files?').setValue(2)
     
    def _save(self):
        
        #state = area.saveState()
        #restoreBtn.setEnabled(True)
        #print('save button')
        #self.d2.setEnabled(False)
        filename = 'settings.json'
        filename = self._saveFileDialog()
        #print('--------')
        vd = self._p.getValues()
        # for key, value in vd.items():
            # print('key:', key)
            # print('len:', len(value))
            # print('value:', value)
            # print('parent:', value[0])
            # d = value[1]
            # for skey, svalue in d.items():
                # print(skey,svalue)
            # print('--')
            
            
        with open(filename, 'w') as fp:
            json.dump(vd, fp, indent=4)
            self.statusbar.showMessage("settings saved to %s" % filename)
            
                
    def _load(self):
        
        #area.restoreState(state)
        #print('load button')
        filename = 'settings.json'
        filename = self._openFileDialog()
        
        #print('---JSON---')
        with open(filename, 'r') as fp:
            try:
                data = json.load(fp, object_pairs_hook=OrderedDict)
                # more compact code, but rely on the ordering of the data stored
                for key, value in data.items():
                    if value[1] is not None:
                        for sk, sv in value[1].items():
                            self._p.param(key,sk).setValue(sv[0])
                            if sv[1] is not None:
                                for ssk, ssv in sv[1].items():
                                    self._p.param(key,sk,ssk).setValue(ssv[0])
                self.statusbar.showMessage("settings loaded from %s" % filename)
            except:
                self.statusbar.showMessage("FAILED to load settings from %s" % filename)
                
        
        
        
def main ():
    dcam_bjarne.apply()
    dstat = Dcamapi.init()
    
    if dstat is False:
        print('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))
        
    app = pg.Qt.QtWidgets.QApplication(sys.argv)
    main = MainWindow(dcamapistat = dstat)
    main.show()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()


