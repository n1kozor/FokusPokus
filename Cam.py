import os
import subprocess
import time
import psutil


class Camera:
    captureCommand = "capture "

    def __init__(self, exeDir=r"C:\Program Files (x86)\digiCamControl", verbose=True):
        if os.path.exists(exeDir + r"\CameraControlRemoteCmd.exe"):
            self.exeDir = exeDir
            self.verbose = verbose

            if not self.isProgramRunning():
                self.openProgram()
                time.sleep(10)
        else:
            print("Error: digiCamControl not found.")

    def openProgram(self):
        subprocess.Popen(self.exeDir + r"\CameraControl.exe")

    def isProgramRunning(self):
        for process in psutil.process_iter():
            if process.name() == "CameraControl.exe":
                return True
        return False

    def capture(self, location=""):
        r = self.runCmd(self.captureCommand + " " + location)
        if r == 0:
            print("Captured image.")

        return self.__getCmd("lastcaptured")

    def setFolder(self, folder: str):
        self.__setCmd("session.folder", folder)

    def setImageName(self, name: str):
        self.__setCmd("session.name", name)

    def setCounter(self, counter=0):
        self.__setCmd("session.Counter", str(counter))

    def setTransfer(self, location: str):
        return self.runCmd("set transfer %s" % (location))
        print("The pictures will be saved to %s." % location)

    def showLiveView(self):
        self.runCmd("do LiveViewWnd_Show")
        print("Showing live view window.")

    def setAutofocus(self, status):
        if status == True:
            self.captureCommand = "Capture"
            print("Autofocus is on.")
        else:
            self.captureCommand = "CaptureNoAf"
            print("Autofocus is off.")

    def setShutterspeed(self, shutterspeed: str):
        return self.__setCmd("shutterspeed", shutterspeed)

    def getCamerainfo(self):
        return self.__getCmd("camera")

    def getShutterspeed(self):
        return self.__getCmd("shutterspeed")

    def listShutterspeed(self):
        return self.__listCmd("shutterspeed")

    def listCamspecs(self):
        return self.__listCmd("session")

    def setIso(self, iso: int):
        return self.__setCmd("Iso", str(iso))

    def getIso(self):
        return self.__getCmd("Iso")

    def listIso(self):
        return self.__listCmd("Iso")

    def listCamsettings(self):
        return self.__listCmd("camera")

    def setAperture(self, aperture: float):
        return self.__setCmd("aperture", str(aperture))

    def getAperture(self):
        return self.__getCmd("aperture")

    def listAperture(self):
        return self.__listCmd("aperture")

    def setExposureComp(self, ec: str):
        return self.__setCmd("exposurecompensation", ec)

    def getExposureComp(self):
        return self.__getCmd("exposurecompensation")

    def listExposureComp(self):
        return self.__listCmd("exposurecompensation")

    def setCompression(self, comp: str):
        return self.__setCmd("compressionsetting", comp)

    def getCompression(self):
        return self.__getCmd("compressionsetting")

    def listCompression(self):
        return self.__listCmd("compressionsetting")

    def setWhitebalance(self, wb: str):
        return self.__setCmd("whitebalance", wb)

    def getWhitebalance(self):
        return self.__getCmd("whitebalance")

    def listWhitebalance(self):
        return self.__listCmd("whitebalance")

    def runCmd(self, cmd: str):
        r = subprocess.check_output("cd %s && CameraControlRemoteCmd.exe /c %s" % (self.exeDir, cmd),
                                    shell=True).decode()
        if 'null' in r:
            return 0
        elif r'""' in r:
            return 0
        else:
            print("Error: %s" % r)
            return -1

    def __setCmd(self, cmd: str, value: str):
        r = subprocess.check_output("cd %s && CameraControlRemoteCmd.exe /c set %s" % (self.exeDir, cmd + " " + value),
                                    shell=True).decode()
        if 'null' in r:
            print("Set the %s to %s" % (cmd, value))
            return 0
        else:
            print("Error: %s" % r[109:])
            return -1

    def __getCmd(self, cmd: str):
        r = subprocess.check_output("cd %s && CameraControlRemoteCmd.exe /c get %s" % (self.exeDir, cmd),
                                    shell=True).decode()
        if 'Unknown parameter' in r:
            print("Error: %s" % r[109:])
            return -1
        else:
            returnValue = r[96:-6]
            print("Current %s: %s" % (cmd, returnValue))
            return returnValue

    def __listCmd(self, cmd: str):
        r = subprocess.check_output("cd %s && CameraControlRemoteCmd.exe /c list %s" % (self.exeDir, cmd),
                                    shell=True).decode()
        if 'Unknown parameter' in r:
            print("Error: %s" % r[109:])
            return -1
        else:
            returnList = r[96:-6].split(",")
            returnList = [e[1:-1] for e in returnList]
            print("List of all possible %ss: %s" % (cmd, returnList))
            return returnList


if __name__ == '__main__':
    print("Beginning unit tests:")

    camera = Camera()

    assert isinstance(camera.listShutterspeed(), list)
    temp = camera.listShutterspeed()[0]
    assert camera.setShutterspeed(temp) == 0
    assert camera.getShutterspeed() == temp

    assert isinstance(camera.listIso(), list)
    temp = camera.listIso()[0]
    assert camera.setIso(temp) == 0
    assert camera.getIso() == temp

    assert isinstance(camera.listAperture(), list)
    temp = camera.listAperture()[0]
    assert camera.setAperture(temp) == 0
    assert camera.getAperture() == temp

    assert isinstance(camera.listExposureComp(), list)
    temp = camera.listExposureComp()[0]
    assert camera.setExposureComp(temp) == 0
    assert camera.getExposureComp() == temp

    assert isinstance(camera.listCompression(), list)
    temp = camera.listCompression()[0]
    assert camera.setCompression(temp) == 0
    assert camera.getCompression() == temp

    assert isinstance(camera.listWhitebalance(), list)
    temp = camera.listWhitebalance()[0]
    assert camera.setWhitebalance(temp) == 0
    assert camera.getWhitebalance() == temp

    print("End unit tests.")
