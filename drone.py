import cv2
import cvzone
from djitellopy import tello
import KeypressModule as kp

thres = 0.56
nmsThres = 0.2

className = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    className = f.read().split('\n')


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

me.streamoff()
me.streamon()
print(me.get_temperature())
# me.set_speed()
def getKeyBordeInpute():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("q"): ud = -speed
    elif kp.getKey("e"): ud = speed

    if kp.getKey("w"): fb = speed
    elif kp.getKey("s"): fb = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    if kp.getKey("t"):  me.takeoff()
    elif kp.getKey("l"): me.land()

    if kp.getKey("f"): me.flip_left()
    elif kp.getKey("g"): me.flip_right()

    if kp.getKey("z"): me.flip_forward()
    elif kp.getKey("c"): me.flip_back()

    # if kp.getKey("x"): me.rotate_clockwise(-speed)
    # elif kp.getKey("v"): me.rotate_counter_clockwise(speed)

    return [lr, fb, ud, yv]


while True:
    img = me.get_frame_read().frame
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nmsThres)

    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            print(classId, conf, box)
            cvzone.cornerRect(img, box, rt=3)
            cv2.putText(img, f'{className[classId - 1].upper()} {round(conf * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2)
            cvzone.cornerRect(img, box)
    except:
        pass

    valus = getKeyBordeInpute()
    print(me.get_battery())
    me.send_rc_control(valus [0], valus[1], valus[2], valus[3])

    cv2.imshow("Image",img)
    cv2.waitKey(1)


