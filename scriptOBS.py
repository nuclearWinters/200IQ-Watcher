import obspython as obs
import time
import cv2

source_name = ""

# ------------------------------------------------------------


def refresh_pressed(props, prop):
    global num
    num = 0
    global video
    video = cv2.VideoCapture(1)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
    ret, frame = video.read()
    print(type(frame))
    cv2.imwrite("C:/Users/Fernandoo/Desktop/prueba.png", frame)
    """
    Called when the 'refresh' button defined below is pressed
    """
    print("Refresh Pressed!")
    obs.timer_add(update_text, 1000)


def update_text():
    global source_name
    global num
    global video
    ret, frame = video.read()
    print(type(frame))
    num += 1
    print(num)
    if num == 2:
        obs.timer_remove(update_text)
        video.release()

# ------------------------------------------------------------


def script_properties():
    """
    Called to define user properties associated with the script. These
    properties are used to define how to show settings properties to a user.
    """
    props = obs.obs_properties_create()
    obs.obs_properties_add_button(props, "button", "Start", refresh_pressed)
    return props


def script_update(settings):
    """
    Called when the script’s settings (if any) have been changed by the user.
    """
    global source_name

    source_name = obs.obs_data_get_string(settings, "source")