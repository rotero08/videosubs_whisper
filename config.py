from torch.cuda import is_available 

class Default:
    #default values
    MODEL_DEF = "medium"
    DUBS_LANGUAGE_DEF = "english"
    SUBS_LANGUAGE_DEF = "english"
    DEVICE_DEF = "cuda:0" if is_available() else "cpu"
    video_types = ['webm', 'mkv', 'flv', 'vob', 'ogv', 'ogg', 'rrc', 'gifv', 'mng', 'mov', 'avi', 'qt', 'wmv', 'yuv', 'rm', 'asf', 'amv', 'mp4', 'm4p', 'm4v', 'mpg', 'mp2', 'mpeg', 'mpe', 'mpv', 'm4v', 'svi', '3gp', '3g2', 'mxf', 'roq', 'nsv', 'f4v', 'f4p', 'f4a', 'f4b', 'mod']

DEFAULT = Default()