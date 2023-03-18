from torch.cuda import is_available 

class Default:
    #default values
    MODEL_DEF = "medium"
    DUBS_LANGUAGE_DEF = "english"
    SUBS_LANGUAGE_DEF = "english"
    DEVICE_DEF = "cuda:0" if is_available() else "cpu"


video_types = ['.mkv', '.mp4']
DEFAULT = Default()