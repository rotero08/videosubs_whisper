import argparse
from os import walk
from config import DEFAULT
from transformation import Transformations as tp

class MainContainer:

    def __init__(self) -> None:
        self.parser = self.init_parser()
    
    def init_parser(self):
        parser = argparse.ArgumentParser(description = "Whisper aplication to sub video")
        
        parser.add_argument(
            '-f','--folder',
            type = str,
            nargs = 1,
            required = True,
        )

        parser.add_argument(
            '-m','--model',
            type = str,
            nargs = 1,
            required = False,
            default = DEFAULT.MODEL_DEF
        )
        
        parser.add_argument(
            '-s','--subs',
            type = str,
            nargs = '+',
            required = False,
            default = DEFAULT.SUBS_LANGUAGE_DEF
        )

        parser.add_argument(
            '-du','--dubs',
            type = str,
            nargs = 1,
            required = False,
            default = DEFAULT.DUBS_LANGUAGE_DEF
        )

        parser.add_argument(
            '-de','--device',
            type = str,
            nargs = 1,
            required = False,
            default = DEFAULT.DEVICE_DEF
        )

        parser.add_argument(
            '-d','--delete',
            action='store_true',
            required = False
        )

        parser.add_argument(
            '-o','--overwrite',
            action='store_true',
            required = False
        )

        return parser
    
    def run_container(self):
        dir = getattr(self.parser.parse_known_args()[0], "folder")[0]
        self.videos = []
        for (dir, _ , file_names) in walk(dir):
            self.videos += list(map(lambda file: dir+ "/" + file if any(file.endswith(ext) for ext in DEFAULT.video_types) else '', file_names))
            self.videos = list(filter(('').__ne__, self.videos))

        print(self.videos)
        
        for video in self.videos:
            tp.process_video(self.parser, video)
        return