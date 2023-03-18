import ffmpeg
import whisper
import os
import math
import moviepy.editor as mp
import datetime


class Transformations:

    @staticmethod
    def extract_audio(video_file: str):
        audio_file = f'{video_file.replace(".mkv" , "")}.mp3'
        video = mp.VideoFileClip(video_file)
        video.audio.write_audiofile(audio_file)
        return audio_file

    @staticmethod
    def audio_transcribe(audio: str, use_model: str):
        model = whisper.load_model(use_model)
        result = model.transcribe(audio)
        segments = result["segments"]
        subs_file = f'{audio.replace(".mp3" , "")}.srt'
        return [segments, subs_file]

    @staticmethod
    def text_translate(segments: dict, language: str): #########################change
        return segments
    
    @staticmethod
    def create_srt_file(data, output_filename, language):
        if language != 'english':
            data = Transformations.text_translate(data, language)

        with open(output_filename, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data):
                start = datetime.timedelta(seconds=item['start'])
                end = datetime.timedelta(seconds=item['end'])
                start_srt = Transformations.start_to_srt(start)
                end_srt = Transformations.start_to_srt(end)
                f.write(str(i+1) + '\n')
                f.write(start_srt + ' --> ' + end_srt + '\n')
                f.write(item['text'] + '\n')
                f.write('\n')

    
    @staticmethod
    def start_to_srt(start):
        hours, remainder = divmod(start.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = math.floor(start.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    
    @staticmethod
    def delete_audio(audio_file):
        os.remove(audio_file)
    
    @staticmethod
    def embedd(video_file: str, subs_files: dict):
        
        #Define input values
        input_ffmpeg = ffmpeg.input(video_file)
        input_subs = []
        metadata = {}
        i = 0
        for k in subs_files.keys():
            input_subs.append(ffmpeg.input(subs_files[k])['s'])
            metadata = {**metadata, **{f'metadata:s:s:{i}': f'title={k}'}}

        #Define output file
        input_video = input_ffmpeg['v']
        input_audio = input_ffmpeg['a']

        input = [input_video, input_audio] + input_subs
        output_ffmpeg = ffmpeg.output(
            *input,
            vcodec='copy', acodec='copy',
            **metadata
        )

        # If the destination file already exists, overwrite it.
        output_ffmpeg = ffmpeg.overwrite_output(output_ffmpeg)

        # Print the equivalent ffmpeg command we could run to perform the same action as above.
        print(ffmpeg.compile(output_ffmpeg))

        # Do it! transcode!
        ffmpeg.run(output_ffmpeg)
        return
    
    @staticmethod
    def process_video(parser, video):
        audio = Transformations.extract_audio(video)
        segments, subs_file = Transformations.audio_transcribe(audio, getattr(parser.parse_known_args()[0], "model")[0])
        Transformations.delete_audio(audio)

        subs_files = {}
        for language in getattr(parser.parse_known_args()[0], "subs"):
            #address = f'{language}_'+subs_file
            address = subs_file
            subs_files = {**subs_files, **{language: address}}
            Transformations.create_srt_file(segments, address, language)
        
        Transformations.embedd(video, subs_files)