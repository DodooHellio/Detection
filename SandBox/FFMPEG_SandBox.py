import ffmpeg

stream = ffmpeg.input('wake_video1.mp4')
stream = ffmpeg.hflip(stream)
stream = ffmpeg.output(stream, 'output.mp4')
ffmpeg.run(stream)