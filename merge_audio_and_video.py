from moviepy import VideoFileClip, AudioFileClip


# import audio and video files
video = "null_potential_path.mp4"
audio = "null_potential.wav"

# Let's deaccelerate the video so that it matches the audio length. First compute audio and video lengths
audioclip = AudioFileClip(audio)
audioclip_duration = audioclip.duration
videoclip = VideoFileClip(video)
videoclip_duration = videoclip.duration
print("audio duration: ", audioclip_duration)
print("video duration: ", videoclip_duration)

# compute decceleration factor
deceleration_factor = videoclip_duration / audioclip_duration
print("deceleration factor: ", deceleration_factor)

# Let's daccelerate the video by the deceleration factor
modified_clip1 = videoclip.time_transform(lambda t: t * deceleration_factor)

# set the audio of the modified video to be the audio file
modified_clip2 = modified_clip1.with_audio(audioclip)

# write the result to a file
modified_clip2.with_duration(audioclip_duration).write_videofile("null_potential_final_video.mp4")




