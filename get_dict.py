import os
import re
d = {}
for local_video_path in os.listdir("vids"):
    d[local_video_path[5:-24]] = ""

print(list(d))