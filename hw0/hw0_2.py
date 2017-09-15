import sys
import math
from PIL import Image

old_im = Image.open(sys.argv[1])
new_im = Image.new('RGB', old_im.size)
output_name = 'Q2.jpg'

width, height = old_im.size
for w in range(width):
	for h in range(height):
		r,g,b = old_im.getpixel((w,h))
		new_im.putpixel((w,h), (math.floor(r/2), math.floor(g/2), math.floor(b/2)))
new_im.save(output_name)
