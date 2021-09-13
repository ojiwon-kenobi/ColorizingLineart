#This module assumes you have a folder full of images to train on.
# What this module will do is:
#   * quantize colors (reduce number of colors in an image)
#       * blur the quantized image if you like using bilateral gaussian blur to preserve edges
#   * generate lineart/ digitalart and put that in a new folder for you.
#       * be aware that there are a lot of different lineart styles which may impact the results of the outcome


