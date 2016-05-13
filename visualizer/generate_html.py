import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import misc
import Image
import os
import sys
import copy
import argparse
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-attenvecs', help="File containing attention vectors",
                        type=string)
    parser.add_argument('-imagedir', help="Directory in which images are located",
                        type=string)
 
    args = parser.parse_args(arguments)
    attenvecs_file = args.attenvecs
    imagedir = args.imagedir
    
    if imagedir[-1] == '/':
        imagedir = imagedir[:-1]
    
    
    page_head = '<!DOCTYPE HTML><head></head><body>'
    page_end = '</body></html>'

    f = h5py.File(attenvecs_file, "r")

    attns_vectors = np.array(f['attns_vectors'])

    num_images = attns_vectors.shape[0]

    index_page = page_head

    if not os.path.exists('website'):
        os.makedirs('website')

    if not os.path.exists('website/images'):
        os.makedirs('website/images')

    if not os.path.exists('website/pages'):
        os.makedirs('website/pages')


    for image_num, line in enumerate(f):
        index_page += '<a href="%08d.html">Image %d</a><br />' % (image_num + 1, image_num + 1)

        webpage = page_head

        words = line.strip().split(' ')


        for word_num, word in enumerate(words):
            attention_vec = attns_vectors[image_num, word_num, :]
            attention_vec = attention_vec.reshape(14,14)

            img = plt.imread('%s/%08d.png' % (image_dir, image_num + 1))

            attention_vec = resize(attention_vec, (img.shape[0], img.shape[1]))


            scaleing = MinMaxScaler(feature_range=(0.1, 1), copy=True)
            attention_vec = scaleing.fit_transform(attention_vec)


            atten_img = copy.deepcopy(img)
            for channel in range(3):
                atten_img[:,:,channel] = np.multiply(atten_img[:,:,channel], attention_vec)
            plt.imsave('website/images/%08d_%04d.png' % (image_num + 1, word_num), atten_img)


            webpage += '%s<br /><img src=../images/%08d_%04d.png /><br /><br />' % (word, image_num + 1, word_num)

        webpage += page_end

        wpf = open('website/pages/%08d.html' % (image_num + 1), 'w')
        wpf.write(webpage)
        wpf.close()

    index_page += page_end
    ipf = open('website/pages/index.html', 'w')
    ipf.write(index_page)
    ipf.close()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))