"""
 #####################################################################################
                 HOME WORK 5
 This code is used to perform basic image manipulation tasks like 
 saving the image and  merging the image for later visualization after training
 It combines the output of 8 images as one image for better visualization
 
             It also contains a plotting function for question 4,5, and 6
             after the training phase is completed all the outputs are saved 
             into a result folder and this function reads those files and 
             show the output
                 THANK YOU!
 #####################################################################################

"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from scipy.misc import imsave
from scipy.misc import imresize


class Plot_Reproduce_Performance():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28, resize_factor=1.0):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = imresize(image, size=(w_,h_), interp='bicubic')

            img[j*h_:j*h_+h_, i*w_:i*w_+w_] = image_

        return img

class Plot_Manifold_Learning_Result():
    def __init__(self, DIR, n_img_x=20, n_img_y=20, img_w=28, img_h=28, resize_factor=1.0, z_range=4):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

        assert z_range > 0
        self.z_range = z_range

        self._set_latent_vectors()

    def _set_latent_vectors(self):

        z = np.rollaxis(np.mgrid[self.z_range:-self.z_range:self.n_img_y * 1j, self.z_range:-self.z_range:self.n_img_x * 1j], 0, 3)
        self.z = z.reshape([-1, 2])

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = imresize(image, size=(w_, h_), interp='bicubic')

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

        return img

    # borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
    def save_scattered_image(self, z, id, name='scattered_image.jpg'):
        N = 10
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        axes = plt.gca()
        axes.set_xlim([-self.z_range-2, self.z_range+2])
        axes.set_ylim([-self.z_range-2, self.z_range+2])
        plt.grid(True)
        plt.savefig(self.DIR + "/" + name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
"""
######################################################
plot the answer for question 4, 5, and 6
######################################################
"""
def plot_the_result():
    """
    ##########################################################################
    Plot the answer to question 4, 5, and 6 by reading the output of the model
    from the Result folder which is created at runtime
    ###########################################################################
    """
    input_image=mpimg.imread('results/input.jpg')
    plt.imshow(input_image)
    plt.title("Original Image with noise")
    plt.show()
    
    """
    After noise is added
    """
    noise_image=mpimg.imread('results/input_noise.jpg')
    plt.imshow(noise_image)
    plt.title("Input Image with noise")
    plt.show()
    
    """
    Plot Reconstratected image after the final iteration
    """
    
    reconstracted_image=mpimg.imread('results/VAE_epoch_09.jpg')
    plt.imshow(reconstracted_image)
    plt.title("Reconstratected image")
    plt.show()
    
    """
    Plot two components from your latent variable after the final iteration (question 5)
    """
    
    reconstracted_image=mpimg.imread('results/VAE_twocomponent.jpg')
    plt.imshow(reconstracted_image)
    plt.title("Two Dimensional Latent space")
    plt.show()
    
    """
    Plot N(0,1) as latent variables and use the decoder to generate some images (Question 6)
    """
    
    reconstracted_image=mpimg.imread('results/randomly_generated_decoded_image.jpg')
    plt.imshow(reconstracted_image)
    plt.title("Generated Image")
    plt.show()
    