""" 
#####################################################################################
                 ANSWER TO HOME WORK_5
 This code Consists of the implementation of Variational auto encoder for EMNIST 
               Letter Reconstruction. 
 
      my_variationa_autoencoder => the structure of the network
      preprocess_emnist => code to preprocess the image data
      plot_and_merge => code to merge and save the final outputs
                 THANK YOU!
 #####################################################################################
"""
import numpy as np
import tensorflow as tf
import data_prepro as emnist
import matplotlib.pyplot as plt 
import my_variational_auto_encoder
import plot_and_merge as plot_utils


IMAGE_SIZE = 28


def begin_vae_training():

    """
    ###################################################################
    Initialization of basic training parameters
    ###################################################################
    """
    n_hidden,dim_z, dim_img = 500,20,28**2
    # train
    n_epochs = 110
    batch_size = 128
    learn_rate = 1E-3
    RESULTS_DIR = 'RESULTS'

    
    # image resizing factor for plotting the output to a file 
    VAE,VAE_n_img_x,VAE_n_img_y , VAE_resize_factor= True,8,8,1.0
    #plot learned result
    result_image,result_image_n_img_x,result_image_n_img_y,result_image_resize_factor,result_image_z_range,result_image_n_samples=False,8,8,1,2,5000
    """
    #####################################################################################
    Prepare the EMNIST Dataset for Training by calling the preprocess function
                        THEN
            Buidl the Tensor Graph
    #######################################################################################
    """
    train_total_data, train_size, _, _, test_data, test_labels = emnist.preprocess()
    n_samples = train_size

    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')

    # dropout
    """
    #######################################################################################
    ADD DROPOUT For Regularization
    #######################################################################################
    """
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for result_image
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

    # network architecture
    """
    ######################################################################################
    Make a call to the variational auto encoder
    #####################################################################################
    """
    y, z, loss, neg_marginal_likelihood, KL_divergence = my_variational_auto_encoder.autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    """ training """
    
    print("Training in progress please wait until it completes")
    # Plot for reproduce performance
    if VAE:
        VAE = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, VAE_n_img_x, VAE_n_img_y, IMAGE_SIZE, IMAGE_SIZE, VAE_resize_factor)

        x_VAE = test_data[0:VAE.n_tot_imgs, :]

        x_VAE_img = x_VAE.reshape(VAE.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
        VAE.save_images(x_VAE_img, name='input.jpg')
        #ADD Gaussian noise to the image
        x_VAE = x_VAE * np.random.randint(2, size=x_VAE.shape)
        x_VAE += np.random.randint(2, size=x_VAE.shape)

        x_VAE_img = x_VAE.reshape(VAE.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
        VAE.save_images(x_VAE_img, name='input_noise.jpg')

    # Plot for manifold learning result
    if result_image and dim_z == 2:

        result_image = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, result_image_n_img_x, result_image_n_img_y, IMAGE_SIZE, IMAGE_SIZE, result_image_resize_factor, result_image_z_range)

        x_result_image = test_data[0:result_image_n_samples, :]
        id_result_image = test_labels[0:result_image_n_samples, :]

       
        x_result_image = x_result_image * np.random.randint(2, size=x_result_image.shape)
        x_result_image += np.random.randint(2, size=x_result_image.shape)
        decoded = my_variational_auto_encoder.decoder(z_in, dim_img, n_hidden)

    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

        for epoch in range(n_epochs):

            # Random shuffling
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-emnist.NUM_LABELS]

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]

                batch_xs_target = batch_xs_input

                # add salt & pepper noise
                batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob : 0.9})
            temp=-tot_loss/100128
            # print cost every epoch
            
            print("Iteration %d: total loss %5.5f" % (epoch, temp))
            
            loss_history.append(temp)
            # if minimum loss is updated or final epoch, plot results
            if min_tot_loss > tot_loss or epoch+1 == n_epochs:
                min_tot_loss = tot_loss
                # Plot for reproduce performance
                if VAE:
                    y_VAE = sess.run(y, feed_dict={x_hat: x_VAE, keep_prob : 1})
                    y_VAE_img = y_VAE.reshape(VAE.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
                    VAE.save_images(y_VAE_img, name="/VAE_epoch_%02d" %(epoch) + ".jpg")

                # Plot for manifold learning result
                if result_image and dim_z == 2:
                    y_result_image = sess.run(decoded, feed_dict={z_in: result_image.z, keep_prob : 1})
                    y_result_image_img = y_result_image.reshape(result_image.n_tot_imgs, IMAGE_SIZE, IMAGE_SIZE)
                    result_image.save_images(y_result_image_img, name="/result_image_epoch_%02d" % (epoch) + ".jpg")

                    # plot distribution of labeled images
                    z_result_image = sess.run(z, feed_dict={x_hat: x_result_image, keep_prob : 1})
                    result_image.save_scattered_image(z_result_image,id_result_image, name="/result_image_map_epoch_%02d" % (epoch) + ".jpg")
def plot_learning_curve():
    plt.plot(loss_history,label='train')
    plt.legend(loc='best')
    plt.xlabel("epochs")
    plt.ylabel("loss bound")
    plt.title("Learning Curve")
    plt.grid()
    plt.show()
  
    
if __name__ == '__main__':
    
    """
    #################################################################################################
     Run the next line of codes to train VAE and plot the answer for question Q3, Q4,Q6
    #################################################################################################
    """
    loss_history=list()
    begin_vae_training()
    #plot the learning curve
    plot_learning_curve()
    plot_utils.plot_the_result()    
    
    """
    ############################################################################################
          ANSWER TO QUESTION 5
          two components latent variable and plot OF reconstructed images by
          varying the value of the latent variable.
          SELECT AND RUN THE FOLLOWING IMPORT STATETEMENT
          THANKS!
    ############################################################################################
    """
    import answer_question_5 as q
    q.begin_training()
    """
    ############################################################################################
           END OF THE CODE
           THANK YOU!
    ###########################################################################################
    """
 
    
    