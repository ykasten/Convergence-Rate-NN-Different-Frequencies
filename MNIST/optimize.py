'''
Written by Shira Kritchman
AUG18


Connecting and configuring the queue
-------
ssh -X mcluster01
qlogin -q gpuint.q
setenv PATH /usr/wisdom/python3/bin:$PATH
setenv PYTHONPATH /usr/wisdom/python3
setenv LD_LIBRARY_PATH /usr/local/cudnn-v6/lib64

'''

# from __future__ import print_function
#
import tensorflow as tf
# #import model
import numpy as np
import os
import numpy.random as rn
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

print('\n*** Importing optimize.py')

def optimize(cfg):

    #################################
    '''      configurations       '''
    #################################

    print('\n\n*** Configuring')
    print('\nConfigs:\n', cfg)
    # choosing GPU device
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5'
    session_config = tf.ConfigProto()
#     session_config.gpu_options.visible_device_list = "3" # which GPUs to use

    # data parameters
    n_mnist_train_and_val = 60000
    dim = 28*28 # MNIST data input (img shape: 28*28=784)
    n_classes = 10 # MNIST total classes (0-9 digits)
    n_val = cfg['n_val']
    pi = cfg['pi']

    # network
    k = cfg['k']
    alpha = cfg['alpha']
    depth = cfg['depth']

    # optimization
    eta = cfg['eta']
    batch_size = cfg['batch_size']
    break_thresh = cfg['break_thresh']
    training_epochs = cfg['training_epochs']
    beta = cfg['beta']

    # meta
    test_name = cfg['test_name']
    save_step = cfg['save_step']
    print_step = cfg['print_step']

    # paths
    data_path = 'data/'
    res_path = 'results/'
    res_name = res_path + test_name

    # read and prepare the data
    print('\n*** Preparing data\n')
    n_train = n_mnist_train_and_val - n_val
    n_train_rand = int(np.floor(pi / 100 * n_train))
    n_train_true = n_train - n_train_rand
    n_val_rand = int(np.floor(pi / 100 * n_val))
    mnist = input_data.read_data_sets(data_path, one_hot=True, \
        validation_size = n_val, seed = 777)
    # split the training samples to true and random
    train_perm = rn.permutation(n_train)
    I_train_rand = train_perm[:n_train_rand]
    I_train_true = train_perm[n_train_rand:]
    # change train random labels to random
    if n_train_rand > 0:
        print('\n*** Sampling random labels for I_train_rand')
        print('I_train_rand.shape = ', I_train_rand.shape)
        print('I_train_true.shape = ', I_train_true.shape)
        print('mnist.train.labels[I_train_rand,:].shape = ', mnist.train.labels[I_train_rand,:].shape)
        mnist.train.labels[I_train_rand,:] = np.apply_along_axis(rn.permutation, 1, mnist.train.labels[I_train_rand,:])
    mnist_train_images = mnist.train.images.copy()
    mnist_train_labels = mnist.train.labels.copy()
    # split the validation samples to true and random
    n_val = mnist.validation.labels.shape[0]
    val_perm = rn.permutation(n_val)
    I_val_rand = val_perm[:n_val_rand]
    #change val rand labels to random
    if n_val_rand > 0:
        print('\n*** Sampling random labels for I_val_rand')
        mnist.validation.labels[I_val_rand] = np.apply_along_axis(rn.permutation, 1, mnist.validation.labels[I_val_rand,])

    print('\n*** Configured according to', cfg, ', with ', \
       n_train_true, 'true labels and', n_train_rand,\
       'random labels')

    # arrays for holding results
    n_acc = training_epochs + 1
    train_acc_list = []
    train_true_acc_list = []
    train_rand_acc_list = []
    test_acc_list = []
    val_acc_list = []
    avg_cost_list = []
    epoch_num_list = []
    w_learned_list = []
    b_learned_list = []

    if batch_size == 0:
        batch_size = n_train
    if (n_train % batch_size != 0):
        print("\n*** Warning! batch size doesn't divide n_train *** \n")
        input("Press enter to continue")
    total_batch = int(n_train/batch_size)

    print('\n*** Building Computation Graph')

    # tf Graph Input
    # mnist data image of shape 28*28=784
    x = tf.placeholder(tf.float32, [None, dim], name='InputData')
    # 0-9 digits recognition => 10 classes
    y = tf.placeholder(tf.float32, [None, n_classes], name='LabelData')

    weights = {}
    biases = {}

    def leaky_relu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

    def multilayer_perceptron(x, weights, biases, alpha):
        layer = x
        n_curr = dim
        for i in range(depth):
            with tf.name_scope('layer'+str(i)):
                w_name = 'w'+str(i)
                b_name = 'b'+str(i)
                weights[w_name] = tf.Variable(tf.random_normal([n_curr, k], stddev=.01), name=w_name)
                biases[b_name] = tf.Variable(tf.random_uniform([k], -1, 1), name=b_name)
                n_curr = k
                layer = tf.add(tf.matmul(layer, weights[w_name]), biases[b_name])
                layer = leaky_relu(layer, alpha)
        # Output layer
        with tf.name_scope('output_layer'):
            weights['w_out'] = tf.Variable(tf.random_normal([n_curr, n_classes], stddev=.01), name='w_out')
            out_layer = tf.matmul(layer, weights['w_out'])
        return out_layer

    # Encapsulating all ops into scopes, making Tensorboard's Graph
    # Visualization more convenient
    with tf.name_scope('Model'):
        # Build model
        pred = multilayer_perceptron(x, weights, biases, alpha)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        # l2 loss on weights
        regularizers = 0
        for i in range(depth):
            w_name = 'w'+str(i)
            regularizers += tf.nn.l2_loss(weights[w_name])
        loss += beta * regularizers

    with tf.name_scope('SGD'):
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(eta)
        # Op to calculate every variable gradient
        grads = tf.gradients(loss, tf.trainable_variables())
        grads_and_vars = list(zip(grads, tf.trainable_variables()))
        # Op to update all variables according to their gradient
        apply_grads = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

    with tf.name_scope('Accuracy'):
        # Accuracy
        acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session(config=session_config) as sess:

        print('\n*** Sess init')
        # Run the initializer
        sess.run(init)

        epoch = 0

        print('\n*** Training')
        # Training cycle
        for epoch in range(training_epochs):

            avg_cost = 0.
            epoch_perm = rn.permutation(n_train)

            first_step_in_epoch = True

            for batch_ind in range(total_batch):

                #save_path = saver.save(sess, ckpt_name)
                if cfg['save_sbs'] or first_step_in_epoch: #and epoch < lim_save_sbs
                    train_acc_val = acc.eval({x: mnist.train.images, y: mnist.train.labels})
                    train_true_acc_val = acc.eval({x: mnist_train_images[I_train_true,:], y: mnist_train_labels[I_train_true,:]})
                    train_rand_acc_val  = acc.eval({x: mnist_train_images[I_train_rand,:], y: mnist_train_labels[I_train_rand,:]})
                    test_acc_val = acc.eval({x: mnist.test.images, y: mnist.test.labels})
                    val_acc_val = acc.eval({x: mnist.validation.images, y: mnist.validation.labels})
                    train_acc_list.append(train_acc_val)
                    train_true_acc_list.append(train_true_acc_val)
                    train_rand_acc_list.append(train_rand_acc_val)
                    test_acc_list.append(test_acc_val)
                    val_acc_list.append(val_acc_val)
                    epoch_num_list.append(epoch)

                    w_last = []
                    b_last = []
                    for i_depth in range(depth):
                        w_name = 'w' + str(i_depth)
                        b_name = 'b' + str(i_depth)
                        w_last.append(np.array(weights[w_name].eval()))
                        b_last.append(np.array(biases[b_name].eval()))
                    w_last.append(np.array(weights['w_out'].eval()))

                    if cfg['save_weights']:
                        w_learned_list.append(w_last)
                        b_learned_list.append(b_last)

                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop), cost op (to get loss value)
                _, loss_on_batch = sess.run([apply_grads, loss],
                                            feed_dict={x: batch_xs, y: batch_ys})

                # Compute average loss
                avg_cost += loss_on_batch / total_batch

                first_step_in_epoch = False

            avg_cost_list.append(avg_cost)

            stopping = (train_acc_val >= break_thresh)

            if (epoch % print_step == 0) or (epoch < 50) or stopping:
                print('\n\nEpoch: {}'.format(epoch))
                print('Before training on epoch, Train acc:         {:.3f}'.format(train_acc_val))
                print('                          Train true acc:    {:.3f}'.format(train_true_acc_val))
                print('                          Train rand acc:    {:.3f}'.format(train_rand_acc_val))
                print('                          Test acc:          {:.3f}'.format(test_acc_val))
                print('                          Val acc:           {:.3f}'.format(val_acc_val))
                print('While training,           Average acc:       {:.9f} ({:.3f})'.format(avg_cost, np.exp(-avg_cost)))

            if (epoch % save_step == 0) or (epoch < 10) or stopping:
                print('\n*** Saving... ', end='')
                ind_try = 0
                while True :
                    ind_try+=1
                    try :
                        np.savez(res_name,
#                            avg_costs=avg_cost_list,
#                            w_learned_last=w_last,
#                            b_learned_last=b_last,
                            train_acc=train_acc_list,
                            # train_true_acc=train_true_acc_list,
                            # train_rand_acc=train_rand_acc_list,
                            test_acc=test_acc_list,
                            val_acc=val_acc_list,
                            cfg=cfg)
                        print('done!')
                        break
                    except PermissionError :
                        print('\n#'*20, end='')
                        print('\n<<< Saving attempt {} failed, trying again >>> \n'.format(ind_try))
                    except KeyboardInterrupt :
                        print('\n<<< Simulation interrupted, cannot save')
                        stopping = True
                        break

            if stopping:
                print('\n*** Epoch: {}\n    Training reached {} threshold and is stopping'.format(epoch, break_thresh))
                break

        print('\n*** Optimization Finished!')
        print('\n*** Configured according to', cfg, ', with ', \
            n_train_true, 'true labels and', n_train_rand,\
            'random labels')    # Test model
        # Calculate accuracy
        print('\n\nTotal Epochs: {}'.format(epoch))
        print('Before training on epoch, Train acc:          {:.3f}'.format(train_acc_val))
        print('                          Train true acc:     {:.3f}'.format(train_true_acc_val))
        print('                          Train rand acc:     {:.3f}'.format(train_rand_acc_val))
        print('                          Test acc:           {:.3f}'.format(test_acc_val))
        print('                          Val acc:            {:.3f}'.format(val_acc_val))
        print('While training,           Average cost (acc): {:.9f} ({:.3f})'.format(avg_cost, np.exp(-avg_cost)))

        return
