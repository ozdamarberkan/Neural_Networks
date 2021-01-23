import sys

question = sys.argv[1]

def berkan_ozdamar_21602353_hw3(question):
    if question == '1' :
        ##question 1 code goes here
        # !/usr/bin/env python
        # coding: utf-8

        # In[1]:

        import numpy as np
        import h5py
        import matplotlib.pyplot as plt

        # In[2]:

        # Part A

        f = h5py.File('assign3_data1.h5', 'r')
        dataKeys = list(f.keys())
        print('The data keys are:' + str(dataKeys))

        # Gathering the  train images, test images, train labels and test labels.
        data = f['data']
        invXForm = f['invXForm']
        xForm = f['xForm']

        # data=np.array(data)
        # invXForm=np.array(invXForm)
        # xForm=np.array(xForm)

        # data = data.reshape(-1,16,16,3)

        print('The size of data is: ' + str(np.shape(data)))
        print('The size of invXForm is: ' + str(np.shape(invXForm)))
        print('The size of xForm is: ' + str(np.shape(xForm)))

        # In[3]:

        data_r = data[:, 0, :, :]
        data_g = data[:, 1, :, :]
        data_b = data[:, 2, :, :]

        data_grayscale = data_r * 0.2126 + data_g * 0.7152 + data_b * 0.0722
        print(np.shape(data_grayscale))

        # In[4]:

        def normalize_data(images):
            data_mean = np.mean(images, axis=(1, 2))
            for i in range(np.shape(data_mean)[0]):
                images[i, :, :] -= data_mean[i]
            return images

        # In[5]:

        def map_std(images):
            data_std = np.std(images)
            mapped_data = np.where(images > 3 * data_std, 3 * data_std, images)
            mapped_data_final = np.where(mapped_data < -3 * data_std, -3 * data_std, mapped_data)
            return mapped_data_final

        # In[6]:

        def clip_data_range(images, min_value, max_value):
            range_val = max_value - min_value
            max_data = np.max(images)
            min_data = np.min(images)

            result = images - min_data

            max_data = np.max(result)
            result = result / max_data * range_val

            result = result + min_value
            return result

        # In[7]:

        data_grayscale_norm = normalize_data(data_grayscale)
        data_grayscale_norm_mapped = map_std(data_grayscale_norm)
        data_final = clip_data_range(data_grayscale_norm_mapped, 0.1, 0.9)

        # In[8]:

        figureNum = 0
        plt.figure(figureNum, figsize=(18, 16))
        np.random.seed(9)
        sample_size = np.shape(data_final)[0]
        random_200 = np.random.randint(sample_size, size=(200))

        for i, value in enumerate(random_200):
            ax1 = plt.subplot(20, 10, i + 1)
            ax1.imshow(np.transpose(data[value], (1, 2, 0)))
            ax1.set_yticks([])
            ax1.set_xticks([])

        plt.show()

        # In[9]:

        figureNum += 1
        plt.figure(figureNum, figsize=(18, 16))

        for subplot, value in enumerate(random_200):
            ax2 = plt.subplot(20, 10, subplot + 1)
            ax2.imshow(data_final[value], cmap='gray')
            ax2.set_yticks([])
            ax2.set_xticks([])
        plt.show()

        # In[10]:

        # Part B

        def sigmoid(x):

            result = 1 / (1 + np.exp(-x))
            return result

        def der_sigmoid(x):

            result = sigmoid(x) * (1 - sigmoid(x))
            return result

        def forward(We, data):

            W1, B1, W2, B2 = We

            # HIDDEN LAYER
            A1 = data.dot(W1) + B1
            Z1 = sigmoid(A1)
            # OUTPUT LAYER
            A2 = Z1.dot(W2) + B2
            y_pred = sigmoid(A2)

            return A1, Z1, A2, y_pred

        def aeCost(We, data, params):
            Lin, Lhid, lambdaa, beta, rho = params
            W1, B1, W2, B2 = We
            sample_size = np.shape(data)[0]

            A1, Z1, A2, y_pred = forward(We, data)
            Z1_mean = np.mean(Z1, axis=0)

            J_1 = (1 / (2 * sample_size)) * np.sum(np.power((data - y_pred), 2))
            J_2 = (lambdaa / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
            KL_1 = rho * np.log(Z1_mean / rho)
            KL_2 = (1 - rho) * np.log((1 - Z1_mean) / (1 - rho))
            J_3 = beta * np.sum(KL_1 + KL_2)
            J = J_1 + J_2 - J_3

            del_out = -(data - y_pred) * der_sigmoid(y_pred)

            del_KL = beta * (-(rho / Z1_mean.T) + ((1 - rho) / (1 - Z1_mean.T)))
            del_KLs = np.vstack([del_KL] * sample_size)

            del_hidden = ((del_out.dot(W1)) + del_KLs) * der_sigmoid(Z1)

            # Gradients
            grad_W2 = (1 / sample_size) * (Z1.T.dot(del_out) + lambdaa * W2)
            grad_B2 = np.mean(del_out, axis=0, keepdims=True)

            grad_W1 = (1 / sample_size) * (data.T.dot(del_hidden) + lambdaa * W1)
            grad_B1 = np.mean(del_hidden, axis=0, keepdims=True)

            gradients = [grad_W2, grad_B2, grad_W1, grad_B1]

            return J, gradients

        def update_weights(We, data, params, learning_rate):

            J, gradients = aeCost(We, data, params)
            grad_W2, grad_B2, grad_W1, grad_B1 = gradients
            W1, B1, W2, B2 = We

            # Update weights

            W2 -= learning_rate * grad_W2
            B2 -= learning_rate * grad_B2

            W1 -= learning_rate * grad_W1
            B1 -= learning_rate * grad_B1

            We_updated = [W1, B1, W2, B2]
            return J, We_updated

        def initialize_weights(Lpre, Lhid):

            np.random.seed(8)

            Lpost = Lpre
            lim_1 = np.sqrt(6 / (Lpre + Lhid))
            lim_2 = np.sqrt(6 / (Lhid + Lpost))

            W1 = np.random.uniform(-lim_1, lim_1, (Lpre, Lhid))
            B1 = np.random.uniform(-lim_1, lim_1, (1, Lhid))

            W2 = np.random.uniform(-lim_2, lim_2, (Lhid, Lpost))
            B2 = np.random.uniform(-lim_2, lim_2, (1, Lpost))

            return W1, B1, W2, B2

        def train_network(data, params, learning_rate, batch_size, epoch):

            np.random.seed(8)

            sample_size = np.shape(data)[0]
            Lin, Lhid, lambdaa, beta, rho = params
            W1, B1, W2, B2 = initialize_weights(Lin, Lhid)

            We = [W1, B1, W2, B2]
            Loss = list()
            for i in range(epoch):
                if (i % 10 == 0):
                    print('Epoch: ' + str(i))
                    # Randomize the dataset for each iteration
                randomIndexes = np.random.permutation(sample_size)
                data = data[randomIndexes]

                number_of_batches = int(sample_size / batch_size)

                for j in range(number_of_batches):
                    # Mini batch start and end index
                    start = int(batch_size * j)
                    end = int(batch_size * (j + 1))

                    _, We = update_weights(We, data[start:end], params, learning_rate)

                J, _ = aeCost(We, data, params)
                Loss.append(J)

            return Loss, We

            # In[11]:

        data_final_flat = np.reshape(data_final, (np.shape(data_final)[0], 16 ** 2))

        Lin = Lpost = 16 ** 2
        Lhid = 64
        lambdaa = 5e-4
        beta = 0.01
        rho = 0.2
        params = [Lin, Lhid, lambdaa, beta, rho]

        # In[12]:

        loss, We_t = train_network(data_final_flat, params, 1e-2, 16, 80)

        # In[13]:

        figureNum += 1
        plt.figure(figureNum)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss (aeLoss) over Epochs')
        plt.plot(loss)
        plt.show()

        # In[14]:

        W1, B1, W2, B2 = We_t
        W2 = np.array(W2)
        W2 = W2.reshape(-1, 16, 16)

        figureNum += 1
        plt.figure(figureNum, figsize=(18, 16))

        for i in range(np.shape(W2)[0]):
            ax3 = plt.subplot(10, 8, i + 1)
            ax3.imshow(W2[i], cmap='gray')
            ax3.set_yticks([])
            ax3.set_xticks([])
        plt.show()

        # In[15]:

        sample_image = 92
        figureNum += 1
        plt.figure(figureNum)
        plt.imshow(data_final[sample_image], cmap='gray')
        plt.title('Original')
        plt.show(block=False)

        # In[16]:

        _, __, ___, reconstructed_sample_image = forward(We_t, data_final_flat[sample_image])
        figureNum += 1

        reconstructed_sample_image = np.array(reconstructed_sample_image)
        reconstructed_sample_image = reconstructed_sample_image.reshape(16, 16)
        plt.figure(figureNum)
        plt.imshow(reconstructed_sample_image, cmap='gray')
        plt.title('Reconstructed')
        plt.show(block=False)

        # In[17]:

        Lin_l = Lpost_l = 16 ** 2
        Lhid_l = 12
        lambdaa_l = 1e-2
        beta_l = 0.001
        rho_l = 0.2
        params_l = [Lin_l, Lhid_l, lambdaa_l, beta_l, rho_l]

        # In[18]:

        loss_l, We_l = train_network(data_final_flat, params_l, 1e-2, 32, 50)

        # In[19]:

        Lin_m = Lpost_m = 16 ** 2
        Lhid_m = 50
        lambdaa_m = 1e-2
        beta_m = 0.001
        rho_m = 0.2
        params_m = [Lin_m, Lhid_m, lambdaa_m, beta_m, rho_m]

        # In[20]:

        loss_m, We_m = train_network(data_final_flat, params_m, 1e-2, 32, 50)

        # In[21]:

        Lin_h = Lpost_h = 16 ** 2
        Lhid_h = 98
        lambdaa_h = 1e-2
        beta_h = 0.001
        rho_h = 0.2
        params_h = [Lin_h, Lhid_h, lambdaa_h, beta_h, rho_h]

        # In[22]:

        loss_h, We_h = train_network(data_final_flat, params_h, 1e-2, 32, 50)

        # In[23]:

        W1_l, B1_l, W2_l, B2_l = We_l
        W2_l = np.array(W2_l)
        W2_l = W2_l.reshape(-1, 16, 16)

        figureNum += 1
        plt.figure(figureNum, figsize=(18, 16))

        for i in range(np.shape(W2_l)[0]):
            ax3 = plt.subplot(10, 8, i + 1)
            ax3.imshow(W2_l[i], cmap='gray')
            ax3.set_yticks([])
            ax3.set_xticks([])
        plt.show()

        # In[24]:

        W1_m, B1_m, W2_m, B2_m = We_m
        W2_m = np.array(W2_m)
        W2_m = W2_m.reshape(-1, 16, 16)

        figureNum += 1
        plt.figure(figureNum, figsize=(18, 16))

        for i in range(np.shape(W2_m)[0]):
            ax3 = plt.subplot(10, 8, i + 1)
            ax3.imshow(W2_m[i], cmap='gray')
            ax3.set_yticks([])
            ax3.set_xticks([])
        plt.show()

        # In[25]:

        W1_h, B1_h, W2_h, B2_h = We_h
        W2_h = np.array(W2_h)
        W2_h = W2_h.reshape(-1, 16, 16)

        figureNum += 1
        plt.figure(figureNum, figsize=(18, 16))

        for i in range(np.shape(W2_h)[0]):
            ax3 = plt.subplot(10, 10, i + 1)
            ax3.imshow(W2_h[i], cmap='gray')
            ax3.set_yticks([])
            ax3.set_xticks([])
        plt.show()



    elif question == '3' :

        # !/usr/bin/env python
        # coding: utf-8

        # In[1]:

        import numpy as np
        import h5py
        import matplotlib.pyplot as plt
        import math
        import time

        # In[2]:

        # Part A

        f = h5py.File('assign3_data3.h5', 'r')
        dataKeys = list(f.keys())
        print('The data keys are:' + str(dataKeys))

        # Gathering the  train images, test images, train labels and test labels.
        train_data = f['trX']
        train_labels = f['trY']
        test_data = f['tstX']
        test_labels = f['tstY']

        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        print('The size of train data is: ' + str(np.shape(train_data)))
        print('The size of train labels is: ' + str(np.shape(train_labels)))
        print('The size of test_data is: ' + str(np.shape(test_data)))
        print('The size of test_labels is: ' + str(np.shape(test_labels)))

        # In[3]:

        def initialize_weights(fan_in, fan_out, wb_shape):

            np.random.seed(8)

            lim = np.sqrt(6 / (fan_in + fan_out))

            weight = np.random.uniform(-lim, lim, size=(wb_shape))

            return weight

        # In[6]:

        class RNN:

            def __init__(self, input_dim=3, hidden_dim=128, seq_len=150, learning_rate=1e-1,
                         momentumCoef=0.85, output_class=6, momentum_condition=False):

                np.random.seed(8)
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim

                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate

                self.momentumCoef = momentumCoef
                self.momentum_condition = momentum_condition
                self.last_t = 149

                # Weight initialization

                self.W1 = initialize_weights(self.input_dim, self.hidden_dim, (self.input_dim, self.hidden_dim))
                self.B1 = initialize_weights(self.input_dim, self.hidden_dim, (1, self.hidden_dim))

                self.W1_rec = initialize_weights(self.hidden_dim, self.hidden_dim, (self.hidden_dim, self.hidden_dim))

                self.W2 = initialize_weights(self.hidden_dim, self.output_class, (self.hidden_dim, self.output_class))
                self.B2 = initialize_weights(self.hidden_dim, self.output_class, (1, self.output_class))

                # momentum updates

                self.momentum_W1 = 0
                self.momentum_B1 = 0
                self.momentum_W1_rec = 0
                self.momentum_W2 = 0
                self.momentum_B2 = 0

            def accuracy(self, y, y_pred):
                '''
                MCE is the accuracy of our network. Mean classification error will be calculated to find accuracy.
                INPUTS:

                    y            : y is the labels for our data.
                    y_pred       : y_pred is the network's prediction.

                RETURNS:

                                 : returns the accuracy between y and y_pred.
                '''
                count = 0
                for i in range(len(y)):
                    if (y[i] == y_pred[i]):
                        count += 1
                N = np.shape(y)[0]

                return 100 * (count / N)

            def tanh(self, x):
                '''
                This function is the hyperbolic tangent for the activation functions of each neuron.
                INPUTS:

                    x            : x is the weighted sum which will be pushed to activation function.

                RETURNS:

                    result       : result is the hyperbolic tangent of the input x.
                '''

                result = 2 / (1 + np.exp(-2 * x)) - 1
                return result

            def sigmoid(self, x):

                '''
                This function is the sigmoid for the activation function.
                INPUTS:

                    x            : x is the weighted sum which will be pushed to activation function.

                RETURNS:

                    result       : result is the sigmoid of the input x.
                '''

                result = 1 / (1 + np.exp(-x))
                return result

            def der_sigmoid(self, x):
                '''
                This function is the derivative of sigmoid function.
                INPUTS:

                    x            : x is the input.

                RETURNS:

                    result       : result is the derivative of sigmoid of the input x.
                '''

                result = self.sigmoid(x) * (1 - self.sigmoid(x))
                return result

            def softmax(self, x):

                '''
                This function is the softmax for the activation function of output layer.
                INPUTS:

                    x            : x is the weighted sum which will be pushed to activation function.

                RETURNS:

                    result       : result is the softmax of the input x.
                '''

                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                result = e_x / np.sum(e_x, axis=-1, keepdims=True)
                return result

            def der_softmax(self, x):

                '''
                This function is the derivative of softmax.
                INPUTS:

                    x            : x is the input.

                RETURNS:

                    result       : result is the derivative of softmax of the input x.
                '''

                p = self.softmax(x)
                result = p * (1 - p)
                return result

            def CategoricalCrossEntropy(self, y, y_pred):

                '''
                cross_entropy is the loss function for the network.
                INPUTS:

                    y            : y is the labels for our data.
                    y_pred       : y_pred is the network's prediction.

                RETURNS:

                    cost         : cost is the cross entropy error between y and y_pred.
                '''

                # To avoid 0
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

                cost = -np.mean(y * np.log(y_pred + 1e-15))
                return cost

            def forward(self, data):

                data_state = dict()
                hidden_state = dict()
                output_state = dict()
                probabilities = dict()

                self.h_prev_state = np.zeros((1, self.hidden_dim))
                hidden_state[-1] = self.h_prev_state
                # Loop over time T = 150 :

                for t in range(self.seq_len):
                    data_state[t] = data[:, t]
                    # Recurrent hidden layer computations:

                    hidden_state[t] = self.tanh(
                        np.dot(data_state[t], self.W1) + np.dot(hidden_state[t - 1], self.W1_rec) + self.B1)
                    output_state[t] = np.dot(hidden_state[t], self.W2) + self.B2
                    # The probabilities per class

                    probabilities[t] = self.softmax(output_state[t])

                cache = [data_state, hidden_state, probabilities]
                return cache

            def BPTT(self, data, Y):

                cache = self.forward(data)

                data_state, hidden_state, probs = cache

                dW1, dW1_rec, dW2 = np.zeros((np.shape(self.W1))), np.zeros((np.shape(self.W1_rec))), np.zeros(
                    (np.shape(self.W2)))
                dB1, dB2 = np.zeros((np.shape(self.B1))), np.zeros((np.shape(self.B2)))
                dhnext = np.zeros((np.shape(hidden_state[0])))

                dy = probs[self.last_t]
                dy[np.arange(len(Y)), np.argmax(Y, 1)] -= 1
                dB2 += np.sum(dy, axis=0, keepdims=True)

                dW2 += np.dot(hidden_state[self.last_t].T, dy)

                for t in reversed(range(1, self.seq_len)):
                    dh = np.dot(dy, self.W2.T) + dhnext
                    dh_rec = (1 - (hidden_state[t] * hidden_state[t])) * dh
                    dB1 += np.sum(dh_rec, axis=0, keepdims=True)
                    dW1 += np.dot(data_state[t].T, dh_rec)
                    dW1_rec += np.dot(hidden_state[t - 1].T, dh_rec)
                    dhnext = np.dot(dh_rec, self.W1_rec.T)

                grads = [dW1, dB1, dW1_rec, dW2, dB2]

                for grad in grads:
                    np.clip(grad, -10, 10, out=grad)

                return grads, cache

            def update_weights(self, data, Y):

                grads, cache = self.BPTT(data, Y)
                dW1, dB1, dW1_rec, dW2, dB2 = grads
                sample_size = np.shape(cache)[0]
                # If momentum is used.
                if (self.momentum_condition == True):

                    self.momentum_W1 = dW1 + (self.momentumCoef * self.momentum_W1)
                    self.momentum_B1 = dB1 + (self.momentumCoef * self.momentum_B1)
                    self.momentum_W1_rec = dW1_rec + (self.momentumCoef * self.momentum_W1_rec)
                    self.momentum_W2 = dW2 + (self.momentumCoef * self.momentum_W2)
                    self.momentum_B2 = dB2 + (self.momentumCoef * self.momentum_B2)

                    self.W1 -= self.learning_rate * self.momentum_W1 / sample_size
                    self.B1 -= self.learning_rate * self.momentum_B1 / sample_size
                    self.W1_rec -= self.learning_rate * self.momentum_W1_rec / sample_size
                    self.W2 -= self.learning_rate * self.momentum_W2 / sample_size
                    self.B2 -= self.learning_rate * self.momentum_B2 / sample_size

                # If momentum is not used.
                else:

                    self.W1 -= self.learning_rate * dW1 / sample_size
                    self.B1 -= self.learning_rate * dB1 / sample_size
                    self.W1_rec -= self.learning_rate * dW1_rec / sample_size
                    self.W2 -= self.learning_rate * dW2 / sample_size
                    self.B2 -= self.learning_rate * dB2 / sample_size

                return cache

            def train_network(self, data, labels, test_data, test_labels, epochs=50, batch_size=32):

                np.random.seed(8)

                valid_loss = list()
                valid_accuracy = list()

                test_loss = list()
                test_accuracy = list()

                sample_size = np.shape(data)[0]
                k = int(sample_size / 10)

                for i in range(epochs):
                    start_time = time.time()
                    print('Epoch : ' + str(i))
                    randomIndexes = np.random.permutation(sample_size)
                    data = data[randomIndexes]

                    number_of_batches = int(sample_size / batch_size)
                    for j in range(number_of_batches):
                        start = int(batch_size * j)
                        end = int(batch_size * (j + 1))

                        data_feed = data[start:end]
                        labels_feed = labels[start:end]

                        cache_train = self.update_weights(data_feed, labels_feed)

                    valid_data = data[0:k]
                    valid_labels = labels[0:k]

                    probs_valid, predictions_valid = self.predict(valid_data)

                    cross_loss_valid = self.CategoricalCrossEntropy(valid_labels, probs_valid[self.last_t])
                    acc_valid = self.accuracy(np.argmax(valid_labels, 1), predictions_valid)

                    probs_test, predictions_test = self.predict(test_data)

                    cross_loss_test = self.CategoricalCrossEntropy(test_labels, probs_test[self.last_t])
                    acc_test = self.accuracy(np.argmax(test_labels, 1), predictions_test)

                    valid_loss.append(cross_loss_valid)
                    valid_accuracy.append(acc_valid)

                    test_loss.append(cross_loss_test)
                    test_accuracy.append(acc_test)

                    end_time = time.time()
                    print('Training time for 1 epoch : ' + str(end_time - start_time))
                valid_loss = np.array(valid_loss)
                valid_accuracy = np.array(valid_accuracy)

                test_loss = np.array(test_loss)
                test_accuracy = np.array(test_accuracy)

                return valid_loss, valid_accuracy, test_loss, test_accuracy

            def predict(self, X):

                cache = self.forward(X)
                probabilities = cache[-1]
                result = np.argmax(probabilities[self.last_t], axis=1)
                return probabilities, result

        # In[7]:

        RNN_model = RNN(input_dim=3, hidden_dim=128, learning_rate=1e-12, momentumCoef=0.85,
                        output_class=6, momentum_condition=True)

        valid_loss, valid_accuracy, test_loss, test_accuracy = RNN_model.train_network(train_data, train_labels,
                                                                                       test_data,
                                                                                       test_labels, epochs=27,
                                                                                       batch_size=32)

        # In[67]:

        figureNum = 0

        plt.figure(figureNum)
        plt.plot(valid_loss)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Cross Entropy for Validation Data over Epochs')

        plt.show()

        # In[14]:

        def confusion_matrix(labels, y_pred):
            labels_ = np.argmax(labels, 1)
            result = np.zeros((6, 6))

            for i in range(len(labels_)):
                lab_i = labels_[i]
                y_pred_i = y_pred[i]
                result[lab_i, y_pred_i] += 1

            return result

        # In[15]:

        def accuracy_(confusion_matrix):
            accuracy = 0
            all_sum = 0
            for i in range(np.shape(confusion_matrix)[0]):
                for j in range(np.shape(confusion_matrix)[1]):
                    all_sum += confusion_matrix[i, j]
                    if (i == j):
                        accuracy += confusion_matrix[i, j]

            return accuracy / all_sum * 100

        # In[16]:

        _, train_preds = RNN_model.predict(train_data)
        _, test_preds = RNN_model.predict(test_data)

        confusion_mat_train = confusion_matrix(train_labels, train_preds)

        confusion_mat_test = confusion_matrix(test_labels, test_preds)

        # In[17]:

        accuracy_RNN_train = accuracy_(confusion_mat_train)
        print('Accuracy of RNN with train data : ' + str(accuracy_RNN_train))

        # In[18]:

        accuracy_RNN_test = accuracy_(confusion_mat_test)
        print('Accuracy of RNN with test data : ' + str(accuracy_RNN_test))

        # In[21]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')
        print('The confusion matrix for the training data : \n \n' + str(confusion_mat_train))

        # In[20]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix for the test data : \n \n' + str(confusion_mat_test))

        # In[22]:

        class LSTM():

            def __init__(self, input_dim=3, hidden_dim=100, output_class=6, seq_len=150,
                         batch_size=30, learning_rate=1e-1, momentumCoef=0.85, momentum_condition=False):

                np.random.seed(150)

                self.input_dim = input_dim
                self.hidden_dim = hidden_dim

                # Unfold case T = 150 :
                self.seq_len = seq_len
                self.output_class = output_class
                self.learning_rate = learning_rate
                self.batch_size = batch_size
                self.momentumCoef = momentumCoef
                self.momentum_condition = momentum_condition
                self.input_stack_dim = self.input_dim + self.hidden_dim
                self.last_t = 149
                # Weight initialization

                self.W_f = initialize_weights(self.input_dim, self.hidden_dim, (self.input_stack_dim, self.hidden_dim))
                self.B_f = initialize_weights(self.input_dim, self.hidden_dim, (1, self.hidden_dim))
                self.W_i = initialize_weights(self.input_dim, self.hidden_dim, (self.input_stack_dim, self.hidden_dim))
                self.B_i = initialize_weights(self.input_dim, self.hidden_dim, (1, self.hidden_dim))
                self.W_c = initialize_weights(self.input_dim, self.hidden_dim, (self.input_stack_dim, self.hidden_dim))
                self.B_c = initialize_weights(self.input_dim, self.hidden_dim, (1, self.hidden_dim))
                self.W_o = initialize_weights(self.input_dim, self.hidden_dim, (self.input_stack_dim, self.hidden_dim))
                self.B_o = initialize_weights(self.input_dim, self.hidden_dim, (1, self.hidden_dim))

                self.W = initialize_weights(self.hidden_dim, self.output_class, (self.hidden_dim, self.output_class))
                self.B = initialize_weights(self.hidden_dim, self.output_class, (1, self.output_class))

                # To keep previous updates in momentum :
                self.momentum_W_f = 0
                self.momentum_B_f = 0
                self.momentum_W_i = 0
                self.momentum_B_i = 0
                self.momentum_W_c = 0
                self.momentum_B_c = 0
                self.momentum_W_o = 0
                self.momentum_B_o = 0
                self.momentum_W = 0
                self.momentum_B = 0

            def accuracy(self, y, y_pred):
                '''
                MCE is the accuracy of our network. Mean classification error will be calculated to find accuracy.
                INPUTS:

                    y            : y is the labels for our data.
                    y_pred       : y_pred is the network's prediction.

                RETURNS:

                                 : returns the accuracy between y and y_pred.
                '''
                count = 0
                for i in range(len(y)):
                    if (y[i] == y_pred[i]):
                        count += 1
                N = np.shape(y)[0]

                return 100 * (count / N)

            def tanh(self, x):
                '''
                This function is the hyperbolic tangent for the activation functions of each neuron.
                INPUTS:

                    x            : x is the weighted sum which will be pushed to activation function.

                RETURNS:

                    result       : result is the hyperbolic tangent of the input x.
                '''

                result = 2 / (1 + np.exp(-2 * x)) - 1
                return result

            def der_tanh(self, x):
                '''
                This function is the derivative hyperbolic tangent. This function will be used in backpropagation.
                INPUTS:

                    x            : x is the input.

                RETURNS:

                    result       : result is the derivative of hyperbolic tangent of the input x.
                '''
                result = 1 - self.tanh(x) ** 2
                return result

            def sigmoid(self, x):

                '''
                This function is the sigmoid for the activation function.
                INPUTS:

                    x            : x is the weighted sum which will be pushed to activation function.

                RETURNS:

                    result       : result is the sigmoid of the input x.
                '''

                result = 1 / (1 + np.exp(-x))
                return result

            def der_sigmoid(self, x):
                '''
                This function is the derivative of sigmoid function.
                INPUTS:

                    x            : x is the input.

                RETURNS:

                    result       : result is the derivative of sigmoid of the input x.
                '''

                result = self.sigmoid(x) * (1 - self.sigmoid(x))
                return result

            def softmax(self, x):

                '''
                This function is the softmax for the activation function of output layer.
                INPUTS:

                    x            : x is the weighted sum which will be pushed to activation function.

                RETURNS:

                    result       : result is the softmax of the input x.
                '''

                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                result = e_x / np.sum(e_x, axis=-1, keepdims=True)
                return result

            def der_softmax(self, x):

                '''
                This function is the derivative of softmax.
                INPUTS:

                    x            : x is the input.

                RETURNS:

                    result       : result is the derivative of softmax of the input x.
                '''

                p = self.softmax(x)
                result = p * (1 - p)
                return result

            def CategoricalCrossEntropy(self, y, y_pred):

                '''
                cross_entropy is the loss function for the network.
                INPUTS:

                    y            : y is the labels for our data.
                    y_pred       : y_pred is the network's prediction.

                RETURNS:

                    cost         : cost is the cross entropy error between y and y_pred.
                '''

                # To avoid 0
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

                cost = -np.mean(y * np.log(y_pred + 1e-15))
                return cost

            def cell_forward(self, X, h_prev, C_prev):

                # print(X.shape,h_prev.shape)
                # Stacking previous hidden state vector with inputs:
                stack = np.column_stack([X, h_prev])

                # Forget gate:
                forget_gate = self.sigmoid(np.dot(stack, self.W_f) + self.B_f)

                # İnput gate:
                input_gate = self.sigmoid(np.dot(stack, self.W_i) + self.B_i)

                # New candidate:
                cell_bar = self.tanh(np.dot(stack, self.W_c) + self.B_c)

                # New Cell state:
                cell_state = forget_gate * C_prev + input_gate * cell_bar

                # Output fate:
                output_gate = self.sigmoid(np.dot(stack, self.W_o) + self.B_o)

                # Hidden state:
                hidden_state = output_gate * self.tanh(cell_state)

                # Classifiers (Softmax) :
                dense = np.dot(hidden_state, self.W) + self.B
                probs = self.softmax(dense)

                cache = [stack, forget_gate, input_gate, cell_bar, cell_state, output_gate, hidden_state, dense, probs]
                return cache

            def forward(self, X, h_prev, C_prev):
                x_s, z_s, f_s, i_s = dict(), dict(), dict(), dict()
                C_bar_s, C_s, o_s, h_s = dict(), dict(), dict(), dict()
                v_s, y_s = dict(), dict()

                h_s[-1] = h_prev
                C_s[-1] = C_prev

                for t in range(150):
                    x_s[t] = X[:, t, :]
                    cache = self.cell_forward(x_s[t], h_s[t - 1], C_s[t - 1])
                    z_s[t], f_s[t], i_s[t], C_bar_s[t], C_s[t], o_s[t], h_s[t], v_s[t], y_s[t] = cache

                result_cache = [z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s, v_s, y_s]
                return result_cache

            def BPTT(self, cache, Y):

                z_s, f_s, i_s, C_bar_s, C_s, o_s, h_s, v_s, y_s = cache

                dW_f = np.zeros((np.shape(self.W_f)))
                dW_i = np.zeros((np.shape(self.W_i)))
                dW_c = np.zeros((np.shape(self.W_c)))
                dW_o = np.zeros((np.shape(self.W_o)))
                dW = np.zeros((np.shape(self.W)))

                dB_f = np.zeros((np.shape(self.B_f)))
                dB_i = np.zeros((np.shape(self.B_i)))
                dB_c = np.zeros((np.shape(self.B_c)))
                dB_o = np.zeros((np.shape(self.B_o)))
                dB = np.zeros((np.shape(self.B)))

                dh_next = np.zeros(np.shape(h_s[0]))
                dC_next = np.zeros(np.shape(C_s[0]))

                # w.r.t. softmax input
                ddense = y_s[self.last_t]
                ddense[np.arange(len(Y)), np.argmax(Y, 1)] -= 1

                # Softmax classifier's :

                dW = np.dot(h_s[149].T, ddense)
                dB = np.sum(ddense, axis=0, keepdims=True)
                # Backprop through time:

                for t in reversed(range(1, 150)):
                    C_prev = C_s[t - 1]

                    # Output gate :
                    dh = np.dot(ddense, self.W.T) + dh_next
                    do = dh * self.tanh(C_s[t])
                    do = do * self.der_sigmoid(o_s[t])
                    dW_o += np.dot(z_s[t].T, do)
                    dB_o += np.sum(do, axis=0, keepdims=True)

                    # Cell state:
                    dC = dC_next
                    dC += dh * o_s[t] * self.der_tanh(C_s[t])
                    dC_bar = dC * i_s[t]
                    dC_bar = dC_bar * self.der_tanh(C_bar_s[t])
                    dW_c += np.dot(z_s[t].T, dC_bar)
                    dB_c += np.sum(dC_bar, axis=0, keepdims=True)

                    # Input gate:
                    di = dC * C_bar_s[t]
                    di = self.der_sigmoid(i_s[t]) * di
                    dW_i += np.dot(z_s[t].T, di)
                    dB_i += np.sum(di, axis=0, keepdims=True)

                    # Forget gate:
                    df = dC * C_prev
                    df = df * self.der_sigmoid(f_s[t])
                    dW_f += np.dot(z_s[t].T, df)
                    dB_f += np.sum(df, axis=0, keepdims=True)
                    dz = np.dot(df, self.W_f.T) + np.dot(di, self.W_i.T) + np.dot(dC_bar, self.W_c.T) + np.dot(do,
                                                                                                               self.W_o.T)
                    dh_next = dz[:, -self.hidden_dim:]
                    dC_next = f_s[t] * dC

                # List of gradients :
                grads = [dW, dB, dW_o, dB_o, dW_c, dB_c, dW_i, dB_i, dW_f, dB_f]

                # Clipping gradients anyway
                for grad in grads:
                    np.clip(grad, -15, 15, out=grad)

                return h_s[self.last_t], C_s[self.last_t], grads

            def train_network(self, data, labels, test_data, test_labels, epochs=50):

                valid_loss = list()
                valid_accuracy = list()

                test_loss = list()
                test_accuracy = list()

                sample_size = np.shape(data)[0]
                k = int(sample_size / 10)

                for epoch in range(epochs):

                    print('Epoch : ' + str(epoch))

                    randomIndexes = np.random.permutation(sample_size)
                    data = data[randomIndexes]

                    h_prev, C_prev = np.zeros((self.batch_size, self.hidden_dim)), np.zeros(
                        (self.batch_size, self.hidden_dim))

                    start_time = time.time()
                    number_of_batches = int(sample_size / self.batch_size)
                    for i in range(number_of_batches):
                        start = int(self.batch_size * i)
                        end = int(self.batch_size * (i + 1))

                        # Feeding random indexes:
                        data_feed = data[start:end]
                        labels_feed = labels[start:end]

                        # Forward + BPTT + SGD:
                        cache_train = self.forward(data_feed, h_prev, C_prev)
                        h, c, grads = self.BPTT(cache_train, labels_feed)

                        self.update_weights(grads)

                        # Hidden state -------> Previous hidden state
                        # Cell state ---------> Previous cell state
                        h_prev, C_prev = h, c

                    end_time = time.time()
                    print('Training time for 1 epoch : ' + str(end_time - start_time))

                    valid_data = data[0:k]
                    valid_labels = labels[0:k]

                    # Validation metrics calculations:

                    valid_prevs = np.zeros((valid_data.shape[0], self.hidden_dim))

                    valid_cache = self.forward(valid_data, valid_prevs, valid_prevs)
                    probs_valid = valid_cache[-1]

                    cross_loss_valid = self.CategoricalCrossEntropy(valid_labels, probs_valid[self.last_t])

                    # Test metrics calculations:
                    test_prevs = np.zeros((test_data.shape[0], self.hidden_dim))

                    test_cache = self.forward(test_data, test_prevs, test_prevs)
                    probs_test = test_cache[-1]

                    cross_loss_test = self.CategoricalCrossEntropy(test_labels, probs_test[self.last_t])
                    predictions_test = np.argmax(probs_test[self.last_t], 1)
                    acc_test = self.accuracy(np.argmax(test_labels, 1), predictions_test)

                    valid_loss.append(cross_loss_valid)
                    test_loss.append(cross_loss_test)

                    test_accuracy.append(acc_test)

                return valid_loss, test_loss, test_accuracy

            def update_weights(self, grads):

                dW, dB, dW_o, dB_o, dW_c, dB_c, dW_i, dB_i, dW_f, dB_f = grads

                # If momentum is used.
                if (self.momentum_condition == True):

                    self.momentum_W_f = dW_f + (self.momentumCoef * self.momentum_W_f)
                    self.momentum_B_f = dB_f + (self.momentumCoef * self.momentum_B_f)
                    self.momentum_W_i = dW_i + (self.momentumCoef * self.momentum_W_i)
                    self.momentum_B_i = dB_i + (self.momentumCoef * self.momentum_B_i)
                    self.momentum_W_c = dW_c + (self.momentumCoef * self.momentum_W_c)
                    self.momentum_B_c = dB_c + (self.momentumCoef * self.momentum_B_c)
                    self.momentum_W_o = dW_o + (self.momentumCoef * self.momentum_W_o)
                    self.momentum_B_o = dB_o + (self.momentumCoef * self.momentum_B_o)
                    self.momentum_W = dW + (self.momentumCoef * self.momentum_W)
                    self.momentum_B = dB + (self.momentumCoef * self.momentum_B)

                    self.W_f -= self.learning_rate * self.momentum_W_f
                    self.B_f -= self.learning_rate * self.momentum_B_f
                    self.W_i -= self.learning_rate * self.momentum_W_i
                    self.B_i -= self.learning_rate * self.momentum_B_i
                    self.W_c -= self.learning_rate * self.momentum_W_c
                    self.B_c -= self.learning_rate * self.momentum_B_c
                    self.W_o -= self.learning_rate * self.momentum_W_o
                    self.B_o -= self.learning_rate * self.momentum_B_o
                    self.W -= self.learning_rate * self.momentum_W
                    self.B -= self.learning_rate * self.momentum_B

                    # If momentum is not used.
                else:

                    self.W_f -= self.learning_rate * dW_f
                    self.B_f -= self.learning_rate * dB_f
                    self.W_i -= self.learning_rate * dW_i
                    self.B_i -= self.learning_rate * dB_i
                    self.W_c -= self.learning_rate * dW_c
                    self.B_c -= self.learning_rate * dB_c
                    self.W_o -= self.learning_rate * dW_o
                    self.B_o -= self.learning_rate * dB_o
                    self.W -= self.learning_rate * dW
                    self.B -= self.learning_rate * dB

            def predict(self, X):

                # Give zeros to hidden/cell states:
                pasts = np.zeros((np.shape(X)[0], self.hidden_dim))

                result_cache = self.forward(X, pasts, pasts)
                probabilities = result_cache[-1]
                result_prob = np.argmax(probabilities[self.last_t], axis=1)

                return result_prob

        # In[37]:

        LSTM_model = LSTM(learning_rate=1e-15, momentumCoef=0.85, batch_size=32, hidden_dim=128,
                          momentum_condition=True)

        valid_loss_lstm, test_loss_lstm, test_accuracy_lstm = LSTM_model.train_network(train_data, train_labels,
                                                                                       test_data, test_labels,
                                                                                       epochs=10)

        # In[66]:

        figureNum += 1

        plt.figure(figureNum)
        plt.plot(valid_loss_lstm)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss of Validation Data')
        plt.show()

        # In[41]:

        train_preds_lstm = LSTM_model.predict(train_data)
        test_preds_lstm = LSTM_model.predict(test_data)
        confusion_mat_train_lstm = confusion_matrix(train_labels, train_preds_lstm)
        confusion_mat_test_lstm = confusion_matrix(test_labels, test_preds_lstm)

        # In[74]:

        accuracy_LSTM_train = accuracy_(confusion_mat_train_lstm)
        print('Accuracy of LSTM with train data : ' + str(accuracy_LSTM_train))

        # In[75]:

        accuracy_LSTM_test = accuracy_(confusion_mat_test_lstm)
        print('Accuracy of LSTM with test data : ' + str(accuracy_LSTM_test))

        # In[44]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix(LSTM) for the training data : \n \n' + str(confusion_mat_train_lstm))

        # In[45]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix(LSTM) for the test data : \n \n' + str(confusion_mat_test_lstm))

        # In[52]:

        # Tensorflow

        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # In[54]:

        RNN = keras.Sequential()
        RNN.add(layers.SimpleRNN(128, batch_input_shape=[30, 150, 3]))
        RNN.add(layers.Dense(6, activation='softmax'))
        optimizer_RNN = keras.optimizers.SGD(learning_rate=0.1, momentum=0.85)
        RNN.compile(loss='categorical_crossentropy',
                    optimizer=optimizer_RNN,
                    metrics=['accuracy'])

        # In[55]:

        model_RNN = RNN.fit(train_data, train_labels, batch_size=30, epochs=15)

        # In[56]:

        train_preds_RNN = RNN.predict_classes(train_data, batch_size=30)
        test_preds_RNN = RNN.predict_classes(test_data, batch_size=30)

        confusion_mat_train_RNN = confusion_matrix(train_labels, train_preds_RNN)

        confusion_mat_test_RNN = confusion_matrix(test_labels, test_preds_RNN)

        # In[76]:

        accuracy_RNN_train_tf = accuracy_(confusion_mat_train_RNN)
        print('Accuracy of RNN with train data : ' + str(accuracy_RNN_train_tf))

        # In[77]:

        accuracy_RNN_test_tf = accuracy_(confusion_mat_test_RNN)
        print('Accuracy of RNN with test data : ' + str(accuracy_RNN_test_tf))

        # In[78]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix(RNN) for the training data : \n \n' + str(confusion_mat_train_RNN))

        # In[79]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix(RNN) for the test data : \n \n' + str(confusion_mat_test_RNN))

        # In[59]:

        LSTM = keras.Sequential()
        LSTM.add(layers.LSTM(128, batch_input_shape=[30, 150, 3]))
        LSTM.add(layers.Dense(6, activation='softmax'))
        optimizer_LSTM = keras.optimizers.SGD(learning_rate=0.1, momentum=0.85)
        LSTM.compile(loss='categorical_crossentropy',
                     optimizer=optimizer_LSTM,
                     metrics=['accuracy'])

        # In[60]:

        model_LSTM = LSTM.fit(train_data, train_labels, batch_size=30, epochs=15)

        # In[61]:

        train_preds_LSTM = LSTM.predict_classes(train_data, batch_size=30)
        test_preds_LSTM = LSTM.predict_classes(test_data, batch_size=30)

        confusion_mat_train_LSTM = confusion_matrix(train_labels, train_preds_LSTM)

        confusion_mat_test_LSTM = confusion_matrix(test_labels, test_preds_LSTM)

        # In[82]:

        accuracy_LSTM_train_tf = accuracy_(confusion_mat_train_LSTM)
        print('Accuracy of LSTM with train data : ' + str(accuracy_LSTM_train_tf))

        # In[83]:

        accuracy_LSTM_test_tf = accuracy_(confusion_mat_test_LSTM)
        print('Accuracy of LSTM with test data : ' + str(accuracy_LSTM_test_tf))

        # In[84]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix(LSTM) for the training data : \n \n' + str(confusion_mat_train_LSTM))

        # In[85]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix(LSTM) for the test data : \n \n' + str(confusion_mat_test_LSTM))

        # In[68]:

        GRU = keras.Sequential()
        GRU.add(layers.GRU(128, batch_input_shape=[30, 150, 3]))
        GRU.add(layers.Dense(6, activation='softmax'))
        optimizer_GRU = keras.optimizers.SGD(learning_rate=0.1, momentum=0.85)
        GRU.compile(loss='categorical_crossentropy',
                    optimizer=optimizer_GRU,
                    metrics=['accuracy'])

        model_GRU = GRU.fit(train_data, train_labels, batch_size=30, epochs=10)

        # In[69]:

        train_preds_GRU = GRU.predict_classes(train_data, batch_size=30)
        test_preds_GRU = GRU.predict_classes(test_data, batch_size=30)

        confusion_mat_train_GRU = confusion_matrix(train_labels, train_preds_GRU)

        confusion_mat_test_GRU = confusion_matrix(test_labels, test_preds_GRU)

        # In[86]:

        accuracy_GRU_train_tf = accuracy_(confusion_mat_train_GRU)
        print('Accuracy of GRU with train data : ' + str(accuracy_GRU_train_tf))

        # In[87]:

        accuracy_GRU_test_tf = accuracy_(confusion_mat_test_GRU)
        print('Accuracy of GRU with test data : ' + str(accuracy_GRU_test_tf))

        # In[88]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix(GRU) for the training data : \n \n' + str(confusion_mat_train_GRU))

        # In[89]:

        print('Columns are : PREDICTION \n')
        print('Rows are : ACTUAL \n')

        print('The confusion matrix(GRU) for the test : \n \n' + str(confusion_mat_test_GRU))

        # In[ ]:


berkan_ozdamar_21602353_hw3(question)



