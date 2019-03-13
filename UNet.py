#!/usr/bin/env python3

'''
Holds the UNet model as described in paper: "Anime Sketch Colowing with Swish-Gated Residual U-Net"

'''


class UNet:
    def __init__(self):
        self.model = build_graph()

    def summary(self):
        print(self.model.summary())

    def predict(self, X):
        return self.model.predict(X)


    @staticmethod
    def build_graph():
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

        conv1_1 = Conv2D_LRELU(filters=96, kernel_size=(3, 3), inputs=inputs) 
        conv1_2 = Conv2D_LRELU(filters=96, kernel_size=(3, 3), inputs=conv1_1)
        max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) (conv1_2) 

        swish1_2 = Swish(cropping=((0,1),(0,1)), inputs=conv1_1)
        inputs2  = concatenate([max_pool1, swish1_2])

        conv2_1 = Conv2D_LRELU(filters=192, kernel_size=(1, 1), inputs=inputs2) 
        conv2_2 = Conv2D_LRELU(filters=192, kernel_size=(3, 3), inputs=conv2_1)
        conv2_3 = Conv2D_LRELU(filters=192, kernel_size=(3, 3), inputs=conv2_2) 
        max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) (conv2_3)

        swish2_3 = Swish(cropping=((0,1),(0,1)), inputs=conv2_1)
        inputs3  = concatenate([max_pool2, swish2_3])

        conv3_1 = Conv2D_LRELU(filters=288, kernel_size=(1, 1), inputs=inputs3) 
        conv3_2 = Conv2D_LRELU(filters=288, kernel_size=(3, 3), inputs=conv3_1) 
        conv3_3 = Conv2D_LRELU(filters=288, kernel_size=(3, 3), inputs=conv3_2) 
        max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) (conv3_3)

        swish3_4 = Swish(cropping=((0,1),(0,1)), inputs=conv3_1)
        inputs4  = concatenate([max_pool3, swish3_4])

        conv4_1 = Conv2D_LRELU(filters=384, kernel_size=(1, 1), inputs=inputs4) 
        conv4_2 = Conv2D_LRELU(filters=384, kernel_size=(3, 3), inputs=conv4_1) 
        conv4_3 = Conv2D_LRELU(filters=384, kernel_size=(3, 3), inputs=conv4_2) 
        max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) (conv4_3)


        swish4_5 = Swish(cropping=((0,1),(0,1)), inputs=conv4_1)
        inputs5  = concatenate([max_pool4, swish4_5])

        conv5_1 = Conv2D_LRELU(filters=480, kernel_size=(1, 1), inputs=inputs5) 
        conv5_2 = Conv2D_LRELU(filters=480, kernel_size=(3, 3), inputs=conv5_1) 
        conv5_3 = Conv2D_LRELU(filters=480, kernel_size=(3, 3), inputs=conv5_2) 
        max_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2,2)) (conv5_3)

        swish5_6 = Swish(cropping=((0,1),(0,1)), inputs=conv5_1)
        inputs6  = concatenate([max_pool5, swish5_6])

        conv6_1 = Conv2D_LRELU(filters=512, kernel_size=(1, 1), inputs=inputs6) 
        conv6_2 = Conv2D_LRELU(filters=512, kernel_size=(3, 3), inputs=conv6_1) 
        conv6_3 = Conv2D_LRELU(filters=512, kernel_size=(3, 3), inputs=conv6_2) 
        deconv6 = Conv2DTranspose_LRELU(filters=512, kernel_size=(2, 2), strides=(2,2), inputs=conv6_3) 

        swish6_5 = Swish(cropping=((0,1),(0,1)), inputs=conv6_1)
        swish5_5 = Swish(cropping=((0,1),(0,1)), inputs=conv5_3)
        inputs5_up = concatenate([swish6_5, swish5_5, deconv6])


        conv5_1_up = Conv2D_LRELU(filters=480, kernel_size=(1, 1), inputs=inputs5_up)
        conv5_2_up = Conv2D_LRELU(filters=480, kernel_size=(3, 3), inputs=conv5_1_up) 
        conv5_3_up = Conv2D_LRELU(filters=480, kernel_size=(3, 3), inputs=conv5_2_up) 
        deconv5 = Conv2DTranspose_LRELU(filters=480, kernel_size=(2, 2), strides=(2,2), inputs=conv5_3_up) 

        swish5_4 = Swish(cropping=((0,1),(0,1)), inputs=conv5_1_up)
        swish4_4 = Swish(cropping=((0,1),(0,1)), inputs=conv4_3)
        inputs4_up = concatenate([swish5_4, siwsh4_4, deconv5])

        conv4_1_up = Conv2D_LRELU(filters=384, kernel_size=(1, 1), inputs=inputs4_up) 
        conv4_2_up = Conv2D_LRELU(filters=384, kernel_size=(3, 3), inputs=conv4_1_up) 
        conv4_3_up = Conv2D_LRELU(filters=384, kernel_size=(3, 3), inputs=conv4_2_up) 
        deconv4 = Conv2DTranspose_LRELU(filters=384, kernel_size=(2, 2), strides=(2,2), inputs=conv4_3_up) 

        swish4_3 = Swish(cropping=((0,1),(0,1)), inputs=conv4_1_up)
        swish3_3 = Swish(cropping=((0,1),(0,1)), inputs=conv3_3)
        inputs3_up = concatenate([swish4_3, swish3_3, deconv4])


        conv3_1_up = Conv2D_LRELU(filters=288, kernel_size=(1, 1), inputs=inputs3_up) 
        conv3_2_up = Conv2D_LRELU(filters=288, kernel_size=(3, 3), inputs=conv3_1_up) 
        conv3_3_up = Conv2D_LRELU(filters=288, kernel_size=(3, 3), inputs=conv3_2_up) 
        deconv3 = Conv2DTranspose_LRELU(filters=288, kernel_size=(2, 2), strides=(2,2), inputs=conv3_3_up) 

        swish3_2 = Swish(cropping=((0,1),(0,1)), inputs=conv3_1_up) 
        swish2_2 = Swish(cropping=((0,1),(0,1)), inputs=conv2_3_up)
        inputs2_up = concatenate([siwsh3_3,swish2_2, deconv3])


        conv2_1_up = Conv2D_LRELU(filters=192, kernel_size=(1, 1), inputs=inputs2_up) 
        conv2_2_up = Conv2D_LRELU(filters=192, kernel_size=(3, 3), inputs=conv2_1_up) 
        conv2_3_up = Conv2D_LRELU(filters=192, kernel_size=(3, 3), inputs=conv2_2_up) 
        deconv2 = Conv2DTranspose_LRELU(filters=192, kernel_size=(2, 2), strides=(2,2), inputs=conv2_3_up) 


        swish2_1 = Swish(cropping=((0,1),(0,1)), inputs=conv2_1_up)
        swish1_1 = Swish(cropping=((0,1),(0,1)), inputs=conv1_3)
        inputs1_up = concatenate([swish2_1, swish1_1, deconv2])

        conv1_1_up = Conv2D_LRELU(filters=96, kernel_size=(1, 1), inputs=inputs1_up)
        conv1_2_up = Conv2D_LRELU(filters=96, kernel_size=(3, 3), inputs=conv1_1_up) 
        conv1_3_up = Conv2D_LRELU(filters=96, kernel_size=(3, 3), inputs=conv1_2_up) 
        conv1_4_up = Conv2D(filters=27, kernel_size=(1, 1), activation=None, padding = 'same', kernel_initializer='he_normal') (conv1_3_up)