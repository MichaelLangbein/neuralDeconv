import keras as k



def createSimpleAutoencoder(imgWidth, imgHeight, nodes=[700, 500, 300, 100]):
    obsSize = imgWidth * imgHeight
    sparseSize = nodes[-1]

    input = k.Input([obsSize])
    x = input
    for n in nodes:
        x = k.layers.Dense(n, activation='sigmoid', name=f"encoder_d{n}")(x)
    encoder = k.Model(input, x, name='encoder')

    decInput = k.Input([sparseSize])
    y = decInput
    for n in nodes[::-1]:
        y = k.layers.Dense(n, activation='sigmoid', name=f"decoder_d{n}")(y)
    y = k.layers.Dense(obsSize, activation='sigmoid', name=f"decoder_d{n+1}")(y)
    decoder = k.Model(decInput, y, name='decoder')

    autoencoder = k.Model(input, decoder(encoder(input)), name='autoencoder')

    return encoder, decoder, autoencoder



def createConv2Autoencoder(imgHeight, imgWidth, channels, nodes=[16, 8, 8]):
    
    input = k.Input(shape=(imgHeight, imgWidth, channels))

    x = input
    for i, n in enumerate(nodes):
        x = k.layers.Conv2D(n, (3, 3), activation='relu', padding='same', name=f"encoder_conv{i}")(x)
        x = k.layers.MaxPooling2D((2, 2), padding='same', name=f"encoder_maxp{i}")(x)

    y = x
    for i, n in enumerate(nodes[::-1]):
        y = k.layers.Conv2D(8, (3, 3), activation='relu', padding='same', name=f"decoder_conv{i}")(y)
        y = k.layers.UpSampling2D((2, 2), name=f"decoder_upspl{i}")(y)
    y = k.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name=f"decoder_conv{i+1}")(y)

    autoencoder = k.Model(input, y)

    return autoencoder