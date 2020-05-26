import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def createAnimation(data):
    t, r, c = data.shape

    fig, axes = plt.subplots()

    def init():
        axes.imshow(data[0])

    def update(i):
        axes.imshow(data[i])

    ani = FuncAnimation(fig, update, frames=t, init_func=init, interval=200)

    return ani


def displayAnimationInNotebook(animation):
    HTML(animation.to_jshtml())


def saveAnimation(animation, target):
    animation.save(target, 'imagemagick', fps=5)