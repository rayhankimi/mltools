from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
import numpy as np
import random
import os


def view_many_image(target_dir, target_class, times):
    images = []
    for time in range(times):
        target_folder = target_dir + target_class
        random_image = random.sample(os.listdir(target_folder), 1)
        img = mpimg.imread(target_folder + "/" + random_image[0])

        plt.imshow(img)
        plt.title(target_class)
        plt.axis("off")
        plt.show()

        print(f"Image Shape is: {img.shape}")
        images.append(img)
    return images


def view_random_image(target_dir, target_class):
    target_folder = target_dir + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    img = mpimg.imread(target_folder + "/" + random_image[0])

    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")

    print(f"Image Shape is: {img.shape}")
    return img


def plot_many_image(target_dir, target_class1, target_class2, times):
    plt.figure(figsize=(15, times * 5))  # Adjust figure size as needed

    for i in range(times):
        # Plot class 1 images
        plt.subplot(times, 2, 2 * i + 1)
        target_folder1 = os.path.join(target_dir, target_class1)
        random_image1 = random.sample(os.listdir(target_folder1), 1)
        img1 = mpimg.imread(os.path.join(target_folder1, random_image1[0]))
        plt.imshow(img1)
        plt.title(target_class1)
        plt.axis("off")

        # Plot class 2 images
        plt.subplot(times, 2, 2 * i + 2)
        target_folder2 = os.path.join(target_dir, target_class2)
        random_image2 = random.sample(os.listdir(target_folder2), 1)
        img2 = mpimg.imread(os.path.join(target_folder2, random_image2[0]))
        plt.imshow(img2)
        plt.title(target_class2)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    x_in = np.c_[
        xx.ravel(), yy.ravel()]  # .C itu maksudnya biar kalau misal ada 2x3 agar jadi 6x1 (Stack 2D array menjadi satu)

    # Prediksi
    y_pred = model.predict(x_in)

    # Cek multiclass

    if len(y_pred[0]) > 1:
        print("Multiclass")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("Binary Classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
    # Bikin Confusion Matrix
    cm = confusion_matrix(y_true, y_pred) / 10
    cm_normalize = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # Perbagus
    fig, ax = plt.subplots(figsize=(10, 7))

    # Bikin matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Greens)
    fig.colorbar(cax)

    if classes:  # Kalau berkelas, maka labelnya adalah kelas kelas
        labels = classes
    else:  # Kalau cuma binary, maka labelnya pada shape 0 (0 dan 1)
        labels = np.arange(cm.shape[0])

    # Beri label ke axes
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Table",
        ylabel="True Label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels
    )

    # Set x axis label ke bawah
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Bikin treshold pakai warna berbeda

    threshold = (cm.max() + cm.min()) / 2.

    # Plot text setiap cell

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"({cm[i, j]:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)
