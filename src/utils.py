import matplotlib.pyplot as plt

def visualize_depth_map(samples, test=False, model=None):
    input, target = samples
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    if test:
        pred = model.predict(input)
        fig, ax = plt.subplots(6, 3, figsize=(50, 50))
        for i in range(6):
            ax[i, 0].imshow(input[i].squeeze().cpu().numpy())
            ax[i, 1].imshow(target[i].squeeze().cpu().numpy(), cmap=cmap)
            ax[i, 2].imshow(pred[i].squeeze(), cmap=cmap)
    else:
        fig, ax = plt.subplots(6, 2, figsize=(50, 50))
        for i in range(6):
            ax[i, 0].imshow(input[i].permute(1, 2, 0).cpu().numpy())  # RGB image
            ax[i, 1].imshow(target[i].squeeze().cpu().numpy(), cmap=cmap)  # Depth map
    # plt.show()
    fig.savefig("visualization.png", bbox_inches='tight', dpi=300)
