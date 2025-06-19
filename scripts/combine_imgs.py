import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def combine_images(image_files, nrows, ncols, title, output='combined.png'):
    if len(image_files) != nrows * ncols:
        print(f"Error: Number of images ({len(image_files)}) does not match grid size ({nrows}x{ncols})")
        sys.exit(1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3), gridspec_kw={'wspace': 0, 'hspace': 0})
    # fig.suptitle(title, fontsize=16)

    # Flatten axes for easy iteration, even if 1D
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for ax, img_file in zip(axes, image_files):
        img = Image.open(img_file).resize((224, 224))
        ax.imshow(img)
        ax.axis('off')
        # ax.set_title(img_file)

    # Hide any unused axes if image count < nrows*ncols
    for ax in axes[len(image_files):]:
        ax.axis('off')


    plt.savefig(output, bbox_inches='tight', pad_inches=0)
    print(f"Combined image saved as {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine images into a grid figure with a title.")
    parser.add_argument('images', nargs='+', help='List of image files to combine')
    parser.add_argument('--rows', type=int, required=True, help='Number of rows in the grid')
    parser.add_argument('--cols', type=int, required=True, help='Number of columns in the grid')
    parser.add_argument('--title', type=str, required=True, help='Title for the figure')
    parser.add_argument('--output', type=str, default='combined.png', help='Output filename')
    args = parser.parse_args()

    combine_images(args.images, args.rows, args.cols, args.title, args.output)
