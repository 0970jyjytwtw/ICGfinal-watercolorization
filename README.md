# ICGfinal_watercolorization

This repo basically implement :

M. Wang *et al*., "Towards Photo Watercolorization with Artistic Verisimilitude," in *IEEE Transactions on Visualization and Computer Graphics*, vol. 20, no. 10, pp. 1451-1460, Oct. 2014, doi: 10.1109/TVCG.2014.2303984.

Some steps are not same as the paper, such as saliency map and segmentation .... , in order to simplified the project. Some effects (Color Adjustment and Granulation) aren't applied in this projects, because I think it is not good to apply them.  And I comment out those effect.

I also add a Watercolor Paper Texture effect to simulate the texture of the paper. 

"colorlib.txt" is referenced from "https://github.com/devin6011/ICGproject-watercolorization". This file is used in Color Adjustment.

## Usage
* Replace "image_name" and "image_path" with your source image.
* Replace "texture_path" with your texture image.

## Packages
Python 3.8.10

* opencv                    4.5.2
* pillow                    8.0.0
* numpy                     1.20.3





