import numpy as np

import matplotlib.pyplot as plt

#import matplotlib.image as img
import PIL.Image as Image 

import math
import cmath

import time

import csv

from numpy import binary_repr


class DCT(object):
    """
    This class DCT implements all the procedures for transforming a given 2D digital image
    into its corresponding frequency-domain image (Forward DCT Transform)
    """
    
    @classmethod
    def __computeSinglePoint2DCT(self, imge, u, v, N):
        """
        A private method that computes a single value of the 2D-DCT from a given image.

        Parameters
        ----------
        imge : ndarray
            The input image.
        
        u : ndarray
            The index in x-dimension.
            
        v : ndarray
            The index in y-dimension.

        N : int
            Size of the image.
            
        Returns
        -------
        result : float
            The computed single value of the DCT.
        """
        result = 0

        for x in range(N):
            for y in range(N):
                result += imge[x, y] * math.cos(((2*x + 1)*u*math.pi)/(2*N)) * math.cos(((2*y + 1)*v*math.pi)/(2*N))

        #Add the tau value to the result
        if (u==0) and (v==0):
            result = result/N
        elif (u==0) or (v==0):
            result = (math.sqrt(2.0)*result)/N
        else:
            result = (2.0*result)/N

        return result
    
    @classmethod
    def __computeSinglePointInverse2DCT(self, imge, x, y, N):
        """
        A private method that computes a single value of the 2D-DCT from a given image.

        Parameters
        ----------
        imge : ndarray
            The input image.
        
        u : ndarray
            The index in x-dimension.
            
        v : ndarray
            The index in y-dimension.

        N : int
            Size of the image.
            
        Returns
        -------
        result : float
            The computed single value of the DCT.
        """
        result = 0

        for u in range(N):
            for v in range(N):
                if (u==0) and (v==0):
                    tau = 1.0/N
                elif (u==0) or (v==0):
                    tau = math.sqrt(2.0)/N
                else:
                    tau = 2.0/N            
                result += tau * imge[u, v] * math.cos(((2*x + 1)*u*math.pi)/(2*N)) * math.cos(((2*y + 1)*v*math.pi)/(2*N))

        return result
    
    @classmethod
    def computeForward2DDCT(self, imge):
        """
        Computes/generates the 2D DCT of an input image in spatial domain.

        Parameters
        ----------
        imge : ndarray
            The input image to be transformed.

        Returns
        -------
        final2DDFT : ndarray
            The transformed image.
        """
 
        # Assuming a square image
        N = imge.shape[0]
        final2DDCT = np.zeros([N, N], dtype=float)
        for u in range(N):
            for v in range(N):
                #Compute the DCT value for each cells/points in the resulting transformed image.
                final2DDCT[u, v] = DCT.__computeSinglePoint2DCT(imge, u, v, N)
        return final2DDCT
    
    @classmethod
    def computeInverse2DDCT(self, imge):
        """
        Computes/generates the 2D DCT of an input image in spatial domain.

        Parameters
        ----------
        imge : ndarray
            The input image to be transformed.

        Returns
        -------
        final2DDFT : ndarray
            The transformed image.
        """
 
        # Assuming a square image
        N = imge.shape[0]
        finalInverse2DDCT = np.zeros([N, N], dtype=float)
        for x in range(N):
            for y in range(N):
                #Compute the DCT value for each cells/points in the resulting transformed image.
                finalInverse2DDCT[x, y] = DCT.__computeSinglePointInverse2DCT(imge, x, y, N)
        return finalInverse2DDCT
    
    @classmethod
    def normalize2DDCTByLog(self, dctImge):
        """
        Computes the log transformation of the transformed DCT image to make the range
        of the DCT values b/n 0 to 255
        
        Parameters
        ----------
        dctImge : ndarray
            The input DCT transformed image.

        Returns
        -------
        dctNormImge : ndarray
            The normalized version of the transformed image.
        """
        
        #Normalize the DCT values of a transformed image:
        dctImge = np.absolute(dctImge)
        dctNormImge = (255/ math.log10(255)) * np.log10(1 + (255/(np.max(dctImge))*dctImge))
        
        return dctNormImge


#Read an image file
imgeLena = Image.open("small_lemon.jpeg") # open an image

#Convert the image file to a matrix
imgeLena = np.array(imgeLena)

#Convert the uint datatype of the matrix values into 'int' for using the negative values
imgeLena = imgeLena.astype('int')

#Display the image:
print("Image Size:", imgeLena.shape)
plt.imshow(imgeLena, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
plt.show()

dctImgeLena = DCT.computeForward2DDCT(imgeLena)

plt.imshow(dctImgeLena, cmap=plt.get_cmap('gray'))

plt.show()
inverseImge = np.round(DCT.computeInverse2DDCT(dctImgeLena))
plt.imshow(inverseImge, cmap=plt.get_cmap('gray'))
plt.show()

