# dct-tiv

Discrete Cosine Transform Terminal Image Viewer

An experiment to see if the discrete cosine transform could be used to improve character selection for use in rendering images to the terminal. it seems to be kinda worse than just comparing pixels but maybe it just needs some tweaking... uses oklab colour space to sample the colours which is an improvement over other terminal image viewers, although the output is still sometimes a bit lacking. Can be used as a library
 or standalone binary (todo: arg parsing).