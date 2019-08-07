

import numpy as np


def change(M, i1=None, i2=None):
    """ Converts from one basis convention to another

      Convention 1: up-south-east (GCMT) (www.globalcmt.org)
        1: up (r), 2: south (theta), 3: east (phi)
     
      Convention 2: Aki and Richards (1980, p. 114-115, 118)
        also Jost and Herrman (1989, Fig. 1)
        1: north, 2: east, 3: down
     
      Convention 3: Stein and Wysession (2003, p. 218)
        also TapeTape2012a "A geometric setting for moment tensors" (p.478)
        also several Kanamori codes
        1: north, 2: west, 3: up
      
      Convention 4: 
        1: east, 2: north, 3: up
      
      Convention 5: TapeTape2013 "The classical model for moment tensors" (p.1704)
        1: south, 2: east, 3: up
    """

    if i1 not in [1,2,3,4,5]:
        raise ValueError

    if i2 not in [1,2,3,4,5]:
        raise ValueError

    # check input array
    assert M.shape == (6,)

    # initialize output array
    Mout = np.empty(6) * np.nan

    if i1==i2:
        Mout = M

    elif (i1,i2) == (1,2):
        # up-south-east (GCMT) to north-east-down (AkiRichards 1980, p.118)
        Mout[0] = M[1]
        Mout[1] = M[2]
        Mout[2] = M[0]
        Mout[3] = -M[5]
        Mout[4] = M[3]
        Mout[5] = -M[4]
    elif (i1,i2) == (1,3):
        # up-south-east (GCMT) to north-west-up (/opt/seismo-util/bin/faultpar2cmtsol.pl)
        Mout[0] = M[1]
        Mout[1] = M[2]
        Mout[2] = M[0]
        Mout[3] = M[5]
        Mout[4] = -M[3]
        Mout[5] = -M[4]
    elif (i1,i2) == (1,4):
        # up-south-east (GCMT) to east-north-up
        Mout[0] = M[2]
        Mout[1] = M[1]
        Mout[2] = M[0]
        Mout[3] = -M[5]
        Mout[4] = M[4]
        Mout[5] = -M[3]
    elif (i1,i2) == (1,5):
        # up-south-east (GCMT) to south-east-up
        Mout[0] = M[1]
        Mout[1] = M[2]
        Mout[2] = M[0]
        Mout[3] = M[5]
        Mout[4] = M[3]
        Mout[5] = M[4]  

    elif (i1,i2) == (2,1):
        # north-east-down (AkiRichards) to up-south-east (GCMT) (AR, 1980, p. 118)
        Mout[0] = M[2]
        Mout[1] = M[0]
        Mout[2] = M[1]
        Mout[3] = M[4]
        Mout[4] = -M[5]
        Mout[5] = -M[3]
    elif (i1,i2) == (2,3):
        # north-east-down (AkiRichards) to north-west-up
        Mout[0] = M[0]
        Mout[1] = M[1]
        Mout[2] = M[2]
        Mout[3] = -M[3]
        Mout[4] = -M[4]
        Mout[5] = M[5]   
    elif (i1,i2) == (2,4):
        # north-east-down (AkiRichards) to east-north-up
        Mout[0] = M[1]
        Mout[1] = M[0]
        Mout[2] = M[2]
        Mout[3] = M[3]
        Mout[4] = -M[5]
        Mout[5] = -M[4]
    elif (i1,i2) == (2,5):
        # north-east-down (AkiRichards) to south-east-up
        Mout[0] = M[0]
        Mout[1] = M[1]
        Mout[2] = M[2]
        Mout[3] = -M[3]
        Mout[4] = M[4]
        Mout[5] = -M[5]   

    elif (i1,i2)==(3,1):
        # north-west-up to up-south-east (GCMT)
        Mout[0] = M[2]
        Mout[1] = M[0]
        Mout[2] = M[1]
        Mout[3] = -M[4]
        Mout[4] = -M[5]
        Mout[5] = M[3]
    elif (i1,i2)==(3,2):
        # north-west-up to north-east-down (AkiRichards)
        Mout[0] = M[0]
        Mout[1] = M[1]
        Mout[2] = M[2]
        Mout[3] = -M[3]
        Mout[4] = -M[4]
        Mout[5] = M[5] 
    elif (i1,i2)==(3,4):
        # north-west-up to east-north-up
        Mout[0] = M[1]
        Mout[1] = M[0]
        Mout[2] = M[2]
        Mout[3] = -M[3]
        Mout[4] = -M[5]
        Mout[5] = M[4] 
    elif (i1,i2)==(3,5):
        # north-west-up to south-east-up
        Mout[0] = M[0]
        Mout[1] = M[1]
        Mout[2] = M[2]
        Mout[3] = M[3]
        Mout[4] = -M[4]
        Mout[5] = -M[5] 

    elif (i1,i2)==(4,1):
        # east-north-up to up-south-east (GCMT)
        Mout[0] = M[2]
        Mout[1] = M[1]
        Mout[2] = M[0]
        Mout[3] = -M[5]
        Mout[4] = M[4]
        Mout[5] = -M[3]
    elif (i1,i2)==(4,2):
        # east-north-up to north-east-down (AkiRichards)
        Mout[0] = M[1]
        Mout[1] = M[0]
        Mout[2] = M[2]
        Mout[3] = M[3]
        Mout[4] = -M[5]
        Mout[5] = -M[4]
    elif (i1,i2)==(4,3):
        # east-north-up to north-west-up
        Mout[0] = M[1]
        Mout[1] = M[0]
        Mout[2] = M[2]
        Mout[3] = -M[3]
        Mout[4] = M[5]
        Mout[5] = -M[4] 
    elif (i1,i2)==(4,5):
        # east-north-up to south-east-up
        Mout[0] = M[1]
        Mout[1] = M[0]
        Mout[2] = M[2]
        Mout[3] = -M[3]
        Mout[4] = -M[5]
        Mout[5] = M[4] 

    elif (i1,i2)==(5,1):
        # south-east-up to up-south-east (GCMT)
        Mout[0] = M[2]
        Mout[1] = M[0]
        Mout[2] = M[1]
        Mout[3] = M[4]
        Mout[4] = M[5]
        Mout[5] = M[3]
    elif (i1,i2)==(5,2):
        # south-east-up to north-east-down (AkiRichards)
        Mout[0] = M[0]
        Mout[1] = M[1]
        Mout[2] = M[2]
        Mout[3] = -M[3]
        Mout[4] = M[4]
        Mout[5] = -M[5]
    elif (i1,i2)==(5,3):
        # south-east-up to north-west-up
        Mout[0] = M[0]
        Mout[1] = M[1]
        Mout[2] = M[2]
        Mout[3] = M[3]
        Mout[4] = -M[4]
        Mout[5] = -M[5]
    elif (i1,i2)==(5,4):
        # south-east-up to east-north-up
        Mout[0] = M[1]
        Mout[1] = M[0]
        Mout[2] = M[2]
        Mout[3] = -M[3]
        Mout[4] = M[5]
        Mout[5] = -M[4] 

    return Mout



def _check(code):
    if code in [0, 1, 2, 3, 4 ,5]:
        return code
    elif code in [0., 1., 2., 3., 4., 5.]:
        return int(code)
    elif code=="Unknown":
        return 0
    elif code=="USE":
        return 1
    elif code=="NED":
        return 2
    elif code=="NWU":
        return 3
    else:
        raise TypeError


