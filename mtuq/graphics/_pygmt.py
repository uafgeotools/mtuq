
import numpy as np


def exists_pygmt():
    try:
        import pygmt
        return True
    except:
        return False


def plot_force_pygmt(filename, force_dict):

    import pygmt
    fig = pygmt.Figure()

    lat = np.degrees(np.pi/2 - np.arccos(force_dict['h']))
    lon = _wrap(force_dict['phi'] + 90.)

    proj_arg="A0/0/6i"
    area_arg="-180/180/-90/90"

    # specify basemap
    fig.basemap(projection=proj_arg, region=area_arg, frame=['xg180','yg30'])

    # plot arcs
    fig.text(x=90./2., y=0., text='E', font='40p')
    fig.plot(x=[90./2.,90./2.], y=[90.,7.5], pen='1.5p,0/0/0/35')
    fig.plot(x=[90./2.,90./2.], y=[-90.,-7.5], pen='1.5p,0/0/0/35')
    #fig.text(x=0., y=0., text='S', font='40p')
    fig.plot(x=[0.,0.], y=[90.,7.5], pen='1.5p,0/0/0/35')
    fig.plot(x=[0.,0.], y=[-90.,-7.5], pen='1.5p,0/0/0/35')
    fig.text(x=-90./2., y=0., text='W', font='40p')
    fig.plot(x=[-90./2.,-90./2.], y=[90.,7.5], pen='1.5p,0/0/0/35')
    fig.plot(x=[-90./2.,-90./2.], y=[-90.,-7.5], pen='1.5p,0/0/0/35')

    # plot force orientation
    fig.plot(x=lon/2., y=lat, style="d40p", pen="1p,black", fill="black")

    fig.savefig(filename)


def _wrap(angle_in_deg):
    """ Wraps angle to (-180, 180)
    """
    angle_in_deg %= 360.
    if angle_in_deg > 180.:
        angle_in_deg -= 360.
    return angle_in_deg

