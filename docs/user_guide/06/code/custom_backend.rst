
.. code::

  def custom_backend(filename, values, stations, origin,
      add_station_labels=True, centered=True, label=''):

      import pygmt

      fig = pygmt.Figure()

      gmt_region = "g-155/-145/59/64"
      gmt_projection = "D-150/61.5/60/63/9c"
      gmt_frame = ["xa5", "ya2"]

      fig.basemap(
          region=gmt_region,
          projection=gmt_projection,
          frame=["xa5", "ya2"],
          )

      fig.coast(
          land="grey80",
          shorelines=True,
          area_thresh=100,
          )

      # construct color palette
      cmap = "polar"

      if centered:
          limits = (-np.max(np.abs(values)), +np.max(np.abs(values)))
      else:
          limits = (np.min(values), np.max(values))

      pygmt.makecpt(
          cmap=cmap,
          series=limits,
          )

      # plot stations
      for _i, station in enumerate(stations):
          fig.plot(
              x=station.longitude, y=station.latitude,
              cmap=True, color="+z", zvalue=values[_i],
              style="t.5c", pen="1p",
              )

      # plot line segments
      for _i, station in enumerate(stations):
          fig.plot(
              x=(origin.longitude, station.longitude),
              y=(origin.latitude, station.latitude),
              cmap=True, zvalue=values[_i],
              pen="thick,+z,-"
              )

      # plot origin
      fig.plot(
          x=origin.longitude, y=origin.latitude,
          style="a.5c", color="black", pen="1p"
          )

      # add colorbar
      fig.colorbar(frame=["x+l%s" % label])

      fig.savefig(filename)

