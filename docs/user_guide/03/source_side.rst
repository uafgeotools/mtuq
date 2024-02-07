
.. warning::

   This page is currently just as stub. 

   To fill in missing documentation, feel free to submit a pull request.


Notes on source-side 3D Green's functions
=========================================

**File format**

- Currently, MTUQ reads source-side 3D Green's functions from SAC files

- SAC output format is natively supported by SPECFEM3D_GLOBE, but not SPECFEM3D Cartesian (output from the latter can be manually converted)


**Units convention**

- MTUQ treats each individual SAC file as the response in meters to a unit (1 Newton-meter) force couple

- Users must ensure that source-side Green's functions are normalized according to the above units


**Basis convention**

- For moment tensors, MTUQ uses an `up-south-east` convention, identical to one the used by SPECFEM3D_GLOBE


**Origin time convention**

- For origin time, MTUQ uses a centroid convention (`more details <https://github.com/uafgeotools/mtuq/issues/140>`_), so that `t=0` in the `GreensTensor` time discretization corresponds to mean source excitation time

- MTUQ uses the `REF_TIME` header from the SPECFEM3D_GLOBE SAC output files, which gives the peak excitation of the source relative to the simulation start time

- MTUQ ignores the origin time given in the CMTSOLUTION file and `ORIGIN_TIME` header


**Depth searches (experimental)**

- Only depth searches are possible with source-side 3D Green's functions (no other hypocenter parameters)

- The current `depth search <https://github.com/uafgeotools/mtuq/blob/568e49a73817e4e2dbab1189210214da6906266f/mtuq/io/clients/SPECFEM3D_SAC.py#L117>`_ implementation is somewhat crude and experimental; consider local modifcations to suit your needs

- To allow depth searches, create subdirectories for each centroid depth, as below

.. code ::

  {event_id}/
      {depth_in_km}/
          {net}.{sta}.{loc}.Z.Mrr.sac
          {net}.{sta}.{loc}.Z.Mtt.sac
          {net}.{sta}.{loc}.Z.Mpp.sac
          {net}.{sta}.{loc}.Z.Mrt.sac
          {net}.{sta}.{loc}.Z.Mrp.sac
          {net}.{sta}.{loc}.Z.Mtp.sac 
          {net}.{sta}.{loc}.R.Mrr.sac
          {net}.{sta}.{loc}.R.Mtt.sac
          {net}.{sta}.{loc}.R.Mpp.sac
          {net}.{sta}.{loc}.R.Mrt.sac
          {net}.{sta}.{loc}.R.Mrp.sac
          {net}.{sta}.{loc}.R.Mtp.sac
          {net}.{sta}.{loc}.T.Mrr.sac
          {net}.{sta}.{loc}.T.Mtt.sac
          {net}.{sta}.{loc}.T.Mpp.sac
          {net}.{sta}.{loc}.T.Mrt.sac
          {net}.{sta}.{loc}.T.Mrp.sac
          {net}.{sta}.{loc}.T.Mtp.sac
      {depth_in_km}/
          {net}.{sta}.{loc}.Z.Mrr.sac
          {net}.{sta}.{loc}.Z.Mtt.sac
          {net}.{sta}.{loc}.Z.Mpp.sac
          {net}.{sta}.{loc}.Z.Mrt.sac
          {net}.{sta}.{loc}.Z.Mrp.sac
          {net}.{sta}.{loc}.Z.Mtp.sac 
          {net}.{sta}.{loc}.R.Mrr.sac
          {net}.{sta}.{loc}.R.Mtt.sac
          {net}.{sta}.{loc}.R.Mpp.sac
          {net}.{sta}.{loc}.R.Mrt.sac
          {net}.{sta}.{loc}.R.Mrp.sac
          {net}.{sta}.{loc}.R.Mtp.sac
          {net}.{sta}.{loc}.T.Mrr.sac
          {net}.{sta}.{loc}.T.Mtt.sac
          {net}.{sta}.{loc}.T.Mpp.sac
          {net}.{sta}.{loc}.T.Mrt.sac
          {net}.{sta}.{loc}.T.Mrp.sac
          {net}.{sta}.{loc}.T.Mtp.sac
      ...


Working example
---------------

A working of example using source-side 3D Green's functions from SPECFEM3D_GLOBE in a moment tensor inversion:

`test_greens_SPECFEM3D_SAC.py <https://github.com/rmodrak/mtuq/blob/master/tests/test_greens_SPECFEM3D_SAC.py>`_

