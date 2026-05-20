"""MWS-internal DDS message types for explicit SHM sensor transports.

Both the Isaac publisher (dds_bridge.py) and the bridge subscriber
(docker/bridge/bridge.py) import this file so the typename strings stay in
sync between the two sides.

Placement: sim/isaac/mws_msgs.py
  - Isaac container: PYTHONPATH includes /workspace/mws-dimos so
    `from sim.isaac.mws_msgs import CameraSignal_, LidarSignal_` works.
  - Bridge container: Dockerfile copies this file to /mws_msgs.py, and
    bridge.py prepends "/" to sys.path, so `from mws_msgs import ...` works
    without any additional packaging.
"""
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class CameraSignal_(idl.IdlStruct, typename="mws_bridge.msg.dds_.CameraSignal_"):
    """Tiny frame-ready signal on DDS topic rt/camera/signal.

    Published after the frame has been written to the shared-memory segment
    /run/mws/camera.rgb. The bridge reads this signal, then mmap-reads the
    frame directly from SHM, eliminating 921 KB of CDR serialization per frame.

    Fields:
        tick:   sim time in milliseconds mod 2³²
        seq:    monotonically increasing frame counter (wraps at 2³²)
        width:  frame width in pixels
        height: frame height in pixels
        step:   bytes per row (= width * 3 for rgb8)
    """

    tick: types.uint32
    seq: types.uint32
    width: types.uint32
    height: types.uint32
    step: types.uint32


@dataclass
@annotate.final
@annotate.autoid("sequential")
class LidarSignal_(idl.IdlStruct, typename="mws_bridge.msg.dds_.LidarSignal_"):
    """Tiny point-cloud-ready signal on DDS topic rt/lidar/signal.

    Published after packed XYZI float32 point bytes have been written to the
    shared-memory segment /run/mws/lidar.xyzi. The bridge reads the signal,
    then mmap-reads the payload from SHM and rehydrates it into LCM
    ``sensor_msgs.PointCloud2`` without paying DDS serialization cost for the
    variable-length byte payload.

    Fields:
        tick:      sim time in milliseconds mod 2³²
        seq:       monotonically increasing point-cloud counter (wraps at 2³²)
        width:     point count
        height:    cloud height (currently always 1)
        point_step: bytes per point (currently always 16 for XYZI float32)
        row_step:  bytes per row (= width * point_step)
        data_size: payload bytes written to SHM
    """

    tick: types.uint32
    seq: types.uint32
    width: types.uint32
    height: types.uint32
    point_step: types.uint32
    row_step: types.uint32
    data_size: types.uint32
